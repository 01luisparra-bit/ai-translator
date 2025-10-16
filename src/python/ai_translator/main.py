#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Translator (Python) — resilient edition
- PDF -> images (multiple backends)
- OCR via Tesseract
- Optional translation via OpenAI (if OPENAI_API_KEY set)
- Health checks, retries, timeouts, circuit breakers
- Graceful degradation + offline mode (cached results)
- Metrics dashboard in ./metrics.json

Requirements:
  pip install pytesseract pdf2image pillow requests
System deps (preferred but optional):
  - Tesseract OCR (tesseract binary in PATH)
  - Poppler (pdftoppm) OR ImageMagick ("magick")
Folder layout expected:
  project/
    pdfs/                <-- put PDFs here
    cache/               <-- auto-created
    output/              <-- auto-created (final .txt/.json)
    src/main.py          <-- this file
Env:
  OPENAI_API_KEY=sk-...  (optional; if absent => no translation)
"""

from __future__ import annotations
import os
import sys
import io
import re
import json
import time
import math
import socket
import shutil
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable

# ---------- Optional third-party modules (we degrade gracefully) ----------
try:
    import pytesseract  # OCR engine (Python wrapper)
    from PIL import Image  # Pillow
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    # Best path for rasterizing PDFs if Poppler is installed
    from pdf2image import convert_from_path  # type: ignore
    PDF2IMAGE_OK = True
except Exception:
    PDF2IMAGE_OK = False

try:
    import requests  # for translation + health checks
    REQUESTS_OK = True
except Exception:
    REQUESTS_OK = False


# ---------- Config ----------
ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "pdfs"          # <--- put your PDFs here
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"
METRICS_FILE = ROOT / "metrics.json"
TESSDATA_HINTS = [
    ROOT / "eng.traineddata",
    ROOT / "spa.traineddata",
    # add more traineddata files alongside project root if you have them
]
LANGS_FOR_OCR = "eng+spa"  # You can adjust, e.g. "eng" or "spa" only

# Timeouts (seconds)
SUBPROCESS_TIMEOUT = 90
HTTP_TIMEOUT = 20

# Retry defaults
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 1.6

# Circuit breaker defaults
CB_FAIL_THRESHOLD = 3        # failures before tripping
CB_COOLDOWN_SECONDS = 60     # stay open this long before half-open


# ---------- Helpers ----------
def log(msg: str) -> None:
    print(msg, flush=True)


def now_ts() -> float:
    return time.time()


def hash_path(p: Path) -> str:
    return hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:12]


def ensure_dirs() -> None:
    for d in (CACHE_DIR, OUTPUT_DIR, PDF_DIR):
        d.mkdir(parents=True, exist_ok=True)


def read_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def internet_up(host="8.8.8.8", port=53, timeout=2) -> bool:
    """Very small internet connectivity check."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# ---------- Metrics Dashboard ----------
@dataclass
class Metrics:
    started_at: float = field(default_factory=now_ts)
    last_run_at: float = 0.0
    pdfs_processed: int = 0
    images_generated: int = 0
    ocr_calls: int = 0
    ocr_failures: int = 0
    translate_calls: int = 0
    translate_failures: int = 0
    offline_mode_used: bool = False
    circuit_breakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "last_run_at": self.last_run_at,
            "pdfs_processed": self.pdfs_processed,
            "images_generated": self.images_generated,
            "ocr_calls": self.ocr_calls,
            "ocr_failures": self.ocr_failures,
            "translate_calls": self.translate_calls,
            "translate_failures": self.translate_failures,
            "offline_mode_used": self.offline_mode_used,
            "circuit_breakers": self.circuit_breakers,
        }


def load_metrics() -> Metrics:
    if METRICS_FILE.exists():
        try:
            data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            m = Metrics(**data)
            return m
        except Exception:
            pass
    return Metrics()


def save_metrics(m: Metrics) -> None:
    m.last_run_at = now_ts()
    METRICS_FILE.write_text(json.dumps(m.to_dict(), indent=2), encoding="utf-8")


METRICS = load_metrics()


# ---------- Circuit Breaker ----------
class CircuitBreaker:
    def __init__(self, name: str, threshold: int = CB_FAIL_THRESHOLD, cooldown: int = CB_COOLDOWN_SECONDS):
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self.fail_count = 0
        self.state = "CLOSED"   # CLOSED -> OPEN -> HALF_OPEN -> CLOSED
        self.opened_at = 0.0

        # load persisted state if exists
        state = METRICS.circuit_breakers.get(name)
        if state:
            self.fail_count = state.get("fail_count", 0)
            self.state = state.get("state", "CLOSED")
            self.opened_at = state.get("opened_at", 0.0)

    def _persist(self):
        METRICS.circuit_breakers[self.name] = {
            "fail_count": self.fail_count,
            "state": self.state,
            "opened_at": self.opened_at,
            "threshold": self.threshold,
            "cooldown": self.cooldown,
        }

    def allow(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            # after cooldown, move to HALF_OPEN
            if now_ts() - self.opened_at >= self.cooldown:
                self.state = "HALF_OPEN"
                self._persist()
                log(f"⚠️  CircuitBreaker[{self.name}] moving to HALF_OPEN, trial allowed.")
                return True
            # remain open
            return False
        if self.state == "HALF_OPEN":
            # allow a single trial attempt
            return True
        return False

    def success(self):
        if self.state in ("HALF_OPEN", "OPEN"):
            log(f"✅ CircuitBreaker[{self.name}] CLOSED after successful trial.")
        self.state = "CLOSED"
        self.fail_count = 0
        self.opened_at = 0.0
        self._persist()

    def failure(self):
        self.fail_count += 1
        if self.fail_count >= self.threshold and self.state != "OPEN":
            self.state = "OPEN"
            self.opened_at = now_ts()
            self._persist()
            log(f"🚨 CircuitBreaker[{self.name}] OPEN (dashboard light ON). Cooldown {self.cooldown}s.")
        else:
            # still closed or half-open but failed
            self._persist()


CB_OCR = CircuitBreaker("ocr")
CB_TRANSLATE = CircuitBreaker("translate")


# ---------- Retry wrapper ----------
def with_retries(
    tries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    def deco(fn: Callable):
        def wrapped(*args, **kwargs):
            delay = 0.0
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= tries:
                        raise
                    delay = delay * backoff + 0.5 if delay else 0.5
                    if on_retry:
                        on_retry(attempt, e)
                    time.sleep(delay)
        return wrapped
    return deco


# ---------- Health checks ----------
def check_tesseract() -> bool:
    path = shutil.which("tesseract")
    ok = path is not None
    if ok:
        log(f"🔍 Health: tesseract found at {path}")
    else:
        log("❌ Health: tesseract not found in PATH (OCR will be limited / skipped).")
    return ok


def check_poppler() -> bool:
    path = shutil.which("pdftoppm")
    if path:
        log(f"🔍 Health: Poppler (pdftoppm) found at {path}")
        return True
    log("ℹ️  Health: Poppler not found; will try ImageMagick or other fallbacks.")
    return False


def check_imagemagick() -> bool:
    path = shutil.which("magick")
    if path:
        log(f"🔍 Health: ImageMagick found at {path}")
        return True
    log("ℹ️  Health: ImageMagick not found; will try other fallbacks.")
    return False


def check_internet() -> bool:
    ok = internet_up()
    log(f"🔍 Health: Internet connectivity = {'OK' if ok else 'DOWN'}")
    return ok


def check_openai_key() -> bool:
    present = bool(read_env("OPENAI_API_KEY"))
    if present:
        log("🔍 Health: OPENAI_API_KEY present (translation enabled).")
    else:
        log("ℹ️  Health: OPENAI_API_KEY missing (translation disabled).")
    return present


# ---------- PDF -> images (multi-backend) ----------
@with_retries(tries=2, backoff=1.5)
def pdf_to_images_poppler(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """Preferred: use Poppler via pdf2image (python) or direct pdftoppm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images: List[Path] = []
    if PDF2IMAGE_OK:
        # pdf2image requires poppler binaries installed
        pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png", output_folder=str(out_dir))
        # pages are PIL Images in memory; save
        for i, im in enumerate(pages, start=1):
            out = out_dir / f"{pdf_path.stem}-{i}.png"
            im.save(out)
            images.append(out)
        return images

    # fallback: shell out to pdftoppm
    if shutil.which("pdftoppm"):
        prefix = out_dir / pdf_path.stem
        cmd = ["pdftoppm", "-png", f"-r{dpi}", str(pdf_path), str(prefix)]
        subprocess.run(cmd, check=True, timeout=SUBPROCESS_TIMEOUT)
        # collect files produced: prefix-1.png, prefix-2.png...
        images = sorted(out_dir.glob(f"{pdf_path.stem}-*.png"))
        return images

    raise RuntimeError("Poppler not available")


@with_retries(tries=2, backoff=1.5)
def pdf_to_images_imagemagick(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """Fallback: ImageMagick 'magick' convert."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("magick"):
        raise RuntimeError("ImageMagick not available")
    out_pattern = out_dir / f"{pdf_path.stem}-%d.png"
    cmd = [
        "magick", "-density", str(dpi),
        str(pdf_path),
        "-colorspace", "sRGB",
        "-alpha", "remove",
        "-alpha", "off",
        str(out_pattern)
    ]
    subprocess.run(cmd, check=True, timeout=SUBPROCESS_TIMEOUT)
    images = sorted(out_dir.glob(f"{pdf_path.stem}-*.png"))
    return images


def pdf_to_images_best_effort(pdf_path: Path, out_dir: Path, dpi: int = 200) -> List[Path]:
    """Try poppler path, then ImageMagick, else fail gracefully."""
    # Try Poppler route
    try:
        return pdf_to_images_poppler(pdf_path, out_dir, dpi=dpi)
    except Exception as e:
        log(f"⚠️  Poppler path failed: {e}")

    # Try ImageMagick route
    try:
        return pdf_to_images_imagemagick(pdf_path, out_dir, dpi=dpi)
    except Exception as e:
        log(f"⚠️  ImageMagick path failed: {e}")

    # Last resort: no images generated
    log("❌ No PDF rasterizer available. Skipping image generation.")
    return []


# ---------- OCR ----------
def safe_logger_payload(m: Dict[str, Any]) -> Dict[str, Any]:
    """Strip functions/complex types from Tesseract logger payload to avoid DataCloneError equivalents."""
    out = {}
    for k, v in m.items():
        if k == "status" and isinstance(v, str):
            out["status"] = v
        elif k == "progress":
            try:
                fv = float(v)
                out["progress"] = round(fv, 3)
            except Exception:
                pass
        # ignore everything else that might not be JSON-serializable
    return out


def ocr_image(path: Path, lang: str = LANGS_FOR_OCR) -> str:
    """
    OCR a single image.
    Wrapped with circuit breaker + retries + timeout + graceful fallback (empty string).
    """
    if CB_OCR.state == "OPEN" and not CB_OCR.allow():
        METRICS.offline_mode_used = True
        log(f"🟡 OCR circuit OPEN. Skipping OCR for {path.name} and using empty text.")
        return ""

    if not (PIL_OK and check_tesseract()):
        log("ℹ️  OCR prerequisites missing (PIL/Tesseract). Returning empty text.")
        CB_OCR.failure()
        METRICS.ocr_failures += 1
        return ""

    @with_retries(tries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF)
    def _do_ocr_image(p: Path) -> str:
        METRICS.ocr_calls += 1
        image = Image.open(p)
        try:
            # Restrict OCR runtime via timeout by using separate process (pytesseract has timeout param)
            text = pytesseract.image_to_string(image, lang=lang, timeout=SUBPROCESS_TIMEOUT)
            return text
        finally:
            try:
                image.close()
            except Exception:
                pass

    try:
        txt = _do_ocr_image(path)
        CB_OCR.success()
        return txt
    except Exception as e:
        log(f"❌ OCR failed for {path.name}: {e}")
        CB_OCR.failure()
        METRICS.ocr_failures += 1
        return ""


# ---------- Caching ----------
def cache_key_for(pdf_path: Path, page_index: int, variant: str = "orig") -> Path:
    safe_name = f"{pdf_path.stem}_p{page_index}_{variant}.txt"
    return CACHE_DIR / safe_name


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_text(path: Path) -> Optional[str]:
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return None
    return None


# ---------- Translation (optional, with circuit breaker) ----------
def translate_text(text: str, target_lang: str = "es") -> str:
    """
    Uses OpenAI API if available & internet up. Circuit breaker + retries + timeout included.
    If not available, returns a labeled fallback.
    """
    api_key = read_env("OPENAI_API_KEY")
    online = REQUESTS_OK and check_internet()
    use_api = bool(api_key and online)

    if CB_TRANSLATE.state == "OPEN" and not CB_TRANSLATE.allow():
        log("🟡 Translate circuit OPEN. Using cached/identity fallback.")
        METRICS.offline_mode_used = True
        return f"[offline-fallback] {text}"

    # Try cache
    h = hashlib.sha256((text + target_lang).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"tx_{h}.txt"
    cached = read_text(cache_path)
    if not use_api and cached:
        METRICS.offline_mode_used = True
        return cached

    if not use_api:
        # No API / offline mode
        METRICS.offline_mode_used = True
        res = f"[no-api] {text}"
        write_text(cache_path, res)
        return res

    @with_retries(tries=DEFAULT_RETRIES, backoff=DEFAULT_BACKOFF)
    def _call_openai(prompt_text: str) -> str:
        if not REQUESTS_OK:
            raise RuntimeError("requests not installed")
        METRICS.translate_calls += 1

        # Minimal prompt to reduce cost and improve speed
        system = "You are a translation engine. Translate the user text to the requested language preserving meaning."
        user = f"Translate to {target_lang}:\n{prompt_text}"

        # Use the Chat Completions API (generic)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2,
            "timeout": HTTP_TIMEOUT  # NOTE: requests timeout is separate, we pass below
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        out = data["choices"][0]["message"]["content"].strip()
        return out

    try:
        result = _call_openai(text)
        CB_TRANSLATE.success()
        write_text(cache_path, result)
        return result
    except Exception as e:
        METRICS.translate_failures += 1
        CB_TRANSLATE.failure()
        log(f"❌ Translate failed: {e}")
        # graceful fallback: cached (if any) or identity
        if cached:
            log("↩️  Using cached translation due to failure.")
            METRICS.offline_mode_used = True
            return cached
        return f"[fallback] {text}"


# ---------- Main pipeline ----------
def process_pdf(pdf_path: Path, target_lang: Optional[str] = None) -> Dict[str, Any]:
    """
    Process one PDF: convert to images (best effort), OCR each page (with rotations),
    combine, optionally translate, and save to output.
    """
    log(f"📄 Processing: {pdf_path}")
    METRICS.pdfs_processed += 1

    # Convert PDF to images (best effort)
    out_dir = CACHE_DIR / pdf_path.stem
    images = pdf_to_images_best_effort(pdf_path, out_dir, dpi=220)
    METRICS.images_generated += len(images)

    combined_text = []
    page_idx = 0
    for img in images:
        page_idx += 1
        # Try OCR with small rotation set to help skewed pages
        variants = [(img, "orig")]
        # low-cost rotations with ImageMagick if available
        if shutil.which("magick"):
            for deg in (-10, -5, 5, 10):
                rotated = img.with_name(f"{img.stem}_r{deg}.png")
                try:
                    subprocess.run(
                        ["magick", str(img), "-deskew", "40%", "-rotate", str(deg), str(rotated)],
                        check=True, timeout=SUBPROCESS_TIMEOUT
                    )
                    variants.append((rotated, f"rot{deg}"))
                except Exception as e:
                    log(f"⚠️  Rotation {deg}° failed for {img.name}: {e}")

        page_text = ""
        for vpath, vname in variants:
            cache_p = cache_key_for(pdf_path, page_idx, vname)
            cached_t = read_text(cache_p)
            if cached_t:
                page_text = cached_t
                break

            txt = ocr_image(vpath, lang=LANGS_FOR_OCR)
            if txt and txt.strip():
                page_text = txt
                write_text(cache_p, txt)
                break  # take the first good variant

        combined_text.append(page_text)

    final_text = "\n\n".join(t.strip() for t in combined_text if t is not None)

    # Save raw OCR
    out_txt = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    write_text(out_txt, final_text)

    translated = None
    if target_lang:
        translated = translate_text(final_text, target_lang=target_lang)
        out_tr = OUTPUT_DIR / f"{pdf_path.stem}.{target_lang}.txt"
        write_text(out_tr, translated)

    result = {
        "pdf": str(pdf_path),
        "pages": len(images),
        "ocr_text_file": str(out_txt),
        "translated_text_file": str(out_tr) if target_lang else None,
        "used_offline": METRICS.offline_mode_used,
        "cb_states": {
            "ocr": CB_OCR.state,
            "translate": CB_TRANSLATE.state
        }
    }
    return result


def main():
    ensure_dirs()

    # ---- Health checks (we *show* where we protect calls) ----
    # Filesystem: (we already create dirs above)
    # OCR deps:
    _ = check_tesseract()      # <-- health check around OCR dependency
    _ = check_poppler()        # <-- health check for Poppler converter
    _ = check_imagemagick()    # <-- health check for ImageMagick fallback
    net_ok = check_internet()  # <-- health check internet
    _ = check_openai_key()     # <-- health check translation capability

    # When internet is down -> plan to use offline/cached flow
    if not net_ok:
        METRICS.offline_mode_used = True
        log("🟡 Internet down: enabling offline/cached mode and simple view.")

    # Collect PDFs
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        log(f"ℹ️  No PDFs found in {PDF_DIR}. Place files and run again.")
        save_metrics(METRICS)
        return

    # Choose target language for translation (optional)
    # You can set TARGET_LANG=es in environment, or leave empty to skip
    target_lang = read_env("TARGET_LANG", "").strip() or None
    if target_lang:
        log(f"🌐 Translation target language: {target_lang}")
    else:
        log("ℹ️  Translation disabled (set TARGET_LANG=es to enable).")

    results = []
    for pdf in pdfs:
        try:
            res = process_pdf(pdf, target_lang=target_lang)
            results.append(res)
        except Exception as e:
            # Graceful: keep going to next PDF
            log(f"❌ Fatal error on {pdf.name}: {e} (skipping)")
            continue

    # Persist metrics / dashboard
    save_metrics(METRICS)

    # Simple “dashboard light” view
    dash = {
        "processed": len(results),
        "ocr_cb": CB_OCR.state,
        "translate_cb": CB_TRANSLATE.state,
        "offline_mode_used": METRICS.offline_mode_used,
        "metrics_file": str(METRICS_FILE),
    }
    print("\n===== DASHBOARD =====")
    print(json.dumps(dash, indent=2))
    print("=====================\n")

    # Summarize outputs
    for r in results:
        print(f"✅ {Path(r['pdf']).name} -> pages={r['pages']}")
        print(f"   OCR: {r['ocr_text_file']}")
        if r.get("translated_text_file"):
            print(f"   TR : {r['translated_text_file']}")
    print("\n🎉 Done.")


if __name__ == "__main__":
    # Load .env if present (gracefully)
    try:
        from dotenv import load_dotenv  # optional nicety
        load_dotenv()
    except Exception:
        pass
    main()
