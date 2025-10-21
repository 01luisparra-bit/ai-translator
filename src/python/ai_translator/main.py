from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ai_translator.utils.health import check_health, Health
from ai_translator.utils.circuit import CircuitBreaker
from ai_translator.utils.cache import get_cache, save_json, load_json
from ai_translator.pdf.convert import pdf_to_pngs
from ai_translator.ocr.engine import ocr_image
from ai_translator.translate.client import (
    translate_text,
    TranslationConfig,
    TransientHTTPError,
)


# ---------- Load environment ----------
load_dotenv()

PDF_DIR = Path(os.getenv("PDF_DIR", "pdfs"))
OUT_DIR = Path(os.getenv("OUT_DIR", "output"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------- Circuit breakers ----------
ocr_cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)
translate_cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)

# ---------- Metrics ----------
METRICS = {
    "processed": 0,
    "ocr_cb": "CLOSED",
    "translate_cb": "CLOSED",
    "offline_mode_used": False,
}
METRICS_PATH = Path("metrics.json")


# ---------- Error Classes ----------
class PdfConvertError(Exception):
    """Raised when PDF→PNG conversion fails."""

    pass


class OcrError(Exception):
    """Raised when OCR processing fails."""

    pass


class TranslateError(Exception):
    """Raised when translation fails."""

    pass


# ---------- Resilient wrappers ----------
@retry(
    stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", "3"))),
    wait=wait_exponential(
        multiplier=float(os.getenv("RETRY_BACKOFF_SECONDS", "1")), max=10
    ),
    retry=retry_if_exception_type(PdfConvertError),
    reraise=True,
)
def safe_pdf_to_pngs(pdf_path: Path, out_dir: Path, pdftoppm: str):
    """Safely converts a PDF into PNG images with retry handling."""
    try:
        return pdf_to_pngs(pdf_path, out_dir, pdftoppm)
    except Exception as e:
        raise PdfConvertError(str(e))


@retry(
    stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", "3"))),
    wait=wait_exponential(
        multiplier=float(os.getenv("RETRY_BACKOFF_SECONDS", "1")), max=10
    ),
    retry=retry_if_exception_type(OcrError),
    reraise=True,
)
def safe_ocr_image(img: Path, tess: str, lang: str):
    """Safely performs OCR with retry handling."""
    try:
        return ocr_image(img, tess, lang)
    except Exception as e:
        raise OcrError(str(e))


# ---------- Dashboard ----------
def dump_dashboard():
    """Standardized metrics schema for tests and monitoring."""
    metrics = {
        "processed": METRICS.get("processed", 0),
        "ocr_cb": ocr_cb.state,
        "translate_cb": translate_cb.state,
        "offline_mode_used": (
            os.getenv("OFFLINE_MODE", "false").lower() == "true"
            or not os.getenv("OPENAI_API_KEY")
        ),
        "last_run_at": time.time(),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n[bold green]===== DASHBOARD =====[/bold green]")
    print(json.dumps(metrics, indent=2))
    print("[bold green]=====================[/bold green]\n")


# ---------- Main entry ----------
def main():
    """Main pipeline: PDF → OCR → Translate → Output."""
    h: Health = check_health()

    # Health summary
    print(f"🔍 Health: tesseract = {h.tesseract_path or '❌ Not found'}")
    print(f"🔍 Health: poppler = {h.poppler_path or '❌ Not found'}")
    print(f"🔍 Health: ImageMagick = {h.magick_path or '❌ Not found'}")
    print(f"🌐 Internet: {'OK' if h.internet_ok else 'OFFLINE'}")
    print(f"🔑 API key present: {'Yes' if h.openai_key_present else 'No'}")

    target_lang = os.getenv("TARGET_LANG") or ""
    if not target_lang:
        print("ℹ️  Translation disabled (no TARGET_LANG in .env).")
    else:
        print(f"🌐 Target language: {target_lang}")

    # Input PDFs
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"ℹ️  No PDFs found in {PDF_DIR.resolve()}. Place files and run again.")
        dump_dashboard()
        return

    for pdf in pdfs:
        print(f"\n📄 Processing: {pdf.name}")
        out_text_path = OUT_DIR / f"{pdf.stem}.txt"
        cache_key = {"pdf": str(pdf), "tess": bool(h.tesseract_path)}
        cache_file = get_cache("ocr", cache_key)
        cached = load_json(cache_file)

        ocr_text_all = ""
        imgs: list[Path] = []

        # PDF → images
        if h.poppler_path:
            try:
                imgs = safe_pdf_to_pngs(pdf, OUT_DIR / "images", h.poppler_path)
            except Exception as e:
                print(f"❌ PDF→PNG failed: {e}. Using cached OCR if available.")
        else:
            print("🟡 No pdftoppm found — skipping image conversion.")

        # OCR processing
        if imgs and h.tesseract_path:
            if not ocr_cb.allow():
                print("🟡 OCR circuit OPEN — using cached OCR if available.")
                if cached and "text" in cached:
                    ocr_text_all = cached["text"]
            else:
                try:
                    for img in imgs:
                        txt = safe_ocr_image(
                            img, h.tesseract_path, os.getenv("OCR_LANG", "eng")
                        )
                        ocr_text_all += txt + "\n"
                    ocr_cb.record_success()
                except Exception as e:
                    print(f"❌ OCR error: {e}")
                    ocr_cb.record_failure()
                    if cached and "text" in cached:
                        ocr_text_all = cached["text"]
        else:
            # fallback to cached OCR
            if cached and "text" in cached:
                ocr_text_all = cached["text"]
            else:
                print("ℹ️  No OCR results available; continuing with empty text.")

        out_text_path.write_text(ocr_text_all, encoding="utf-8")
        save_json(cache_file, {"text": ocr_text_all})

        # Translation step
        if target_lang:
            if not translate_cb.allow():
                print("🟡 Translate circuit OPEN — skipping translation.")
            else:
                try:
                    cfg = TranslationConfig(
                        target_lang=target_lang,
                        timeout=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
                    )
                    chunks = [c for c in ocr_text_all.split("\n\n") if c.strip()]
                    translated = []
                    for c in chunks:
                        try:
                            translated.append(translate_text(c, cfg))
                        except TransientHTTPError as te:
                            print(f"⚠️ Transient translation error (chunk): {te}")
                            translated.append(f"[{cfg.target_lang}] {c}")
                    (OUT_DIR / f"{pdf.stem}.translated.{target_lang}.txt").write_text(
                        "\n\n".join(translated),
                        encoding="utf-8",
                    )
                    translate_cb.record_success()
                except Exception as e:
                    print(f"❌ Translate error: {e}")
                    translate_cb.record_failure()

        METRICS["processed"] += 1

    dump_dashboard()
    print("🎉 Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[red]Fatal error:[/red] {e}")
        sys.exit(1)
