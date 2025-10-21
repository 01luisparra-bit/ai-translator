from __future__ import annotations
import os
import shutil
import socket
from dataclasses import dataclass


@dataclass
class Health:
    tesseract_path: str | None
    poppler_path: str | None
    magick_path: str | None
    internet_ok: bool
    openai_key_present: bool


def _check_internet(host="8.8.8.8", port=53, timeout=2) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except Exception:
        return False


def check_health() -> Health:
    tess = os.getenv("TESSERACT_PATH") or shutil.which("tesseract")
    poppler = os.getenv("POPPLER_PATH") or shutil.which("pdftoppm")
    magick = os.getenv("IMAGEMAGICK_PATH") or shutil.which("magick")
    internet_ok = _check_internet()
    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
    return Health(
        tesseract_path=tess,
        poppler_path=poppler,
        magick_path=magick,
        internet_ok=internet_ok,
        openai_key_present=openai_key_present,
    )
