from __future__ import annotations
import subprocess
import shlex
from pathlib import Path


def ocr_image(image_path: Path, tesseract: str, lang: str = "eng") -> str:
    """
    Perform OCR on a single image using Tesseract CLI.
    Returns the recognized text and writes it to a .txt file next to the image.
    """
    out_txt = image_path.with_suffix(".txt")

    # Build command safely with proper quoting
    cmd = f'"{tesseract}" "{image_path}" stdout --dpi 300 -l {lang}'
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "tesseract failed")

    text = proc.stdout
    out_txt.write_text(text, encoding="utf-8", errors="ignore")
    return text
