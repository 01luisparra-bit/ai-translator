from __future__ import annotations
import subprocess
import shlex
from pathlib import Path
from typing import List


def pdf_to_pngs(pdf_path: Path, out_dir: Path, pdftoppm: str) -> List[Path]:
    """
    Convert a single PDF to a list of PNG image files using Poppler (pdftoppm).
    Produces files like {stem}-1.png, {stem}-2.png in the given output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    out_base = out_dir / stem

    # Ensure Windows-safe quoting
    cmd = f'"{pdftoppm}" -png -scale-to 1500 "{pdf_path}" "{out_base}"'
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {proc.stderr.strip()}")

    # Return all generated image paths
    return sorted(out_dir.glob(f"{stem}-*.png"), key=lambda p: p.name)
