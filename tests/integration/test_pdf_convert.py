from ai_translator.pdf.convert import pdf_to_pngs
from pathlib import Path
import pytest


@pytest.mark.needs_tools
def test_pdf_to_pngs_handles_missing_poppler(tmp_path: Path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake pdf")
    with pytest.raises(Exception):
        pdf_to_pngs(fake_pdf, tmp_path, "nonexistent_pdftoppm.exe")
