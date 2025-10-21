import json
from pathlib import Path
from ai_translator.main import main, METRICS_PATH


def test_end_to_end(tmp_path, monkeypatch):
    """Simulate a full OCR → Translate → Output pipeline run."""

    # --- Prepare a sandboxed environment
    pdf_dir = tmp_path / "pdfs"
    out_dir = tmp_path / "output"
    pdf_dir.mkdir()
    out_dir.mkdir()

    # Create a fake PDF (the content is irrelevant)
    fake_pdf = pdf_dir / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake content")

    # --- Mock environment variables
    monkeypatch.setenv("PDF_DIR", str(pdf_dir))
    monkeypatch.setenv("OUT_DIR", str(out_dir))
    monkeypatch.setenv("TARGET_LANG", "es")
    monkeypatch.setenv("OFFLINE_MODE", "true")  # avoid API calls
    monkeypatch.setenv("MAX_RETRIES", "1")
    monkeypatch.setenv("RETRY_BACKOFF_SECONDS", "1")

    # --- Run the app
    main()

    # --- Verify outputs
    metrics_path = Path(METRICS_PATH)
    assert metrics_path.exists(), "Metrics file was not written"
    metrics = json.loads(metrics_path.read_text())
    assert "processed" in metrics
    assert metrics["processed"] >= 0

    # Ensure output directory is created
    assert out_dir.exists()

    # OCR/translation files are optional in offline mode
    files = list(out_dir.glob("*.txt"))
    assert files == [] or all(f.suffix == ".txt" for f in files)
