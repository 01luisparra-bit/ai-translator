from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from ai_translator.utils.health import check_health
from ai_translator.translate.client import translate_text

app = FastAPI(
    title="AI Translator API",
    description="Translate text or PDF content into your target language using AI. 🚀",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Root route (homepage)
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🎉 Welcome to the AI Translator API!",
        "status": "live",
        "docs": "/docs",
        "usage": {
            "translate_text": "POST /translate with {'text': 'Hello world', 'target_lang': 'es'}",
            "health_check": "GET /health",
        },
    }


# ---------------------------------------------------------------------------
# Health check route
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    health = check_health()
    return {
        "tesseract": str(health.tesseract_path),
        "poppler": str(health.poppler_path),
        "magick": str(health.magick_path),
        "internet_ok": health.internet_ok,
        "openai_key_present": health.openai_key_present,
    }


# ---------------------------------------------------------------------------
# Text translation endpoint
# ---------------------------------------------------------------------------
@app.post("/translate")
async def translate_endpoint(
    text: str = Form(...),
    target_lang: str = Form("es"),
):
    """
    Translate text into the given target language.
    """
    try:
        result = translate_text(text, target_lang)
        return {"original": text, "translated": result, "target_lang": target_lang}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "hint": "Check your API key or input text."},
        )


# ---------------------------------------------------------------------------
# PDF upload endpoint (optional)
# ---------------------------------------------------------------------------
@app.post("/translate-pdf")
async def translate_pdf(file: UploadFile, target_lang: str = Form("es")):
    """
    Accepts a PDF upload and translates its text.
    """
    pdf_dir = Path("pdfs")
    pdf_dir.mkdir(exist_ok=True)
    temp_path = pdf_dir / file.filename

    # Save uploaded PDF
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Placeholder response
    return {
        "file": file.filename,
        "target_lang": target_lang,
        "status": "Uploaded successfully",
    }


# ---------------------------------------------------------------------------
# Run (for local testing only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("ai_translator.api:app", host="0.0.0.0", port=port, reload=True)
