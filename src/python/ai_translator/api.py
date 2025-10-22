from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel

from ai_translator.translate.client import translate_text, TranslationConfig

app = FastAPI(title="AI Translator API", version="0.1.0")


class TranslateTextRequest(BaseModel):
    text: str
    target_lang: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"app": "ai-translator", "version": "0.1.0"}


@app.post("/translate_text")
def translate_text_endpoint(req: TranslateTextRequest):
    target = req.target_lang or os.getenv("TARGET_LANG") or "es"
    cfg = TranslationConfig(
        target_lang=target, timeout=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )
    if not req.text.strip():
        return {"translation": ""}
    out = translate_text(req.text, cfg)
    return {"translation": out, "target_lang": target}
