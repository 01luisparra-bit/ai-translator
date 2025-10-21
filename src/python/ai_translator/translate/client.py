from __future__ import annotations
import os
import json
from dataclasses import dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import urllib.request


@dataclass
class TranslationConfig:
    target_lang: str
    timeout: int = 30


class TransientHTTPError(Exception):
    """Temporary network failure that should trigger retry logic."""

    pass


def _http_post_json(url: str, payload: dict, timeout: int) -> dict:
    """Send JSON payload via HTTP POST, with timeout and retry."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise TransientHTTPError(str(e))


@retry(
    stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", "3"))),
    wait=wait_exponential(
        multiplier=float(os.getenv("RETRY_BACKOFF_SECONDS", "1")), max=10
    ),
    retry=retry_if_exception_type(TransientHTTPError),
    reraise=True,
)
def translate_text(text: str, cfg: TranslationConfig) -> str:
    """
    Demo translator stub.

    If OFFLINE_MODE=true or no OPENAI_API_KEY, it returns a mock translation.
    Otherwise, this simulates a real translation API call.
    """
    if not text.strip():
        return ""

    # Offline mode: skip network call
    if os.getenv("OFFLINE_MODE", "false").lower() == "true" or not os.getenv(
        "OPENAI_API_KEY"
    ):
        return f"[{cfg.target_lang}] {text}"

    # Example (replace with real API URL later)
    result = _http_post_json(
        "http://127.0.0.1:9/fake",
        {"text": text, "lang": cfg.target_lang},
        cfg.timeout,
    )
    return result.get("translation", f"[{cfg.target_lang}] {text}")


__all__ = ["translate_text", "TranslationConfig", "TransientHTTPError"]
