from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path("output/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key_for(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_cache(name: str, obj: Any) -> Path:
    return CACHE_DIR / f"{name}-{_key_for(obj)}.json"


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
