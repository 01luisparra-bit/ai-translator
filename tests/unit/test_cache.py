from ai_translator.utils.cache import get_cache, save_json, load_json
from pathlib import Path


def test_cache_save_and_load(tmp_path: Path):
    data = {"key": "value"}
    f = tmp_path / "cache.json"
    save_json(f, data)
    loaded = load_json(f)
    assert loaded == data


def test_get_cache_creates_unique_names():
    p1 = get_cache("test", {"a": 1})
    p2 = get_cache("test", {"a": 2})
    assert p1 != p2
