from ai_translator.utils.health import check_health, Health


def test_health_returns_dataclass():
    h = check_health()
    assert isinstance(h, Health)
    assert hasattr(h, "internet_ok")
