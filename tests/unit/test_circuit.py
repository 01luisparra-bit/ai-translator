from ai_translator.utils.circuit import CircuitBreaker
import time


def test_circuit_initially_closed():
    cb = CircuitBreaker()
    assert cb.state == "CLOSED"


def test_circuit_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)
    cb.record_failure()
    cb.record_failure()
    assert cb.state == "OPEN"


def test_circuit_allows_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=1)
    cb.record_failure()
    time.sleep(1.1)
    assert cb.allow()
