from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    reset_timeout: int = 60
    state: str = "CLOSED"
    failures: int = 0
    opened_at: float | None = None

    def allow(self) -> bool:
        if self.state == "OPEN":
            if (time.time() - (self.opened_at or 0)) >= self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "CLOSED"
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "HALF_OPEN" or self.failures >= self.failure_threshold:
            self.state = "OPEN"
            self.opened_at = time.time()
