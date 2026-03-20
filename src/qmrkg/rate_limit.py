"""Shared rate limiting utilities for task-scoped LLM calls."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class RollingRateLimiter:
    """Enforce a rolling requests-per-minute cap across worker threads."""

    def __init__(
        self,
        rpm: int,
        *,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        if rpm <= 0:
            raise ValueError("rpm must be greater than 0")
        self.rpm = rpm
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._lock = threading.Lock()
        self._requests: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            wait_for = 0.0
            with self._lock:
                now = self._time_fn()
                self._trim(now)
                if len(self._requests) < self.rpm:
                    self._requests.append(now)
                    return
                wait_for = max(0.0, 60.0 - (now - self._requests[0]))
            if wait_for > 0:
                self._sleep_fn(wait_for)

    def _trim(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= 60.0:
            self._requests.popleft()
