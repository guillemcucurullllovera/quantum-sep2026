"""Real exponential-backoff retry logic for hardware job submission, which
is the one place in this platform where a transient network/queue error
should be retried rather than immediately surfaced to the caller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.1
    max_delay_s: float = 30.0
    backoff_factor: float = 2.0
    retry_on: tuple[type[Exception], ...] = (Exception,)

    def delay_for_attempt(self, attempt: int) -> float:
        """attempt is 1-indexed (first retry = attempt 1)."""
        delay = self.base_delay_s * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay_s)


class RetryExhaustedError(Exception):
    def __init__(self, attempts: int, last_exception: Exception) -> None:
        super().__init__(f"Gave up after {attempts} attempts. Last error: {last_exception!r}")
        self.attempts = attempts
        self.last_exception = last_exception


def call_with_retry(fn: Callable[[], T], policy: RetryPolicy | None = None, sleep: Callable[[float], None] = time.sleep) -> T:
    """Call `fn` (a zero-arg callable) with retry. `sleep` is injectable so
    tests can assert on backoff timing without actually waiting."""
    policy = policy or RetryPolicy()
    last_exc: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retry_on as exc:  # type: ignore[misc]
            last_exc = exc
            if attempt == policy.max_attempts:
                break
            sleep(policy.delay_for_attempt(attempt))
    raise RetryExhaustedError(policy.max_attempts, last_exc)  # type: ignore[arg-type]


def with_retry(policy: RetryPolicy | None = None):
    """Decorator form of call_with_retry."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return call_with_retry(lambda: func(*args, **kwargs), policy=policy)

        return wrapper

    return decorator


__all__ = ["RetryPolicy", "RetryExhaustedError", "call_with_retry", "with_retry"]
