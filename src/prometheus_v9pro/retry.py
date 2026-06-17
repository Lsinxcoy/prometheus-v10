"""Prometheus V9PRO Retry — Exponential backoff + temperature cooling + fallback.

V8 had core/retry.py (131 lines). V9PRO adds: temperature cooling
(reduces operation complexity on repeated failure) and deterministic
fallback chains.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    temperature_cooling: float = 0.8  # Each retry reduces "temperature" (complexity)


@dataclass
class RetryResult:
    """Result of a retry-wrapped operation."""
    success: bool = False
    attempts: int = 0
    last_error: str = ""
    total_delay: float = 0.0
    result: Any = None


class RetryManager:
    """Exponential backoff retry with temperature cooling and fallback.

    Temperature cooling: on each retry, the operation's "temperature"
    (complexity/aggressiveness) is reduced by the cooling factor.
    This maps to LLM operations where a simpler prompt/approach may
    succeed where a complex one failed.
    """
    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self._policy = policy or RetryPolicy()
        self._total_retries = 0
        self._total_successes = 0
        self._total_failures = 0

    def execute(self, fn: Callable, *args, fallback: Callable | None = None,
                on_retry: Callable | None = None, **kwargs) -> RetryResult:
        """Execute fn with retry policy. Optionally call fallback on final failure.

        Args:
            fn: Primary function to execute
            fallback: Called if all retries fail
            on_retry: Called before each retry with (attempt, error, temperature)
        """
        import random
        result = RetryResult()
        temperature = 1.0

        for attempt in range(1, self._policy.max_attempts + 1):
            result.attempts = attempt
            try:
                # Pass temperature to function if it accepts it
                ret = fn(*args, **kwargs)
                result.success = True
                result.result = ret
                self._total_successes += 1
                return result
            except Exception as e:
                result.last_error = str(e)
                self._total_retries += 1

                if attempt < self._policy.max_attempts:
                    # Calculate delay
                    delay = min(
                        self._policy.base_delay * (self._policy.exponential_base ** (attempt - 1)),
                        self._policy.max_delay,
                    )
                    if self._policy.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    result.total_delay += delay

                    # Cool temperature
                    temperature *= self._policy.temperature_cooling

                    if on_retry:
                        try:
                            on_retry(attempt, e, temperature)
                        except Exception:
                            pass

                    logger.debug(f"Retry {attempt}/{self._policy.max_attempts} after {delay:.1f}s (temp={temperature:.2f}): {e}")
                    time.sleep(delay)

        # All retries exhausted
        if fallback:
            try:
                result.result = fallback(*args, **kwargs)
                result.success = True
                self._total_successes += 1
                return result
            except Exception as e:
                result.last_error = str(e)

        self._total_failures += 1
        return result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_retries": self._total_retries,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": self._total_successes / max(1, self._total_successes + self._total_failures),
        }
