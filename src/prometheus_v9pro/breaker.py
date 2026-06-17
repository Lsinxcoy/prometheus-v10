"""Prometheus V9 Circuit Breaker — Per-tool state machine (CLOSED→OPEN→HALF_OPEN)."""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class BreakerState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class BreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class PerToolCircuitBreaker:
    """Per-tool circuit breaker. Each tool/provider has independent state."""

    def __init__(self, config: BreakerConfig | None = None) -> None:
        self._config = config or BreakerConfig()
        self._breakers: dict[str, dict[str, Any]] = {}

    def _get_breaker(self, tool_name: str) -> dict[str, Any]:
        if tool_name not in self._breakers:
            self._breakers[tool_name] = {
                "state": BreakerState.CLOSED,
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": 0.0,
                "half_open_calls": 0,
            }
        return self._breakers[tool_name]

    def can_execute(self, tool_name: str) -> bool:
        """Check if tool can execute based on breaker state."""
        breaker = self._get_breaker(tool_name)
        if breaker["state"] == BreakerState.CLOSED:
            return True
        if breaker["state"] == BreakerState.OPEN:
            if time.time() - breaker["last_failure_time"] > self._config.recovery_timeout:
                breaker["state"] = BreakerState.HALF_OPEN
                breaker["half_open_calls"] = 0
                return True
            return False
        if breaker["state"] == BreakerState.HALF_OPEN:
            return breaker["half_open_calls"] < self._config.half_open_max_calls
        return False

    def record_success(self, tool_name: str) -> None:
        breaker = self._get_breaker(tool_name)
        breaker["success_count"] += 1
        if breaker["state"] == BreakerState.HALF_OPEN:
            breaker["state"] = BreakerState.CLOSED
            breaker["failure_count"] = 0

    def record_failure(self, tool_name: str) -> None:
        breaker = self._get_breaker(tool_name)
        breaker["failure_count"] += 1
        breaker["last_failure_time"] = time.time()
        if breaker["state"] == BreakerState.HALF_OPEN:
            breaker["state"] = BreakerState.OPEN
        elif breaker["failure_count"] >= self._config.failure_threshold:
            breaker["state"] = BreakerState.OPEN

    def get_state(self, tool_name: str) -> BreakerState:
        return self._get_breaker(tool_name)["state"]

    def get_all_states(self) -> dict[str, str]:
        return {name: b["state"].value for name, b in self._breakers.items()}
