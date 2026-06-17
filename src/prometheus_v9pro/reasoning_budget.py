"""Prometheus V9 Reasoning Budget — Token/entropy/time triple budget control.

From MiMo insights: reasoning efficiency crisis — problems aren't too much reasoning
but misallocated reasoning (over-reasoning simple, under-reasoning hard).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetState:
    """Current budget state."""
    tokens_used: int = 0
    tokens_limit: int = 4000
    time_used: float = 0.0
    time_limit: float = 240.0
    entropy: float = 1.0  # 0=deterministic, 1=maximum randomness
    steps_taken: int = 0
    max_steps: int = 50

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.tokens_limit - self.tokens_used)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.time_limit - self.time_used)

    @property
    def is_exhausted(self) -> bool:
        return (
            self.tokens_used >= self.tokens_limit
            or self.time_used >= self.time_limit
            or self.steps_taken >= self.max_steps
        )

    @property
    def utilization(self) -> float:
        return max(
            self.tokens_used / max(1, self.tokens_limit),
            self.time_used / max(0.1, self.time_limit),
            self.steps_taken / max(1, self.max_steps),
        )


class ReasoningBudget:
    """Triple budget controller: tokens + time + entropy.

    Entropy control (from Prism paper):
    - High entropy (>0.7): exploratory, creative, divergent
    - Medium entropy (0.3-0.7): balanced, analytical
    - Low entropy (<0.3): deterministic, convergent, focused

    Budget allocation strategy:
    - Simple tasks: low token budget, high entropy allowed
    - Hard tasks: high token budget, low entropy (focused)
    """

    def __init__(self, tokens: int = 4000, time_seconds: int = 240,
                 max_steps: int = 50, initial_entropy: float = 1.0) -> None:
        self._state = BudgetState(
            tokens_limit=tokens, time_limit=float(time_seconds),
            max_steps=max_steps, entropy=initial_entropy,
        )
        self._start_time = time.time()
        self._checkpoints: list[dict] = []

    @property
    def state(self) -> BudgetState:
        # Update time_used on each access
        self._state.time_used = time.time() - self._start_time
        return self._state

    @property
    def is_exhausted(self) -> bool:
        return self.state.is_exhausted

    def consume_tokens(self, count: int) -> bool:
        """Try to consume tokens. Returns False if over budget."""
        self._state.tokens_used += count
        if self._state.tokens_used > self._state.tokens_limit:
            logger.warning(f"Token budget exhausted: {self._state.tokens_used}/{self._state.tokens_limit}")
            return False
        return True

    def step(self) -> bool:
        """Take one reasoning step. Returns False if over budget."""
        self._state.steps_taken += 1
        self._state.time_used = time.time() - self._start_time
        if self._state.is_exhausted:
            return False
        return True

    def adjust_entropy(self, difficulty: float) -> None:
        """Adjust entropy based on estimated task difficulty.

        difficulty: 0=trivial, 1=extremely hard
        - Easy tasks → high entropy (exploratory)
        - Hard tasks → low entropy (focused)
        """
        self._state.entropy = max(0.1, min(1.0, 1.0 - difficulty * 0.8))

    def checkpoint(self, label: str = "") -> dict:
        """Save a budget checkpoint."""
        cp = {
            "label": label, "tokens_used": self._state.tokens_used,
            "steps": self._state.steps_taken, "entropy": self._state.entropy,
            "utilization": self.state.utilization,
        }
        self._checkpoints.append(cp)
        return cp

    @property
    def recommendation(self) -> str:
        """Get budget allocation recommendation."""
        u = self.state.utilization
        if u < 0.3:
            return "plenty_of_budget"
        elif u < 0.6:
            return "moderate_usage"
        elif u < 0.8:
            return "budget_tightening"
        else:
            return "budget_critical"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "tokens": f"{self._state.tokens_used}/{self._state.tokens_limit}",
            "time": f"{self._state.time_used:.1f}/{self._state.time_limit:.1f}",
            "steps": f"{self._state.steps_taken}/{self._state.max_steps}",
            "entropy": self._state.entropy,
            "utilization": f"{self.state.utilization:.2f}",
            "recommendation": self.recommendation,
        }
