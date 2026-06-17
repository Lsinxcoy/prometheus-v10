"""Prometheus V9 Safe Harbor — Emergency rollback mechanism.

When evolution goes wrong (fitness crashes, safety violations, corruption),
Safe Harbor provides a last-resort rollback to a known-good state.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import Genome

logger = logging.getLogger(__name__)


@dataclass
class SafeHarborState:
    """A safe harbor checkpoint."""
    genome: Genome | None = None
    fitness: float = 0.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class SafeHarbor:
    """Emergency rollback mechanism.

    Triggers:
    - Fitness drops > 50% from best
    - Safety violation detected
    - Manual emergency stop

    Action:
    - Restore genome from last safe checkpoint
    - Reset evolution state
    - Alert user
    """

    def __init__(self) -> None:
        self._safe_state: SafeHarborState | None = None
        self._rollback_count: int = 0
        self._last_fitness: float = 0.0

    def anchor(self, genome: Genome, reason: str = "periodic") -> None:
        """Set current state as safe harbor."""
        self._safe_state = SafeHarborState(
            genome=genome, fitness=genome.fitness, reason=reason,
        )
        self._last_fitness = genome.fitness
        logger.info(f"Safe harbor anchored: fitness={genome.fitness:.3f} ({reason})")

    def check_and_rollback(self, current_genome: Genome) -> Genome | None:
        """Check if rollback is needed and execute it."""
        if not self._safe_state:
            return None

        # Check fitness crash
        if self._safe_state.fitness > 0 and current_genome.fitness < self._safe_state.fitness * 0.5:
            logger.warning(f"Fitness crash detected: {current_genome.fitness:.3f} vs safe {self._safe_state.fitness:.3f}")
            return self._rollback()

        return None

    def emergency_rollback(self) -> Genome | None:
        """Force an emergency rollback."""
        return self._rollback()

    def _rollback(self) -> Genome | None:
        """Execute rollback to safe state."""
        if not self._safe_state or not self._safe_state.genome:
            return None
        self._rollback_count += 1
        import copy
        genome = copy.deepcopy(self._safe_state.genome)
        logger.warning(f"Safe harbor rollback #{self._rollback_count}: restored fitness={genome.fitness:.3f}")
        return genome

    @property
    def is_anchored(self) -> bool:
        return self._safe_state is not None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "anchored": self.is_anchored,
            "safe_fitness": self._safe_state.fitness if self._safe_state else 0,
            "rollbacks": self._rollback_count,
        }
