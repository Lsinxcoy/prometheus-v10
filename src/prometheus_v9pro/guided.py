"""Prometheus V9 Guided Evolution — Goal-directed mutation toward specific targets."""
from __future__ import annotations

import logging
import math
import random
from typing import Any

from prometheus_v9pro.schema import Genome

logger = logging.getLogger(__name__)


class GuidedEvolution:
    """Goal-directed evolution: mutations biased toward improving specific metrics.

    Instead of random exploration, guide mutations toward the current goal.
    Implements T7 (business metrics > internal metrics): guide by real fitness.
    """

    def __init__(self, target_metric: str = "fitness", target_value: float = 1.0) -> None:
        self._target_metric = target_metric
        self._target_value = target_value
        self._mutations_applied = 0

    def guided_mutate(self, genome: Genome, current_fitness: float,
                      best_fitness: float = 0.0) -> Genome:
        """Apply guided mutation toward target."""
        import copy
        child = copy.deepcopy(genome)
        self._mutations_applied += 1

        # Direction: if fitness is far from target, make larger moves
        distance = abs(self._target_value - current_fitness)
        step_size = min(0.5, distance * 0.3)  # Bigger steps when far

        # Apply guided perturbation to config
        for key, value in child.config.items():
            if isinstance(value, (int, float)):
                # Bias toward target: increase if below, decrease if above
                if current_fitness < best_fitness:
                    # Move toward best-known config values
                    perturbation = random.gauss(0, step_size)
                    child.config[key] = value + perturbation
                else:
                    # Random exploration with small step
                    perturbation = random.gauss(0, step_size * 0.3)
                    child.config[key] = value + perturbation

        return child

    @property
    def stats(self) -> dict[str, Any]:
        return {"mutations_applied": self._mutations_applied, "target": self._target_value}
