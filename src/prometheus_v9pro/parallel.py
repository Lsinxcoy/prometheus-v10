"""Prometheus V9 Parallel Evolution — Multiple genomes evolving simultaneously."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import Genome

logger = logging.getLogger(__name__)


class ParallelEvolution:
    """Run multiple genomes in parallel and select the best.

    Implements T3 (quantity > quality): more genomes = more chances.
    """

    def __init__(self, engine=None, population_size: int = 5) -> None:
        self._engine = engine
        self._population_size = population_size

    def evolve_population(self, genomes: list[Genome], steps: int = 1) -> list[Genome]:
        """Evolve a population of genomes in parallel."""
        if not self._engine:
            return genomes
        results = []
        for genome in genomes:
            for _ in range(steps):
                updated, _ = self._engine.evolve_single_step(genome)
                genome = updated
            results.append(genome)
        # Sort by fitness
        results.sort(key=lambda g: g.fitness, reverse=True)
        return results

    def select_survivors(self, genomes: list[Genome], n: int | None = None) -> list[Genome]:
        """Select top N genomes (natural selection)."""
        n = n or self._population_size
        sorted_genomes = sorted(genomes, key=lambda g: g.fitness, reverse=True)
        return sorted_genomes[:n]
