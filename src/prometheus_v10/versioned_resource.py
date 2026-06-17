"""Prometheus V9 Versioned Resource — Genome version management with diff."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.schema import Genome

logger = logging.getLogger(__name__)


@dataclass
class Version:
    """A version of a genome."""
    number: int = 0
    genome: Genome | None = None
    fitness: float = 0.0
    timestamp: float = field(default_factory=time.time)
    change_description: str = ""


class VersionedResource:
    """Genome version management with history and diff."""

    def __init__(self, max_versions: int = 50) -> None:
        self._versions: list[Version] = []
        self._max_versions = max_versions

    def commit(self, genome: Genome, description: str = "") -> Version:
        """Commit current genome as a new version."""
        ver = Version(
            number=len(self._versions),
            genome=copy.deepcopy(genome),
            fitness=genome.fitness,
            change_description=description,
        )
        self._versions.append(ver)
        while len(self._versions) > self._max_versions:
            self._versions.pop(0)
        return ver

    def checkout(self, version: int | None = None) -> Genome | None:
        """Checkout a specific version (None = latest)."""
        if not self._versions:
            return None
        idx = version if version is not None else len(self._versions) - 1
        if 0 <= idx < len(self._versions):
            return copy.deepcopy(self._versions[idx].genome)
        return None

    def diff(self, v1: int, v2: int) -> dict[str, Any]:
        """Compute diff between two versions."""
        g1 = self.checkout(v1)
        g2 = self.checkout(v2)
        if not g1 or not g2:
            return {}
        return {
            "config_diff": {k: (g1.config.get(k), g2.config.get(k))
                          for k in set(g1.config) | set(g2.config)
                          if g1.config.get(k) != g2.config.get(k)},
            "fitness_diff": g2.fitness - g1.fitness,
            "code_changed": g1.code != g2.code,
        }

    @property
    def latest(self) -> Genome | None:
        return self.checkout()

    @property
    def best(self) -> Genome | None:
        if not self._versions:
            return None
        best_ver = max(self._versions, key=lambda v: v.fitness)
        return copy.deepcopy(best_ver.genome)

    @property
    def stats(self) -> dict[str, Any]:
        return {"versions": len(self._versions), "best_fitness": max((v.fitness for v in self._versions), default=0)}
