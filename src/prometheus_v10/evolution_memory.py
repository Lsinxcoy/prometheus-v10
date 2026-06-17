"""Prometheus V9 Evolution Memory — Cross-generation learning history.

Tracks what worked and what didn't across generations, so the engine
doesn't repeat the same failed experiments.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvolutionRecord:
    """Record of one evolution attempt."""
    generation: int = 0
    layer: int = 0
    strategy: str = ""
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    success: bool = False
    timestamp: float = field(default_factory=time.time)


class EvolutionMemory:
    """Cross-generation learning history.

    Enables:
    1. Avoid repeating failed strategies
    2. Bias toward previously successful strategies
    3. Detect regime changes (what worked before no longer works)
    """

    def __init__(self, max_records: int = 10000) -> None:
        self._records: deque[EvolutionRecord] = deque(maxlen=max_records)
        self._strategy_stats: dict[str, dict[str, float]] = {}

    def record(self, gen: int, layer: int, strategy: str,
               fitness_before: float, fitness_after: float) -> None:
        """Record an evolution attempt."""
        rec = EvolutionRecord(
            generation=gen, layer=layer, strategy=strategy,
            fitness_before=fitness_before, fitness_after=fitness_after,
            success=fitness_after > fitness_before,
        )
        self._records.append(rec)

        # Update strategy stats
        if strategy not in self._strategy_stats:
            self._strategy_stats[strategy] = {"attempts": 0, "successes": 0, "avg_delta": 0.0}
        stats = self._strategy_stats[strategy]
        stats["attempts"] += 1
        if rec.success:
            stats["successes"] += 1
        delta = fitness_after - fitness_before
        stats["avg_delta"] = (stats["avg_delta"] * (stats["attempts"] - 1) + delta) / stats["attempts"]

    def get_best_strategies(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top N strategies by success rate."""
        scored = []
        for strategy, stats in self._strategy_stats.items():
            rate = stats["successes"] / max(1, stats["attempts"])
            scored.append((strategy, rate))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def should_avoid(self, strategy: str) -> bool:
        """Check if a strategy has been consistently failing."""
        stats = self._strategy_stats.get(strategy)
        if not stats or stats["attempts"] < 3:
            return False
        rate = stats["successes"] / stats["attempts"]
        return rate < 0.2 and stats["attempts"] >= 5

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "strategies_tracked": len(self._strategy_stats),
            "best_strategies": self.get_best_strategies(3),
        }
