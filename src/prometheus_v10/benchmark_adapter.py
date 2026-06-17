"""Prometheus V10 Benchmark Adapter — External real-task fitness anchoring.

Inspired by ALE (arXiv:2606.05405):
- Anchor internal fitness to real-world task performance
- Long-horizon evaluation: sustained N-round success rate
- Economic value proxy: correlate internal metrics with external task pass rate

Key difference from V9PRO:
- V9PRO: ThreeStageFitness (complexity+diversity+correctness) — internal only
- V10: benchmark_adapter.py bridges internal fitness ↔ external task performance
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from one benchmark task evaluation."""
    task_id: str = ""
    task_domain: str = ""
    passed: bool = False
    score: float = 0.0
    internal_fitness: float = 0.0  # what ThreeStageFitness reported
    duration_seconds: float = 0.0


@dataclass
class CorrelationReport:
    """Correlation between internal fitness and external benchmark performance."""
    pearson_r: float = 0.0
    sample_size: int = 0
    calibration_slope: float = 1.0  # how much to scale internal fitness


class BenchmarkAdapter:
    """Bridge between internal evolution fitness and external task performance.

    Following ALE's philosophy:
    - Internal fitness is necessary but not sufficient
    - External task pass rate is the ultimate signal
    - Calibration: learn the mapping internal → external
    """

    def __init__(self, min_correlation_target: float = 0.7) -> None:
        self._results: deque[BenchmarkResult] = deque(maxlen=1000)
        self._min_correlation = min_correlation_target
        self._calibration_slope: float = 1.0
        self._calibrated: bool = False

    def record(self, task_id: str, domain: str, passed: bool,
               score: float, internal_fitness: float) -> None:
        """Record a benchmark result."""
        self._results.append(BenchmarkResult(
            task_id=task_id,
            task_domain=domain,
            passed=passed,
            score=score,
            internal_fitness=internal_fitness,
        ))

    def compute_correlation(self) -> CorrelationReport:
        """Compute correlation between internal fitness and external score."""
        if len(self._results) < 10:
            return CorrelationReport(sample_size=len(self._results))

        internal_scores = [r.internal_fitness for r in self._results]
        external_scores = [r.score for r in self._results]

        # Pearson correlation
        n = len(internal_scores)
        mean_i = sum(internal_scores) / n
        mean_e = sum(external_scores) / n

        cov = sum((i - mean_i) * (e - mean_e) for i, e in zip(internal_scores, external_scores)) / n
        std_i = (sum((i - mean_i) ** 2 for i in internal_scores) / n) ** 0.5
        std_e = (sum((e - mean_e) ** 2 for e in external_scores) / n) ** 0.5

        pearson_r = cov / (std_i * std_e) if std_i > 0 and std_e > 0 else 0.0

        # Calibration slope: linear regression coefficient
        if std_i > 0:
            slope = cov / (std_i ** 2)
        else:
            slope = 1.0

        self._calibration_slope = slope
        self._calibrated = True

        return CorrelationReport(
            pearson_r=pearson_r,
            sample_size=n,
            calibration_slope=slope,
        )

    def calibrated_fitness(self, internal_fitness: float) -> float:
        """Adjust internal fitness using calibration slope."""
        if not self._calibrated:
            return internal_fitness
        return max(0.0, min(1.0, internal_fitness * self._calibration_slope))

    def long_horizon_pass_rate(self, n_rounds: int = 10) -> float:
        """Compute sustained pass rate over N rounds (ALE-style evaluation)."""
        recent = list(self._results)[-n_rounds * 10:]  # last N*10 results
        if not recent:
            return 0.0
        return sum(1 for r in recent if r.passed) / len(recent)

    def is_calibrated(self) -> bool:
        """Check if calibration meets minimum correlation target."""
        if not self._calibrated or len(self._results) < 10:
            return False
        report = self.compute_correlation()
        return abs(report.pearson_r) >= self._min_correlation

    def stats(self) -> dict[str, Any]:
        return {
            "total_results": len(self._results),
            "calibrated": self._calibrated,
            "calibration_slope": self._calibration_slope,
            "pass_rate": self.long_horizon_pass_rate(),
        }
