"""Prometheus V9 Anomaly Detection — Statistical anomaly and deviation detection.

Detects when evolution metrics deviate from expected patterns:
- Fitness drops > 2σ from mean
- Unusual layer utilization patterns
- Memory growth rate anomalies
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Detected anomaly."""
    type: str = ""
    metric: str = ""
    value: float = 0.0
    expected_range: tuple[float, float] = (0.0, 1.0)
    severity: float = 0.0  # 0-1
    description: str = ""


class AnomalyDetector:
    """Statistical anomaly detector using σ-based thresholds.

    Methods:
    1. Z-score: detect values > 2σ from mean
    2. Trend: detect monotonic degradation over N steps
    3. Pattern: detect unusual sequences
    """

    def __init__(self, window_size: int = 50, sigma_threshold: float = 2.0) -> None:
        self._window_size = window_size
        self._sigma_threshold = sigma_threshold
        self._metric_history: dict[str, deque[float]] = {}
        self._anomalies: list[AnomalyReport] = []

    def record_metric(self, name: str, value: float) -> AnomalyReport | None:
        """Record a metric value and check for anomalies."""
        if name not in self._metric_history:
            self._metric_history[name] = deque(maxlen=self._window_size)
        history = self._metric_history[name]
        history.append(value)

        # Need at least 10 values for meaningful stats
        if len(history) < 10:
            return None

        # Z-score anomaly detection
        mean = sum(history) / len(history)
        variance = sum((v - mean) ** 2 for v in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.001
        z_score = abs(value - mean) / std

        if z_score > self._sigma_threshold:
            anomaly = AnomalyReport(
                type="z_score", metric=name, value=value,
                expected_range=(mean - self._sigma_threshold * std, mean + self._sigma_threshold * std),
                severity=min(1.0, z_score / (self._sigma_threshold * 2)),
                description=f"{name}={value:.3f} is {z_score:.1f}σ from mean={mean:.3f}",
            )
            self._anomalies.append(anomaly)
            return anomaly

        # Trend detection: 5+ consecutive decreases
        if len(history) >= 5:
            recent = list(history)[-5:]
            if all(recent[i] > recent[i+1] for i in range(4)):
                drop_pct = abs(recent[0] - recent[-1]) / max(0.001, recent[0])
                anomaly = AnomalyReport(
                    type="trend", metric=name, value=value,
                    severity=min(1.0, drop_pct * 2),
                    description=f"{name} declining for 5 steps: {recent[0]:.3f}→{recent[-1]:.3f}",
                )
                self._anomalies.append(anomaly)
                return anomaly

        return None

    @property
    def recent_anomalies(self) -> list[AnomalyReport]:
        return self._anomalies[-20:]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "metrics_tracked": len(self._metric_history),
            "anomalies_detected": len(self._anomalies),
            "recent_severity": max((a.severity for a in self._anomalies[-10:]), default=0),
        }
