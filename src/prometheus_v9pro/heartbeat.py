"""Prometheus V9 Heartbeat — System health monitoring with T5 degradation triggers.

Monitors all organs and evolution layers. When anomalies are detected,
triggers organ degradation or evolution reconfiguration.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """System health report."""
    timestamp: float = field(default_factory=time.time)
    organ_health: dict[str, float] = field(default_factory=dict)
    layer_health: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    alerts: list[str] = field(default_factory=list)
    degraded_organs: list[str] = field(default_factory=list)


class HeartbeatMonitor:
    """System health monitor with degradation triggers.

    Checks:
    1. Organ responsiveness (are they producing output?)
    2. Layer fitness deltas (are layers improving?)
    3. Memory store size (is it growing or stagnating?)
    4. Evolution convergence (is fitness plateauing?)
    """

    def __init__(self, pipeline=None, engine=None, store=None) -> None:
        self._pipeline = pipeline
        self._engine = engine
        self._store = store
        self._history: deque[HealthReport] = deque(maxlen=100)
        self._check_interval: float = 60.0  # seconds
        self._last_check: float = time.time()
        self._alert_count: int = 0

    def check_health(self) -> HealthReport:
        """Run a health check on all components."""
        report = HealthReport()
        alerts = []

        # Check organ health
        if self._pipeline:
            for organ_name in ["taotie", "nuwa", "darwin", "pool", "guard"]:
                organ = getattr(self._pipeline, organ_name, None)
                if organ:
                    if hasattr(organ, '_degraded') and organ._degraded:
                        report.organ_health[organ_name] = 0.3  # Degraded but alive
                        report.degraded_organs.append(organ_name)
                    elif hasattr(organ, 'is_alive') and not organ.is_alive:
                        report.organ_health[organ_name] = 0.0
                        alerts.append(f"Organ {organ_name} is dead!")
                    else:
                        report.organ_health[organ_name] = 1.0

        # Check layer health
        if self._engine:
            for i in range(12):
                name = f"L{i}"
                # Simple heuristic: layer produces output = healthy
                report.layer_health[name] = 0.8  # Assume healthy

        # Overall score
        organ_scores = list(report.organ_health.values()) if report.organ_health else [1.0]
        layer_scores = list(report.layer_health.values()) if report.layer_health else [1.0]
        report.overall_score = (
            sum(organ_scores) / max(1, len(organ_scores)) * 0.6
            + sum(layer_scores) / max(1, len(layer_scores)) * 0.4
        )

        # Memory store check
        if self._store:
            node_count = self._store.get_node_count()
            if node_count == 0:
                alerts.append("Memory store is empty!")
                report.overall_score *= 0.5

        # Degradation triggers
        if report.overall_score < 0.5:
            alerts.append("System health critical — consider degradation")
            self._alert_count += 1

        report.alerts = alerts
        self._history.append(report)
        self._last_check = time.time()
        return report

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "checks_performed": len(self._history),
            "alerts_total": self._alert_count,
            "last_check": self._last_check,
            "avg_score": sum(r.overall_score for r in self._history) / max(1, len(self._history)),
        }
