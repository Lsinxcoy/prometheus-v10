"""Prometheus V9PRO Dynamic Security — 4-level adaptive security posture.

Security level adjusts automatically based on threat assessment:
LOW → MEDIUM → HIGH → CRITICAL

Each level has different constraints on what operations are allowed.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.forbidden_ops import ForbiddenOpsChecker, ForbiddenPattern

logger = logging.getLogger(__name__)


class SecurityLevel(str):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Thresholds for auto-escalation
LEVEL_THRESHOLDS: dict[str, dict[str, float]] = {
    SecurityLevel.LOW:      {"max_risk": 0.3, "max_violations_per_hour": 0},
    SecurityLevel.MEDIUM:   {"max_risk": 0.5, "max_violations_per_hour": 2},
    SecurityLevel.HIGH:     {"max_risk": 0.7, "max_violations_per_hour": 5},
    SecurityLevel.CRITICAL: {"max_risk": 1.0, "max_violations_per_hour": 999},
}

# What's allowed at each level
LEVEL_CONSTRAINTS: dict[str, dict[str, bool]] = {
    SecurityLevel.LOW: {
        "allow_code_execution": True,
        "allow_network_access": True,
        "allow_file_write": True,
        "allow_subprocess": True,
        "require_approval": False,
    },
    SecurityLevel.MEDIUM: {
        "allow_code_execution": True,
        "allow_network_access": True,
        "allow_file_write": True,
        "allow_subprocess": False,
        "require_approval": False,
    },
    SecurityLevel.HIGH: {
        "allow_code_execution": True,
        "allow_network_access": False,
        "allow_file_write": False,
        "allow_subprocess": False,
        "require_approval": True,
    },
    SecurityLevel.CRITICAL: {
        "allow_code_execution": False,
        "allow_network_access": False,
        "allow_file_write": False,
        "allow_subprocess": False,
        "require_approval": True,
    },
}


@dataclass
class SecurityEvent:
    """A security-relevant event."""
    event_type: str = ""  # violation | escalation | deescalation | approval
    level: str = ""
    details: str = ""
    timestamp: float = field(default_factory=time.time)


class DynamicSecurity:
    """4-level adaptive security posture with auto-escalation.

    Tracks violations and risk signals to automatically adjust
    the security level. De-escalation requires a cooldown period
    with no violations.

    Integrates with ForbiddenOpsChecker for violation detection
    and PlanValidator for risk assessment.
    """
    def __init__(self, initial_level: str = SecurityLevel.LOW,
                 cooldown_seconds: float = 3600.0) -> None:
        self._level = initial_level
        self._cooldown = cooldown_seconds
        self._forbidden = ForbiddenOpsChecker()
        self._events: deque[SecurityEvent] = deque(maxlen=1000)
        self._violations: deque[float] = deque(maxlen=100)  # timestamps
        self._last_escalation: float = 0.0
        self._last_deescalation: float = 0.0
        self._check_count = 0

    @property
    def level(self) -> str:
        return self._level

    @property
    def constraints(self) -> dict[str, bool]:
        return LEVEL_CONSTRAINTS.get(self._level, LEVEL_CONSTRAINTS[SecurityLevel.HIGH])

    def check_operation(self, content: str, risk_score: float = 0.0) -> dict[str, Any]:
        """Check if an operation is allowed under current security level."""
        self._check_count += 1
        constraints = self.constraints

        # Check forbidden patterns
        violations = self._forbidden.check(content)
        if violations:
            self._record_violation(violations)
            return {
                "allowed": False,
                "reason": "forbidden_pattern",
                "violations": [v.description for v in violations],
                "level": self._level,
            }

        # Check risk score against current level threshold
        threshold = LEVEL_THRESHOLDS.get(self._level, LEVEL_THRESHOLDS[SecurityLevel.HIGH])
        if risk_score > threshold["max_risk"]:
            self._record_violation([f"Risk score {risk_score:.2f} exceeds threshold {threshold['max_risk']}"])
            return {
                "allowed": False,
                "reason": "risk_exceeds_threshold",
                "risk_score": risk_score,
                "max_risk": threshold["max_risk"],
                "level": self._level,
            }

        # Check if approval required
        if constraints["require_approval"]:
            return {
                "allowed": False,
                "reason": "approval_required",
                "level": self._level,
                "note": "Operation requires manual approval at current security level",
            }

        return {"allowed": True, "level": self._level}

    def _record_violation(self, violations: list) -> None:
        """Record a violation and potentially escalate."""
        self._violations.append(time.time())
        self._events.append(SecurityEvent(
            event_type="violation",
            level=self._level,
            details=str(violations)[:200],
        ))
        # Auto-escalate if violation rate exceeds threshold
        self._auto_escalate()

    def _auto_escalate(self) -> None:
        """Escalate security level if violation rate is too high."""
        now = time.time()
        one_hour_ago = now - 3600
        recent_violations = sum(1 for t in self._violations if t > one_hour_ago)

        threshold = LEVEL_THRESHOLDS.get(self._level, {})
        max_violations = threshold.get("max_violations_per_hour", 0)

        if recent_violations > max_violations:
            new_level = self._escalate_level(self._level)
            if new_level != self._level:
                old_level = self._level
                self._level = new_level
                self._last_escalation = now
                self._events.append(SecurityEvent(
                    event_type="escalation",
                    level=new_level,
                    details=f"Escalated from {old_level} (violations: {recent_violations}/hr)",
                ))
                logger.warning(f"Security escalated: {old_level} → {new_level}")

    def _escalate_level(self, current: str) -> str:
        """Get next higher security level."""
        escalation_order = [
            SecurityLevel.LOW, SecurityLevel.MEDIUM,
            SecurityLevel.HIGH, SecurityLevel.CRITICAL,
        ]
        try:
            idx = escalation_order.index(current)
            return escalation_order[min(idx + 1, len(escalation_order) - 1)]
        except ValueError:
            return SecurityLevel.HIGH

    def try_deescalate(self) -> bool:
        """Attempt to lower security level after cooldown."""
        now = time.time()
        if now - self._last_escalation < self._cooldown:
            return False

        # Check no recent violations
        recent = sum(1 for t in self._violations if t > now - self._cooldown)
        if recent > 0:
            return False

        deescalation_order = [
            SecurityLevel.CRITICAL, SecurityLevel.HIGH,
            SecurityLevel.MEDIUM, SecurityLevel.LOW,
        ]
        try:
            idx = deescalation_order.index(self._level)
            if idx < len(deescalation_order) - 1:
                old_level = self._level
                self._level = deescalation_order[idx + 1]
                self._last_deescalation = now
                self._events.append(SecurityEvent(
                    event_type="deescalation",
                    level=self._level,
                    details=f"De-escalated from {old_level}",
                ))
                logger.info(f"Security de-escalated: {old_level} → {self._level}")
                return True
        except ValueError:
            pass
        return False

    def approve(self) -> None:
        """Manually approve an operation at current level (one-time)."""
        self._events.append(SecurityEvent(
            event_type="approval",
            level=self._level,
            details="Manual approval granted",
        ))

    @property
    def stats(self) -> dict[str, Any]:
        now = time.time()
        recent_violations = sum(1 for t in self._violations if t > now - 3600)
        return {
            "current_level": self._level,
            "constraints": self.constraints,
            "checks_performed": self._check_count,
            "recent_violations_1h": recent_violations,
            "total_events": len(self._events),
            "last_escalation": self._last_escalation,
            "last_deescalation": self._last_deescalation,
        }
