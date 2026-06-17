"""Prometheus V9 Initiative — 7-layer governance with real safety/autonomy calls."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.autonomy import AutonomyManager, AutonomyLevel, AutonomyDecision
from prometheus_v10.trust import TrustManager, MoatAssessment
from prometheus_v10.manager import SafetyManager

logger = logging.getLogger(__name__)


class InitiativeLayer:
    """7-layer governance: each layer makes real calls to safety/autonomy/trust."""

    LAYERS = [
        "observation",    # L0: Observe and collect data
        "analysis",       # L1: Analyze observations
        "planning",       # L2: Plan actions
        "safety_check",   # L3: Safety validation (real SafetyManager call)
        "autonomy_check", # L4: Autonomy gate (real AutonomyManager call)
        "execution",      # L5: Execute if approved
        "audit",          # L6: Log and audit
    ]

    def __init__(self, autonomy: AutonomyManager | None = None, trust: TrustManager | None = None, safety: SafetyManager | None = None, max_initiatives_per_hour: int = 10) -> None:
        self._autonomy = autonomy or AutonomyManager()
        self._trust = trust or TrustManager()
        self._safety = safety or SafetyManager()
        self._max_per_hour = max_initiatives_per_hour
        self._initiative_log: list[dict] = []
        self._hourly_count = 0
        self._hourly_reset = time.time()

    def propose(self, action: str, content: str = "", source: str = "system", context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Propose an initiative through 7 governance layers."""
        # Rate limit
        if time.time() - self._hourly_reset > 3600:
            self._hourly_count = 0
            self._hourly_reset = time.time()
        self._hourly_count += 1
        if self._hourly_count > self._max_per_hour:
            return {"approved": False, "reason": "rate_limited", "layer": "rate_limit"}

        result: dict[str, Any] = {"action": action, "layers": {}}

        # L0: Observation
        result["layers"]["observation"] = {"status": "collected", "context_keys": list((context or {}).keys())}

        # L1: Analysis
        moat = self._trust.moat_assess(content, source, context)
        result["layers"]["analysis"] = {"moat_overall": moat.overall, "moat_issues": moat.issues}

        # L2: Planning
        result["layers"]["planning"] = {"action": action, "content_length": len(content)}

        # L3: Safety check (REAL call)
        safety_result = self._safety.check(action, content, confidence=moat.overall, context=context)
        result["layers"]["safety_check"] = {"allowed": safety_result.get("allowed", False), "reason": safety_result.get("reason", "unknown")}
        if not safety_result.get("allowed", False):
            result["approved"] = False
            result["reason"] = "safety_rejected"
            self._log_initiative(result)
            return result

        # L4: Autonomy check (REAL call)
        autonomy_decision = self._autonomy.decide(action, context)
        result["layers"]["autonomy_check"] = {"auto_execute": autonomy_decision.auto_execute, "level": autonomy_decision.level.value, "requires_approval": autonomy_decision.requires_approval}
        if not autonomy_decision.auto_execute and not autonomy_decision.requires_approval:
            result["approved"] = False
            result["reason"] = "autonomy_insufficient"
            self._log_initiative(result)
            return result

        # L5: Execution
        if autonomy_decision.auto_execute:
            result["layers"]["execution"] = {"status": "auto_executed"}
            result["approved"] = True
        elif autonomy_decision.requires_approval:
            result["layers"]["execution"] = {"status": "pending_approval"}
            result["approved"] = "pending"  # Not bool — needs human
        else:
            result["layers"]["execution"] = {"status": "deferred"}
            result["approved"] = False

        # L6: Audit
        result["layers"]["audit"] = {"audit_hash": self._safety.audit_hash, "timestamp": time.time()}
        self._log_initiative(result)
        return result

    def _log_initiative(self, result: dict) -> None:
        self._initiative_log.append({"timestamp": time.time(), "action": result.get("action"), "approved": result.get("approved")})

    @property
    def stats(self) -> dict[str, Any]:
        approved = sum(1 for i in self._initiative_log if i["approved"] is True)
        return {"total": len(self._initiative_log), "approved": approved, "hourly_count": self._hourly_count}
