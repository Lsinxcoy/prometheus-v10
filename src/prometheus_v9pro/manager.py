"""Prometheus V9 Safety Manager — Serializes breaker+gate+validator, T5 degradation coordination."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from prometheus_v9pro.breaker import PerToolCircuitBreaker, BreakerConfig
from prometheus_v9pro.gate import ConfidenceGate, GateDecision, GateResult
from prometheus_v9pro.validator import ChainValidator, ValidationResult

logger = logging.getLogger(__name__)


class SafetyManager:
    """Serializes all safety modules: breaker → gate → validator. T5 degradation coordination."""

    def __init__(self, breaker: PerToolCircuitBreaker | None = None, gate: ConfidenceGate | None = None, validator: ChainValidator | None = None) -> None:
        self._breaker = breaker or PerToolCircuitBreaker()
        self._gate = gate or ConfidenceGate()
        self._validator = validator or ChainValidator()
        self._violation_log: list[dict] = []
        self._audit_hash: str = ""

    def check(self, tool_name: str, code: str, confidence: float = 0.8, existing_knowledge: list[str] | None = None, context: dict | None = None) -> dict[str, Any]:
        """Full safety check: breaker → gate → validator."""
        # Always update audit hash for traceability (even on rejection)
        self._update_audit_hash(code)

        # Step 1: Circuit breaker check
        if not self._breaker.can_execute(tool_name):
            self._log_violation(tool_name, "breaker_open", "Circuit breaker is open")
            self._breaker.record_failure(tool_name)
            return {"allowed": False, "reason": "circuit_breaker_open", "breaker_state": self._breaker.get_state(tool_name).value}

        # Step 2: Confidence gate + OEP check
        gate_result = self._gate.evaluate(code, "code", confidence, existing_knowledge)
        if gate_result.decision == GateDecision.DEFER:
            self._log_violation(tool_name, "gate_defer", gate_result.reason)
            return {"allowed": False, "reason": "confidence_defer", "confidence": confidence, "oep_suspect": gate_result.oep_suspect}
        if gate_result.oep_suspect:
            self._log_violation(tool_name, "oep_suspect", "Clean experience poisoning suspected")

        # Step 3: Chain validation
        validation = self._validator.validate(code, context)
        if not validation.valid:
            self._log_violation(tool_name, "validation_failed", str(validation.issues))
            self._breaker.record_failure(tool_name)
            return {"allowed": False, "reason": "validation_failed", "issues": validation.issues, "attack_chain": validation.attack_chain_detected}

        # All checks passed
        self._breaker.record_success(tool_name)
        needs_approval = gate_result.decision == GateDecision.ASK

        return {
            "allowed": True,
            "needs_approval": needs_approval,
            "confidence": confidence,
            "validation": validation.dimensions,
            "oep_suspect": gate_result.oep_suspect,
            "breaker_states": self._breaker.get_all_states(),
        }

    def _log_violation(self, tool_name: str, violation_type: str, detail: str) -> None:
        self._violation_log.append({"tool": tool_name, "type": violation_type, "detail": detail, "timestamp": time.time()})

    def _update_audit_hash(self, code: str) -> None:
        self._audit_hash = hashlib.sha256((self._audit_hash + code).encode()).hexdigest()[:16]

    @property
    def audit_hash(self) -> str:
        return self._audit_hash

    @property
    def violation_count(self) -> int:
        return len(self._violation_log)
