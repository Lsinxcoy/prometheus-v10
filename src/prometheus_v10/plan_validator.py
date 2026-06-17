"""Prometheus V9PRO Plan Validator — 3-layer plan validation.

Validates execution plans at 3 levels:
1. Single-step: each step individually valid
2. Combination: steps together don't create attack vectors
3. Topology: detect multi-step attack chains

Prevents "each step is legal but combination is an attack" scenarios.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.forbidden_ops import ForbiddenOpsChecker

logger = logging.getLogger(__name__)


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    action: str = ""
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    risk_level: float = 0.0  # 0-1


@dataclass
class PlanValidationResult:
    """Result of plan validation across 3 layers."""
    single_step_pass: bool = True
    combination_pass: bool = True
    topology_pass: bool = True
    overall_pass: bool = True
    violations: list[str] = field(default_factory=list)
    risk_score: float = 0.0


# Known attack chain patterns (step sequence → risk)
ATTACK_CHAINS: list[dict[str, Any]] = [
    {
        "name": "credential_harvest",
        "steps": ["read_env", "encode_data", "http_post"],
        "risk": 0.9,
    },
    {
        "name": "privilege_escalation_chain",
        "steps": ["check_permissions", "modify_config", "restart_service"],
        "risk": 0.8,
    },
    {
        "name": "data_exfiltration",
        "steps": ["query_database", "write_file", "transfer_file"],
        "risk": 0.85,
    },
    {
        "name": "persistence_install",
        "steps": ["write_cron", "modify_startup", "hide_process"],
        "risk": 0.95,
    },
]


class PlanValidator:
    """3-layer plan validation: single-step → combination → topology.

    Layer 1: Each step individually passes forbidden_ops check.
    Layer 2: Step combinations don't create known attack vectors.
    Layer 3: Step topology (sequence graph) doesn't match attack chains.
    """
    def __init__(self) -> None:
        self._forbidden_checker = ForbiddenOpsChecker()
        self._validation_count = 0
        self._rejection_count = 0
        self._attack_chain_detections: dict[str, int] = {}

    def validate(self, steps: list[PlanStep]) -> PlanValidationResult:
        """Validate plan across all 3 layers."""
        self._validation_count += 1
        result = PlanValidationResult()

        # Layer 1: Single-step validation
        for i, step in enumerate(steps):
            step_content = f"{step.action} {step.target} {step.params}"
            violations = self._forbidden_checker.check(step_content)
            if violations:
                result.single_step_pass = False
                for v in violations:
                    result.violations.append(f"Step {i}: {v.description}")

        # Layer 2: Combination validation
        result.combination_pass = self._check_combinations(steps, result)

        # Layer 3: Topology validation (attack chain detection)
        result.topology_pass = self._check_topology(steps, result)

        # Overall
        result.overall_pass = (result.single_step_pass and
                               result.combination_pass and
                               result.topology_pass)

        # Risk score
        result.risk_score = max(
            (s.risk_level for s in steps), default=0.0
        )
        if not result.combination_pass:
            result.risk_score = max(result.risk_score, 0.7)
        if not result.topology_pass:
            result.risk_score = max(result.risk_score, 0.8)

        if not result.overall_pass:
            self._rejection_count += 1

        return result

    def _check_combinations(self, steps: list[PlanStep],
                            result: PlanValidationResult) -> bool:
        """Check if step combinations create attack vectors."""
        # Check for read-then-exfiltrate pattern
        has_read = any("read" in s.action.lower() or "query" in s.action.lower() for s in steps)
        has_send = any("post" in s.action.lower() or "send" in s.action.lower() or
                       "upload" in s.action.lower() or "transfer" in s.action.lower()
                       for s in steps)
        if has_read and has_send:
            result.violations.append("Combination: read + send may exfiltrate data")
            return False

        # Check for modify-then-execute pattern
        has_modify = any("write" in s.action.lower() or "modify" in s.action.lower() for s in steps)
        has_exec = any("exec" in s.action.lower() or "run" in s.action.lower() for s in steps)
        if has_modify and has_exec:
            result.violations.append("Combination: modify + execute may inject code")
            return False

        return True

    def _check_topology(self, steps: list[PlanStep],
                        result: PlanValidationResult) -> bool:
        """Check step sequence against known attack chain patterns."""
        step_actions = [s.action.lower() for s in steps]

        for chain in ATTACK_CHAINS:
            chain_steps = [s.lower() for s in chain["steps"]]
            # Check if chain steps appear in order within plan
            chain_idx = 0
            for action in step_actions:
                if chain_idx < len(chain_steps) and chain_steps[chain_idx] in action:
                    chain_idx += 1

            if chain_idx == len(chain_steps):
                result.violations.append(
                    f"Topology: attack chain '{chain['name']}' detected"
                )
                self._attack_chain_detections[chain["name"]] = \
                    self._attack_chain_detections.get(chain["name"], 0) + 1
                return False

        return True

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "validations": self._validation_count,
            "rejections": self._rejection_count,
            "rejection_rate": self._rejection_count / max(1, self._validation_count),
            "attack_chain_detections": dict(self._attack_chain_detections),
        }
