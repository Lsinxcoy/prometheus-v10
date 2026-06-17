"""Prometheus V9PRO Operational Mirror — RL pathology detection + Deterministic Gate.

Integrates HarnessX §4.1-4.2 (Operational Mirror + Pathology detection).
Maps harness evolution onto RL constructs:
  Policy π → Harness-update procedure
  State s_t → (H_t, T_t)
  Action a_t → Typed harness edit
  Feedback → Trace τ + verifier score r
  Update → H_{t+1} ← U(Ĥ_t, T_t, r_t) [deterministic gate]

Three pathologies:
1. Reward hacking → Evolver constructs structured exploits
2. Catastrophic forgetting → Shared component edits propagate non-locally
3. Under-exploration → Bias toward low-risk local edits
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import HarnessConfig, HarnessEdit, HookPoint
from prometheus_v9pro.trace_store import TraceRecord, TraceStore

logger = logging.getLogger(__name__)


# ── Risk Assessment ────────────────────────────────────────────────

@dataclass
class RiskAssessment:
    """Assessment of a specific RL pathology in symbolic space."""
    pathology: str = ""  # reward_hacking | catastrophic_forgetting | under_exploration
    severity: float = 0.0  # 0-1
    evidence: list[str] = field(default_factory=list)
    recommendation: str = ""  # reject | require_revision | accept_with_monitoring


# ── Gate Result ────────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of deterministic gate evaluation."""
    accepted: bool = False
    check_results: dict[str, bool] = field(default_factory=dict)
    rejection_reason: str = ""


# ── Operational Mirror ─────────────────────────────────────────────

class OperationalMirror:
    """Maps harness evolution onto RL constructs. HarnessX §4.1-4.2.

    The mirror is predictive: it identifies three failure modes analogous
    to known RL pathologies. AEGIS addresses each with a dedicated mechanism:
    - Reward hacking → Critic
    - Catastrophic forgetting → DeterministicGate + seesaw
    - Under-exploration → Planner + AdaptationLandscape

    Note: This is a design heuristic (checklist), not a formal framework.
    The three pathologies are representative, not exhaustive.
    """
    def __init__(self) -> None:
        self._assessments: deque[RiskAssessment] = deque(maxlen=500)
        self._hacking_detections = 0
        self._forgetting_detections = 0
        self._exploration_detections = 0

    def detect_reward_hacking(self, candidate: HarnessEdit,
                               traces: TraceStore) -> RiskAssessment:
        """Detect reward hacking patterns. HarnessX §4.2.

        In standard RL, reward hacking exploits loopholes in the reward signal.
        Symbolic harness evolution amplifies this: the evolver can target the
        verification protocol directly — embedding benchmark answers into prompts,
        exploiting format regularities, or introducing processors that rewrite
        outputs to match verifier expectations.

        HarnessX evidence: GAIA R10, tool+prompt composite passed seesaw but
        subset passed via format regularity, not genuine retrieval.
        """
        assessment = RiskAssessment(pathology="reward_hacking")

        # Check manifest for suspicious patterns
        manifest = candidate.manifest
        components = manifest.get("changed_components", [])

        # Pattern 1: Edit adds context that contains specific task answers
        for comp in components:
            if "answer" in comp.lower() or "solution" in comp.lower():
                assessment.severity = 0.8
                assessment.evidence.append(f"Component {comp} may embed answers")
                assessment.recommendation = "reject"

        # Pattern 2: Edit modifies verification/evaluation pipeline
        for comp in components:
            if "eval" in comp.lower() or "verif" in comp.lower() or "reward" in comp.lower():
                assessment.severity = min(1.0, assessment.severity + 0.5)
                assessment.evidence.append(f"Component {comp} modifies evaluation pipeline")

        # Pattern 3: Check if trace shows format exploitation
        failing = traces.get_failing_tasks()
        for task_id, task_traces in failing.items():
            for trace in task_traces[-3:]:
                if "format" in trace.failure_category.lower():
                    assessment.severity = min(1.0, assessment.severity + 0.3)
                    assessment.evidence.append(f"Task {task_id}: format exploitation detected")

        if assessment.severity >= 0.5 and not assessment.recommendation:
            assessment.recommendation = "require_revision"
        elif assessment.severity > 0 and not assessment.recommendation:
            assessment.recommendation = "accept_with_monitoring"
        elif not assessment.recommendation:
            assessment.recommendation = "accept_with_monitoring"

        if assessment.severity > 0:
            self._hacking_detections += 1
        self._assessments.append(assessment)
        return assessment

    def detect_catastrophic_forgetting(self, candidate: HarnessEdit,
                                        traces: TraceStore) -> RiskAssessment:
        """Detect catastrophic forgetting patterns. HarnessX §4.2.

        In symbolic harness evolution, an edit that repairs failure pattern A
        can silently regress pattern B, because effects propagate through shared
        context, tools, memory policies, and control rules.

        HarnessX evidence: τ³-Bench Telecom R7, 6 consecutive same-type edits
        accumulated sub-threshold coupling → -14.0% regression.
        GAIA GPT-5.4 Global: 73.8% → 49.5% (peak–final gap: -24.3%).
        """
        assessment = RiskAssessment(pathology="catastrophic_forgetting")

        # Check: edit modifies shared component (affects multiple tasks)
        manifest = candidate.manifest
        components = manifest.get("changed_components", [])

        shared_components = {"context", "memory", "control", "processors.step_start",
                            "processors.before_model", "processors.after_model"}
        for comp in components:
            comp_lower = comp.lower()
            for shared in shared_components:
                if shared in comp_lower:
                    assessment.severity = min(1.0, assessment.severity + 0.3)
                    assessment.evidence.append(f"Shared component: {comp}")

        # Check: recent history shows regression pattern
        implicated = traces.get_implicated_components()
        for comp in components:
            comp_lower = comp.lower()
            for imp_comp, count in implicated.items():
                if imp_comp.lower() in comp_lower and count > 3:
                    assessment.severity = min(1.0, assessment.severity + 0.2)
                    assessment.evidence.append(f"Component {comp} implicated {count} times in failures")

        if assessment.severity >= 0.5:
            assessment.recommendation = "require_revision"
        elif assessment.severity > 0:
            assessment.recommendation = "accept_with_monitoring"
        else:
            assessment.recommendation = "accept_with_monitoring"

        if assessment.severity > 0:
            self._forgetting_detections += 1
        self._assessments.append(assessment)
        return assessment

    def detect_under_exploration(self, edit_history: list[dict[str, Any]],
                                  traces: TraceStore) -> RiskAssessment:
        """Detect under-exploration patterns. HarnessX §4.2.

        Manifests as a bias toward low-risk local edits: prompt rephrasing,
        tool-description tuning, or minor control-flow tweaks. These edits
        are cheap to generate and frequently pass gating, biasing subsequent
        Planner hypotheses toward the same edit neighborhood.

        HarnessX evidence: ALFWorld R4-R7, <1% gain/round,
        ship-prediction 80%→0%, prompt-space exhausted.
        """
        assessment = RiskAssessment(pathology="under_exploration")

        if not edit_history:
            assessment.recommendation = "accept_with_monitoring"
            self._assessments.append(assessment)
            return assessment

        # Check: concentration of edit types
        type_counts: dict[str, int] = {}
        for entry in edit_history:
            t = entry.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        total = len(edit_history)
        if total > 0:
            max_type = max(type_counts.values())
            concentration = max_type / total
            if concentration > 0.7:
                assessment.severity = min(1.0, concentration)
                dominant_type = max(type_counts, key=type_counts.get)
                assessment.evidence.append(
                    f"Edit type '{dominant_type}' dominates: {max_type}/{total} ({concentration:.0%})"
                )

        # Check: recent edits are all same type
        if len(edit_history) >= 4:
            recent_types = [e.get("type", "") for e in edit_history[-4:]]
            if len(set(recent_types)) == 1:
                assessment.severity = min(1.0, assessment.severity + 0.3)
                assessment.evidence.append(f"Last 4 edits all type '{recent_types[0]}'")

        # Check: dimension coverage
        dimensions_touched: set[str] = set()
        for entry in edit_history:
            for dim in entry.get("dimensions", []):
                dimensions_touched.add(dim)
        untouched = 9 - len(dimensions_touched)
        if untouched > 5:
            assessment.severity = min(1.0, assessment.severity + 0.2)
            assessment.evidence.append(f"Only {9 - untouched}/9 dimensions explored")

        if assessment.severity >= 0.5:
            assessment.recommendation = "require_revision"
        elif assessment.severity > 0:
            assessment.recommendation = "accept_with_monitoring"
        else:
            assessment.recommendation = "accept_with_monitoring"

        if assessment.severity > 0:
            self._exploration_detections += 1
        self._assessments.append(assessment)
        return assessment

    @property
    def stats(self) -> dict[str, Any]:
        severity_dist = {"none": 0, "low": 0, "medium": 0, "high": 0}
        for a in self._assessments:
            if a.severity == 0:
                severity_dist["none"] += 1
            elif a.severity < 0.3:
                severity_dist["low"] += 1
            elif a.severity < 0.7:
                severity_dist["medium"] += 1
            else:
                severity_dist["high"] += 1
        return {
            "total_assessments": len(self._assessments),
            "hacking_detections": self._hacking_detections,
            "forgetting_detections": self._forgetting_detections,
            "exploration_detections": self._exploration_detections,
            "severity_distribution": severity_dist,
        }


# ── Deterministic Gate (HarnessX §4.3) ────────────────────────────

class DeterministicGate:
    """Decouples LLM judgment from acceptance. HarnessX §4.3.

    Sequence: manifest_completeness → config_normalization →
    build/smoke_test → seesaw_constraint.
    First failing check halts; passing candidates are committed.

    Design principle: Language-model subagents explore, hypothesize, and propose;
    typed structure and deterministic gates determine what ships.
    """
    def __init__(self) -> None:
        self._evaluations = 0
        self._acceptances = 0
        self._rejections = 0

    def evaluate(self, candidate: HarnessEdit,
                 current: HarnessConfig,
                 traces: TraceStore | None = None) -> GateResult:
        """Run deterministic gate checks in sequence."""
        self._evaluations += 1
        result = GateResult()

        # Check 1: Manifest completeness
        manifest = candidate.manifest
        has_components = bool(manifest.get("changed_components"))
        has_effect = bool(manifest.get("intended_effect"))
        result.check_results["manifest_completeness"] = has_components and has_effect

        if not result.check_results["manifest_completeness"]:
            result.accepted = False
            result.rejection_reason = "manifest_incomplete"
            self._rejections += 1
            return result

        # Check 2: Config normalization (canonical form)
        # Verify the edit action is valid
        valid_actions = {"insert", "replace", "remove"}
        result.check_results["config_normalization"] = candidate.action in valid_actions

        if not result.check_results["config_normalization"]:
            result.accepted = False
            result.rejection_reason = "invalid_action"
            self._rejections += 1
            return result

        # Check 3: Smoke test (if applicable)
        smoke_passed = manifest.get("smoke_test_passed", True)
        result.check_results["smoke_test"] = smoke_passed

        if not result.check_results["smoke_test"]:
            result.accepted = False
            result.rejection_reason = "smoke_test_failed"
            self._rejections += 1
            return result

        # Check 4: Seesaw constraint (regression check)
        predicted_regress = manifest.get("tasks_expected_regress", [])
        no_regression = len(predicted_regress) == 0
        result.check_results["seesaw_constraint"] = no_regression

        if not result.check_results["seesaw_constraint"]:
            result.accepted = False
            result.rejection_reason = "regression_predicted"
            self._rejections += 1
            return result

        # All checks passed
        result.accepted = True
        self._acceptances += 1
        return result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "evaluations": self._evaluations,
            "acceptances": self._acceptances,
            "rejections": self._rejections,
            "acceptance_rate": self._acceptances / max(1, self._evaluations),
        }
