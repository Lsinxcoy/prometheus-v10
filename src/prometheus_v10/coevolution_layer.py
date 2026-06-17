"""Prometheus V10 Co-Evolution Layer — Unified skill+tool atomic bundle evolution.

Inspired by SkillSmith (arXiv:2606.01314) §3.1:
- Unified proposal space: reflection produces atomic bundles that jointly modify skills and tools
- Vertical attribution: failure from skill orchestration OR tool deficiency
- Horizontal attribution: failure from multi-skill conflict
- Feedback function: structured diagnostics beyond scalar score

Key difference from V9PRO:
- V9PRO: L2Skill and L4Code are independent evolution layers
- V10: CoEvolutionLayer unifies them with atomic bundles + dual attribution
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.anti_pattern import AntiPatternMemory, FailureSignature, CausalAttribution
from prometheus_v10.ecosystem import SkillEcosystem
from prometheus_v10.tool_lifecycle import ProposalBundle, LifecycleEdit, LifecycleOp

logger = logging.getLogger(__name__)


@dataclass
class FailureAnalysis:
    """Analysis of a failure with vertical and horizontal attribution."""
    vertical: str = ""    # "skill_defect" or "tool_deficiency"
    horizontal: str = ""  # "single_component" or "multi_skill_conflict"
    affected_skills: list[str] = field(default_factory=list)
    affected_tools: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    severity: float = 0.0


class CoEvolutionLayer:
    """Unified skill+tool co-evolution with atomic bundle proposals.

    Following SkillSmith §3.1:
    1. Sample candidate from Pareto front
    2. Execute tasks, collect failures
    3. Analyze failures: vertical (skill vs tool) + horizontal (single vs multi)
    4. Generate atomic proposal bundle
    5. Validate and update
    """

    def __init__(self, ecosystem: SkillEcosystem | None = None,
                 anti_pattern: AntiPatternMemory | None = None) -> None:
        self._ecosystem = ecosystem or SkillEcosystem()
        self._anti_pattern = anti_pattern or AntiPatternMemory()
        self._bundle_history: list[ProposalBundle] = []
        self._bundle_id_counter: int = 0

    def analyze_failure(self, failure_trace: dict[str, Any]) -> FailureAnalysis:
        """Analyze a failure with vertical and horizontal attribution.

        Vertical: does the failure stem from a skill orchestration defect
        or from insufficient tool capability?

        Horizontal: does the failure involve interactions among multiple skills?
        """
        affected_skills = failure_trace.get("affected_skills", [])
        affected_tools = failure_trace.get("affected_tools", [])
        error_type = failure_trace.get("error_type", "unknown")
        error_message = failure_trace.get("error_message", "")

        # Vertical attribution
        if "tool_error" in error_type or "tool_timeout" in error_type:
            vertical = "tool_deficiency"
        elif "skill_logic" in error_type or "workflow_error" in error_type:
            vertical = "skill_defect"
        else:
            # Heuristic: if tools are affected, lean toward tool deficiency
            vertical = "tool_deficiency" if affected_tools else "skill_defect"

        # Horizontal attribution
        if len(affected_skills) > 1:
            horizontal = "multi_skill_conflict"
            # Check ecosystem for known conflicts
            for i, skill_a in enumerate(affected_skills):
                for skill_b in affected_skills[i+1:]:
                    conflicts = self._ecosystem.interaction_matrix.get_conflicts(skill_a)
                    if skill_b in conflicts:
                        horizontal = "multi_skill_conflict"
                        break
        else:
            horizontal = "single_component"

        # Get anti-pattern diagnostic context
        signature = FailureSignature(
            failure_type=error_type,
            component=affected_tools[0] if affected_tools else (affected_skills[0] if affected_skills else "unknown"),
            error_pattern=error_message[:100],
        )
        diagnostic_context = self._anti_pattern.get_diagnostic_context(signature)

        diagnostics = [f"Vertical: {vertical}", f"Horizontal: {horizontal}"]
        if diagnostic_context.get("known_patterns"):
            for pattern in diagnostic_context["known_patterns"]:
                diagnostics.append(f"Known pattern: {pattern.get('root_cause', 'unknown')}")

        # Record the failure
        attribution = CausalAttribution(
            root_cause=vertical,
            affected_skills=affected_skills,
            affected_tools=affected_tools,
            confidence=0.7 if vertical != "skill_defect" else 0.5,
        )
        self._anti_pattern.record_failure(signature, attribution)

        severity = min(1.0, 0.3 + 0.2 * len(affected_skills) + 0.3 * len(affected_tools))

        return FailureAnalysis(
            vertical=vertical,
            horizontal=horizontal,
            affected_skills=affected_skills,
            affected_tools=affected_tools,
            diagnostics=diagnostics,
            severity=severity,
        )

    def propose_bundle(self, analysis: FailureAnalysis) -> ProposalBundle:
        """Generate an atomic proposal bundle from failure analysis.

        The bundle jointly modifies skills and tools to address the root cause.
        Atomicity ensures interdependent changes take effect simultaneously.
        """
        self._bundle_id_counter += 1
        bundle_id = f"bundle_{self._bundle_id_counter}"

        skill_edits: list[dict[str, Any]] = []
        tool_edits: list[LifecycleEdit] = []

        if analysis.vertical == "tool_deficiency":
            # Tool is the bottleneck → patch or wrap the tool
            for tool in analysis.affected_tools:
                tool_edits.append(LifecycleEdit(
                    op=LifecycleOp.EDIT,
                    target=tool,
                    patch=f"Fix {analysis.diagnostics[0]}",
                ))
            # Also update skills that depend on the tool
            for skill in analysis.affected_skills:
                skill_edits.append({
                    "skill": skill,
                    "change": "update_tool_invocation",
                    "rationale": f"Adapt to patched tool {analysis.affected_tools}",
                })

        elif analysis.vertical == "skill_defect":
            # Skill orchestration is broken → rewrite skill workflow
            for skill in analysis.affected_skills:
                skill_edits.append({
                    "skill": skill,
                    "change": "rewrite_workflow",
                    "rationale": f"Fix {analysis.diagnostics[0]}",
                })
            # No tool changes needed

        if analysis.horizontal == "multi_skill_conflict":
            # Skills conflict → may need to compose or retire conflicting tools
            for i, skill_a in enumerate(analysis.affected_skills):
                for skill_b in analysis.affected_skills[i+1:]:
                    conflicts = self._ecosystem.interaction_matrix.get_conflicts(skill_a)
                    if skill_b in conflicts:
                        skill_edits.append({
                            "skill": skill_a,
                            "change": "add_conflict_guard",
                            "rationale": f"Guard against conflict with {skill_b}",
                        })

        bundle = ProposalBundle(
            skill_edits=skill_edits,
            tool_edits=tool_edits,
            bundle_id=bundle_id,
            rationale="; ".join(analysis.diagnostics),
            vertical_attribution=analysis.vertical,
            horizontal_attribution=analysis.horizontal,
        )

        # Check anti-pattern veto
        for edit in tool_edits:
            should_veto, reason = self._anti_pattern.should_veto(
                proposal_type=edit.op.value,
                target=edit.target,
            )
            if should_veto:
                logger.warning(f"Bundle {bundle_id} vetoed: {reason}")
                bundle.rationale += f" [VETOED: {reason}]"
                return bundle  # return vetoed bundle (caller checks)

        self._bundle_history.append(bundle)
        return bundle

    @property
    def ecosystem(self) -> SkillEcosystem:
        return self._ecosystem

    @property
    def anti_pattern(self) -> AntiPatternMemory:
        return self._anti_pattern

    def stats(self) -> dict[str, Any]:
        return {
            "total_bundles": len(self._bundle_history),
            "tool_deficiency_bundles": sum(1 for b in self._bundle_history if b.vertical_attribution == "tool_deficiency"),
            "skill_defect_bundles": sum(1 for b in self._bundle_history if b.vertical_attribution == "skill_defect"),
            "multi_skill_conflicts": sum(1 for b in self._bundle_history if b.horizontal_attribution == "multi_skill_conflict"),
        }
