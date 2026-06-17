"""Prometheus V10 Workflow Memory — Procedural memory with conditional branching + personalization.

Inspired by DRFLOW (arXiv:2606.18191):
- Store action-step sequences (not just facts/semantics)
- Generic→personalized: company docs define generic flow, personal docs determine branch
- Conditional resolution: if appointment=tenure-track → Provost approval
- 7 metrics: factuality/recall/precision/F1/condition_resolution/topology_order/personalization

Key difference from V9PRO:
- V9PRO store.py: SQLiteStore with facts, nodes, FTS5 search
- V10 workflow_memory.py: procedural step sequences with conditional branching
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStep:
    """One step in a workflow."""
    step_id: str = ""
    description: str = ""
    action: str = ""             # the action to take
    tool_required: str = ""      # which tool to use
    condition: str = ""          # condition for this step (empty = unconditional)
    condition_branch: str = ""   # "if_true" or "if_false"
    evidence_sources: list[str] = field(default_factory=list)  # source documents
    order: int = 0               # execution order


@dataclass
class WorkflowTemplate:
    """A generic workflow template (from company/org documents)."""
    template_id: str = ""
    name: str = ""
    domain: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    conditions: list[dict[str, str]] = field(default_factory=list)  # {condition, true_branch, false_branch}
    source_documents: list[str] = field(default_factory=list)


@dataclass
class PersonalizedWorkflow:
    """A personalized workflow resolved from template + personal context."""
    workflow_id: str = ""
    template_id: str = ""
    user_context: dict[str, Any] = field(default_factory=dict)
    resolved_steps: list[WorkflowStep] = field(default_factory=list)
    condition_resolutions: dict[str, bool] = field(default_factory=dict)  # condition → resolved value
    personalization_score: float = 0.0


@dataclass
class WorkflowEvaluation:
    """7-metric evaluation of a predicted workflow."""
    factuality: float = 0.0     # citations support claims
    recall: float = 0.0         # coverage of golden steps
    precision: float = 0.0      # correctness of predicted steps
    f1: float = 0.0             # harmonic mean of precision and recall
    condition_resolution: float = 0.0  # correct branch resolution
    topology_order: float = 0.0        # correct step ordering
    personalization: float = 0.0       # use of personal evidence


class WorkflowMemory:
    """Procedural memory with conditional branching and personalization.

    Two-layer design (from DRFLOW):
    1. Generic templates: extracted from organization documents
    2. Personalized workflows: resolved from templates using user context
    """

    def __init__(self) -> None:
        self._templates: dict[str, WorkflowTemplate] = {}
        self._personalized: dict[str, PersonalizedWorkflow] = {}
        self._template_id_counter: int = 0

    def store_template(self, name: str, domain: str,
                       steps: list[dict[str, Any]],
                       conditions: list[dict[str, str]] | None = None,
                       source_documents: list[str] | None = None) -> WorkflowTemplate:
        """Store a generic workflow template."""
        self._template_id_counter += 1
        template_id = f"wf_tpl_{self._template_id_counter}"

        workflow_steps = []
        for i, step_dict in enumerate(steps):
            step = WorkflowStep(
                step_id=f"{template_id}_s{i}",
                description=step_dict.get("description", ""),
                action=step_dict.get("action", ""),
                tool_required=step_dict.get("tool", ""),
                condition=step_dict.get("condition", ""),
                condition_branch=step_dict.get("branch", ""),
                order=i,
            )
            workflow_steps.append(step)

        template = WorkflowTemplate(
            template_id=template_id,
            name=name,
            domain=domain,
            steps=workflow_steps,
            conditions=conditions or [],
            source_documents=source_documents or [],
        )
        self._templates[template_id] = template
        return template

    def resolve_workflow(self, template_id: str,
                         user_context: dict[str, Any]) -> PersonalizedWorkflow | None:
        """Resolve a personalized workflow from template + user context.

        Following DRFLOW: company documents define generic process,
        personal documents determine which branch applies.
        """
        template = self._templates.get(template_id)
        if not template:
            return None

        # Resolve conditions based on user context
        condition_resolutions: dict[str, bool] = {}
        for cond in template.conditions:
            condition_key = cond.get("condition", "")
            # Check if user context provides a value for this condition
            if condition_key in user_context:
                condition_resolutions[condition_key] = bool(user_context[condition_key])

        # Filter steps based on resolved conditions
        resolved_steps: list[WorkflowStep] = []
        for step in template.steps:
            if not step.condition:
                # Unconditional step → always include
                resolved_steps.append(step)
            elif step.condition in condition_resolutions:
                # Conditional step → include if branch matches resolution
                resolved = condition_resolutions[step.condition]
                if step.condition_branch == "if_true" and resolved:
                    resolved_steps.append(step)
                elif step.condition_branch == "if_false" and not resolved:
                    resolved_steps.append(step)
            else:
                # Unknown condition → include with caution flag
                resolved_steps.append(step)

        # Compute personalization score
        personalization = len(condition_resolutions) / max(1, len(template.conditions))

        workflow_id = f"wf_{template_id}_{int(time.time())}"
        workflow = PersonalizedWorkflow(
            workflow_id=workflow_id,
            template_id=template_id,
            user_context=user_context,
            resolved_steps=resolved_steps,
            condition_resolutions=condition_resolutions,
            personalization_score=personalization,
        )
        self._personalized[workflow_id] = workflow
        return workflow

    def evaluate_workflow(self, predicted: PersonalizedWorkflow,
                          ground_truth: PersonalizedWorkflow) -> WorkflowEvaluation:
        """Evaluate a predicted workflow against ground truth using 7 DRFLOW metrics."""
        # Step-level matching
        pred_descriptions = {s.description for s in predicted.resolved_steps}
        gt_descriptions = {s.description for s in ground_truth.resolved_steps}

        if not gt_descriptions:
            return WorkflowEvaluation()

        # Recall: coverage of golden steps
        recall = len(pred_descriptions & gt_descriptions) / len(gt_descriptions) if gt_descriptions else 0.0

        # Precision: correctness of predicted steps
        precision = len(pred_descriptions & gt_descriptions) / len(pred_descriptions) if pred_descriptions else 0.0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Condition resolution accuracy
        cond_correct = 0
        cond_total = 0
        for cond_key, gt_val in ground_truth.condition_resolutions.items():
            cond_total += 1
            if predicted.condition_resolutions.get(cond_key) == gt_val:
                cond_correct += 1
        condition_resolution = cond_correct / max(1, cond_total)

        # Topology order (check if step ordering is consistent)
        order_correct = 0
        order_total = 0
        pred_steps_list = list(predicted.resolved_steps)
        gt_steps_list = list(ground_truth.resolved_steps)
        for i, gt_step in enumerate(gt_steps_list):
            for j, pred_step in enumerate(pred_steps_list):
                if pred_step.description == gt_step.description:
                    order_total += 1
                    # Check relative ordering with neighbors
                    if i > 0 and j > 0:
                        if pred_steps_list[j-1].description == gt_steps_list[i-1].description:
                            order_correct += 1
                    break
        topology_order = order_correct / max(1, order_total)

        # Personalization: use of personal evidence
        personalization = predicted.personalization_score

        # Factuality: evidence sources overlap
        pred_sources = set()
        gt_sources = set()
        for s in predicted.resolved_steps:
            pred_sources.update(s.evidence_sources)
        for s in ground_truth.resolved_steps:
            gt_sources.update(s.evidence_sources)
        factuality = len(pred_sources & gt_sources) / max(1, len(gt_sources)) if gt_sources else 1.0

        return WorkflowEvaluation(
            factuality=factuality,
            recall=recall,
            precision=precision,
            f1=f1,
            condition_resolution=condition_resolution,
            topology_order=topology_order,
            personalization=personalization,
        )

    def get_template(self, template_id: str) -> WorkflowTemplate | None:
        return self._templates.get(template_id)

    def search_templates(self, domain: str = "", query: str = "") -> list[WorkflowTemplate]:
        """Search templates by domain and/or query."""
        results = []
        for tpl in self._templates.values():
            if domain and tpl.domain != domain:
                continue
            if query and query.lower() not in tpl.name.lower():
                continue
            results.append(tpl)
        return results

    def stats(self) -> dict[str, Any]:
        return {
            "templates": len(self._templates),
            "personalized_workflows": len(self._personalized),
            "domains": list(set(t.domain for t in self._templates.values())),
        }
