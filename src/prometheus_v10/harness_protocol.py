"""Prometheus V9PRO Harness Protocol — Processor ABC + VariantPool + Seesaw + Planner.

Integrates HarnessX §3 (Composition), §4.3 (AEGIS pipeline), §4.5 (Variant Isolation).
All types follow V9PRO schema.py conventions (bytes IDs, str Enums, @dataclass).
"""
from __future__ import annotations

import copy
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator

from prometheus_v10.schema import (
    HarnessConfig, HarnessDimension, HarnessEdit, HookPoint, ProcessorEvent,
    LAYER_TO_DIMENSIONS, DIMENSION_TO_LAYERS,
)

logger = logging.getLogger(__name__)


# ── Processor Abstraction (HarnessX §3.2) ──────────────────────────

class Processor(ABC):
    """Typed atomic component. 5 outcomes: pass/transform/split/intercept/interrupt.

    Each processor consumes one event and yields zero or more.
    Attached to a HookPoint; type contract: input type = output type.
    """
    _singleton_group: str = ""
    _order: int = 0   # -1=PRE, 0=NORMAL, 1=POST
    _after: list[str] = []

    @abstractmethod
    def process(self, event: ProcessorEvent) -> Iterator[ProcessorEvent]:
        """Process one event, yield zero or more events."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class PassThroughProcessor(Processor):
    """Default processor: yields input unchanged."""
    _singleton_group = "passthrough"

    def process(self, event: ProcessorEvent) -> Iterator[ProcessorEvent]:
        yield event


class InterceptProcessor(Processor):
    """Processor that blocks event propagation (yields nothing)."""
    _singleton_group = "intercept"

    def __init__(self, target_hook: HookPoint | None = None) -> None:
        self._target_hook = target_hook

    def process(self, event: ProcessorEvent) -> Iterator[ProcessorEvent]:
        if self._target_hook and event.hook != self._target_hook:
            yield event  # pass non-target events
        # else: intercept (yield nothing)


class TransformProcessor(Processor):
    """Processor that modifies event payload."""
    _singleton_group = "transform"

    def __init__(self, transform_fn: Any = None) -> None:
        self._transform_fn = transform_fn
        self._transform_count = 0

    def process(self, event: ProcessorEvent) -> Iterator[ProcessorEvent]:
        if not event.read_only and self._transform_fn:
            new_payload = self._transform_fn(event.payload)
            event.payload = new_payload
            self._transform_count += 1
        yield event

    @property
    def stats(self) -> dict[str, Any]:
        return {"transform_count": self._transform_count}


class ProcessorPipeline:
    """Ordered list of processors at a single hook point.
    Validates hook contracts: read-only hooks cannot yield modified events.
    """
    def __init__(self, hook: HookPoint) -> None:
        self._hook = hook
        self._processors: list[Processor] = []

    def add(self, processor: Processor) -> None:
        """Add processor respecting _order and _singleton_group."""
        # Check singleton exclusion
        if processor._singleton_group:
            for existing in self._processors:
                if existing._singleton_group == processor._singleton_group:
                    idx = self._processors.index(existing)
                    self._processors[idx] = processor
                    logger.debug(f"Replaced singleton {processor._singleton_group} at {self._hook.value}")
                    return
        self._processors.append(processor)
        self._processors.sort(key=lambda p: p._order)

    def remove(self, singleton_group: str) -> bool:
        """Remove processor by singleton group name."""
        for i, p in enumerate(self._processors):
            if p._singleton_group == singleton_group:
                self._processors.pop(i)
                return True
        return False

    def execute(self, event: ProcessorEvent) -> list[ProcessorEvent]:
        """Run all processors sequentially. Each processor receives output of previous."""
        events = [event]
        for proc in self._processors:
            next_events: list[ProcessorEvent] = []
            for evt in events:
                try:
                    for out in proc.process(evt):
                        next_events.append(out)
                except Exception as e:
                    logger.warning(f"Processor {proc.name} raised at {self._hook.value}: {e}")
                    # Interrupt: stop propagation for this event
            events = next_events
            if not events:
                break
        return events

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "hook": self._hook.value,
            "processor_count": len(self._processors),
            "processors": [p.name for p in self._processors],
        }


# ── Seesaw Constraint (HarnessX §4.3 Gate) ─────────────────────────

@dataclass
class SeesawResult:
    """Result of seesaw constraint check."""
    passes: bool = True
    regressed_tasks: list[str] = field(default_factory=list)
    sub_threshold_warnings: list[str] = field(default_factory=list)


class SeesawConstraint:
    """Candidate must not regress any previously solved task. HarnessX §4.3.

    The deterministic gate applies: manifest_completeness → config_normalization →
    seesaw constraint. First failing check halts; passing candidates committed.
    """
    def __init__(self) -> None:
        self._solved_tasks: dict[str, float] = {}  # task_id → best_score
        self._check_count = 0

    def record_solved(self, task_id: str, score: float) -> None:
        """Record a task as solved with its score."""
        if task_id not in self._solved_tasks or score > self._solved_tasks[task_id]:
            self._solved_tasks[task_id] = score

    def check(self, candidate: HarnessEdit, current_scores: dict[str, float] | None = None) -> SeesawResult:
        """Check if candidate edit would regress any previously solved task."""
        self._check_count += 1
        result = SeesawResult()

        # Check manifest completeness
        manifest = candidate.manifest
        if not manifest.get("changed_components"):
            result.passes = False
            result.regressed_tasks = ["manifest_incomplete"]
            return result

        # Check against predicted regressions
        predicted_regress = manifest.get("tasks_expected_regress", [])
        if predicted_regress:
            result.passes = False
            result.regressed_tasks = predicted_regress
            return result

        # Check sub-threshold: tasks with declining but not flipped scores
        if current_scores:
            for task_id, score in current_scores.items():
                if task_id in self._solved_tasks and score < self._solved_tasks[task_id] * 0.9:
                    result.sub_threshold_warnings.append(
                        f"{task_id}: {score:.2f} < {self._solved_tasks[task_id]:.2f} * 0.9"
                    )

        return result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "solved_tasks": len(self._solved_tasks),
            "checks_performed": self._check_count,
        }


# ── Variant Pool (HarnessX §4.5 Ensemble Routing) ──────────────────

@dataclass
class VariantRecord:
    """Track one harness variant's performance."""
    variant_id: str = ""
    config: HarnessConfig = field(default_factory=HarnessConfig)
    task_scores: dict[str, float] = field(default_factory=dict)  # task_id → success_rate
    created_at: float = field(default_factory=time.time)
    edits_applied: int = 0


class VariantPool:
    """Maintain up to K harness variants with ensemble routing. HarnessX §4.5.

    Key insight from HarnessX:
    - Single harness on heterogeneous tasks → treadmill effect
    - Per-variant seesaw constraint → edits don't cross-contaminate
    - Fork on conflicting improvements instead of rejecting

    Experiment: GAIA GPT-5.4 from Δ=0.0% → +13.6%, peak=final (0.0 degradation).
    """
    def __init__(self, max_variants: int = 5) -> None:
        self._max_variants = max_variants
        self._variants: dict[str, VariantRecord] = {}
        self._default_id = "default"

    def initialize(self, config: HarnessConfig) -> str:
        """Create initial variant from config. Returns variant ID."""
        vid = self._default_id
        self._variants[vid] = VariantRecord(
            variant_id=vid, config=copy.deepcopy(config),
        )
        return vid

    def route(self, task_id: str) -> str:
        """Route task to variant with highest estimated success rate on its cluster."""
        if not self._variants:
            return self._default_id

        best_variant = self._default_id
        best_score = -1.0

        for vid, record in self._variants.items():
            # Check exact match first
            if task_id in record.task_scores:
                score = record.task_scores[task_id]
            else:
                # Estimate from average
                scores = list(record.task_scores.values())
                score = sum(scores) / max(1, len(scores)) if scores else 0.0

            if score > best_score:
                best_score = score
                best_variant = vid

        return best_variant

    def propose_edit(self, variant_id: str, edit: HarnessEdit,
                     improved_tasks: list[str], regressed_tasks: list[str],
                     seesaw: SeesawConstraint | None = None) -> dict[str, Any]:
        """Evaluate edit against variant's task cluster only.

        Returns: {"action": "apply"|"fork"|"reject", "target_variant": str, ...}
        """
        if variant_id not in self._variants:
            return {"action": "reject", "reason": "unknown_variant"}

        if not regressed_tasks:
            # Clean improvement — apply to target variant
            self._variants[variant_id].edits_applied += 1
            return {"action": "apply", "target_variant": variant_id,
                    "affected_tasks": improved_tasks}

        # Conflicting improvement — fork new variant
        if len(self._variants) >= self._max_variants:
            # Pool full: retire worst-performing variant
            retired = self._retire_worst()
            if not retired:
                return {"action": "reject", "reason": "pool_full"}

        # Fork: create new variant with the edit
        new_id = f"v_{len(self._variants)}_{int(time.time())}"
        parent = self._variants[variant_id]
        new_config = copy.deepcopy(parent.config)
        self._variants[new_id] = VariantRecord(
            variant_id=new_id, config=new_config,
        )
        logger.info(f"Forked variant {new_id} from {variant_id} for conflicting edit")
        return {"action": "fork", "target_variant": new_id,
                "affected_tasks": improved_tasks + regressed_tasks}

    def update_score(self, variant_id: str, task_id: str, score: float) -> None:
        """Update variant's task score after evaluation."""
        if variant_id in self._variants:
            self._variants[variant_id].task_scores[task_id] = score

    def _retire_worst(self) -> str | None:
        """Retire lowest-performing variant. Returns retired ID or None."""
        if len(self._variants) <= 1:
            return None

        worst_id = None
        worst_score = float("inf")
        for vid, record in self._variants.items():
            if vid == self._default_id:
                continue
            scores = list(record.task_scores.values())
            avg = sum(scores) / max(1, len(scores)) if scores else 0.0
            if avg < worst_score:
                worst_score = avg
                worst_id = vid

        if worst_id and worst_id != self._default_id:
            del self._variants[worst_id]
            logger.info(f"Retired variant {worst_id} (avg_score={worst_score:.2f})")
            return worst_id
        return None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "variant_count": len(self._variants),
            "max_variants": self._max_variants,
            "variants": {
                vid: {
                    "edits_applied": r.edits_applied,
                    "task_count": len(r.task_scores),
                    "avg_score": sum(r.task_scores.values()) / max(1, len(r.task_scores)) if r.task_scores else 0.0,
                }
                for vid, r in self._variants.items()
            },
        }


# ── Adaptation Landscape (HarnessX §4.3 Planner) ──────────────────

@dataclass
class Landscape:
    """Adaptation landscape from Planner. HarnessX §4.3.

    Tracks: failing tasks, attempted edits, implicated components,
    untried edit types, concentration risk, exploration ratio.
    """
    failing_tasks: list[dict[str, Any]] = field(default_factory=list)
    edit_type_distribution: dict[str, int] = field(default_factory=dict)
    untried_dimensions: list[str] = field(default_factory=list)
    concentration_risk: bool = False
    exploration_ratio: float = 0.0
    dimension_coverage: dict[str, float] = field(default_factory=dict)


class AdaptationLandscape:
    """Build adaptation landscape to prevent under-exploration. HarnessX §4.3.

    Without explicit landscape construction, the pipeline converges on
    trace-conditional local repair. Structural changes rarely emerge
    from local hypothesis formation.
    """
    def __init__(self) -> None:
        self._edit_history: list[dict[str, Any]] = []
        self._build_count = 0

    def record_edit(self, edit_type: str, dimensions: list[HarnessDimension],
                    success: bool) -> None:
        """Record an applied edit for landscape tracking."""
        self._edit_history.append({
            "type": edit_type,
            "dimensions": [d.value for d in dimensions],
            "success": success,
            "timestamp": time.time(),
        })

    def build(self, failing_tasks: list[dict[str, Any]] | None = None,
              edit_history: list[HarnessEdit] | None = None) -> Landscape:
        """Construct landscape from current state + edit history."""
        self._build_count += 1
        landscape = Landscape()

        # Failing tasks
        landscape.failing_tasks = failing_tasks or []

        # Edit type distribution
        type_counts: dict[str, int] = {}
        for entry in self._edit_history:
            t = entry["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        landscape.edit_type_distribution = type_counts

        # Concentration risk: >3 consecutive same-type edits
        if len(self._edit_history) >= 4:
            last_types = [e["type"] for e in self._edit_history[-4:]]
            if len(set(last_types)) == 1:
                landscape.concentration_risk = True

        # Dimension coverage
        dim_counts: dict[str, int] = {}
        total_edits = len(self._edit_history)
        for entry in self._edit_history:
            for dim in entry["dimensions"]:
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
        for dim in HarnessDimension:
            landscape.dimension_coverage[dim.value] = dim_counts.get(dim.value, 0) / max(1, total_edits)

        # Untouched dimensions (0 edits)
        landscape.untried_dimensions = [
            dim.value for dim in HarnessDimension
            if dim.value not in dim_counts
        ]

        # Exploration ratio: structural edits (tool, processor, config) vs prompt-only
        structural_types = {"tool", "processor", "config", "architecture", "memory"}
        structural = sum(1 for e in self._edit_history if e["type"] in structural_types)
        landscape.exploration_ratio = structural / max(1, total_edits)

        return landscape

    def suggest_gagarin_edits(self, landscape: Landscape,
                              ratio: float = 0.2) -> list[str]:
        """T4 Gagarin channel: suggest ≥ratio structural/exploratory edits.
        Maps to HarnessX's 'structural changes' vs 'incremental prompt edits'.
        """
        suggestions = []
        if landscape.concentration_risk:
            suggestions.append("CONCENTRATION_RISK: pivot from current edit type")
        if landscape.exploration_ratio < ratio:
            suggestions.append("UNDER_EXPLORATION: need more structural edits")
        for dim in landscape.untried_dimensions:
            suggestions.append(f"UNTRIED_DIMENSION: {dim}")
        return suggestions

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "build_count": self._build_count,
            "edit_history_length": len(self._edit_history),
        }
