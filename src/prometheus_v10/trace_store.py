"""Prometheus V9PRO Trace Store — Execution trace recording + Digester compression.

Integrates HarnessX §4.3 (Digester) + D8 (Observability).
Trace records capture every model turn, tool call, and tool result.
The Digester compresses raw traces into structured per-task summaries.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.schema import HookPoint

logger = logging.getLogger(__name__)


# ── Trace Records ──────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Record of one step within a trace."""
    hook: HookPoint = HookPoint.STEP_START
    processor_name: str = ""
    input_event: dict[str, Any] = field(default_factory=dict)
    output_events: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class TraceRecord:
    """Complete execution trace for one task run. HarnessX D8.
    Captures every model turn, tool call, and tool result.
    """
    task_id: str = ""
    harness_version: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    outcome: str = ""           # pass | fail | partial
    failure_category: str = ""  # e.g. "tool_error", "model_error", "format_mismatch"
    implicated_components: list[str] = field(default_factory=list)
    evidence_excerpts: list[str] = field(default_factory=list)
    verifier_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def add_step(self, step: StepRecord) -> None:
        self.steps.append(step)

    def mark_failure(self, category: str, component: str, evidence: str) -> None:
        self.outcome = "fail"
        self.failure_category = category
        if component not in self.implicated_components:
            self.implicated_components.append(component)
        self.evidence_excerpts.append(evidence[:200])

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def failed_steps(self) -> list[StepRecord]:
        return [s for s in self.steps if not s.success]


@dataclass
class TaskSummary:
    """Compressed per-task summary from Digester. HarnessX §4.3.
    10M raw trace tokens → 10K structured summaries.
    """
    task_id: str = ""
    outcome: str = ""
    failure_category: str = ""
    implicated_components: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    cross_iteration_history: list[dict[str, Any]] = field(default_factory=list)
    verifier_score: float = 0.0
    harness_version: str = ""
    step_count: int = 0
    failed_step_count: int = 0


# ── Trace Store ────────────────────────────────────────────────────

class TraceStore:
    """Structured trace storage with FIFO eviction. HarnessX D8.

    Accumulates traces across iterations. Each trace is annotated with
    the harness version that produced it, enabling cross-version analysis.
    """
    def __init__(self, capacity: int = 1000) -> None:
        self._capacity = capacity
        self._records: deque[TraceRecord] = deque(maxlen=capacity)
        self._append_count = 0

    def append(self, record: TraceRecord) -> None:
        """Append a trace record. FIFO eviction when capacity exceeded."""
        self._records.append(record)
        self._append_count += 1

    def query(self, task_id: str | None = None,
              harness_version: str | None = None,
              outcome: str | None = None,
              limit: int = 100) -> list[TraceRecord]:
        """Query traces by filters."""
        results = []
        for record in self._records:
            if task_id and record.task_id != task_id:
                continue
            if harness_version and record.harness_version != harness_version:
                continue
            if outcome and record.outcome != outcome:
                continue
            results.append(record)
            if len(results) >= limit:
                break
        return results

    def get_task_history(self, task_id: str) -> list[TraceRecord]:
        """Get all traces for a task across harness versions."""
        return [r for r in self._records if r.task_id == task_id]

    def get_failing_tasks(self) -> dict[str, list[TraceRecord]]:
        """Get all tasks with failures, grouped by task_id."""
        failing: dict[str, list[TraceRecord]] = {}
        for record in self._records:
            if record.outcome == "fail":
                failing.setdefault(record.task_id, []).append(record)
        return failing

    def get_implicated_components(self) -> dict[str, int]:
        """Count how often each component is implicated in failures."""
        counts: dict[str, int] = {}
        for record in self._records:
            if record.outcome == "fail":
                for comp in record.implicated_components:
                    counts[comp] = counts.get(comp, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @property
    def stats(self) -> dict[str, Any]:
        outcomes = {"pass": 0, "fail": 0, "partial": 0}
        for r in self._records:
            outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1
        return {
            "capacity": self._capacity,
            "current_size": len(self._records),
            "total_appended": self._append_count,
            "outcomes": outcomes,
        }


# ── Digester (HarnessX §4.3) ──────────────────────────────────────

class Digester:
    """Compress raw traces into structured per-task summaries.

    HarnessX: a single GAIA iteration (103 tasks, pass@2) generates ~10M tokens
    of raw traces. Passing this directly to downstream stages exceeds context limits.
    The Digester compresses each task's traces into a structured summary:
    binary outcome, failure category, implicated components, evidence excerpts.

    Also provides cross-iteration continuity: each task's summary links to
    its history of prior outcomes and shipped edits.
    """
    def __init__(self) -> None:
        self._digest_count = 0
        self._summary_cache: dict[str, list[TaskSummary]] = {}

    def digest(self, traces: list[TraceRecord],
               store: TraceStore | None = None) -> list[TaskSummary]:
        """Compress traces into per-task summaries with cross-iteration history."""
        self._digest_count += 1
        summaries: list[TaskSummary] = []

        # Group traces by task_id
        by_task: dict[str, list[TraceRecord]] = {}
        for trace in traces:
            by_task.setdefault(trace.task_id, []).append(trace)

        for task_id, task_traces in by_task.items():
            # Use most recent trace as primary
            primary = task_traces[-1]

            # Build cross-iteration history from store
            history = []
            if store:
                past = store.get_task_history(task_id)
                for past_trace in past:
                    if past_trace.timestamp < primary.timestamp:
                        history.append({
                            "outcome": past_trace.outcome,
                            "version": past_trace.harness_version,
                            "score": past_trace.verifier_score,
                        })

            summary = TaskSummary(
                task_id=task_id,
                outcome=primary.outcome,
                failure_category=primary.failure_category,
                implicated_components=list(primary.implicated_components),
                evidence=primary.evidence_excerpts[:5],
                cross_iteration_history=history[-10:],  # last 10 iterations
                verifier_score=primary.verifier_score,
                harness_version=primary.harness_version,
                step_count=primary.step_count,
                failed_step_count=len(primary.failed_steps),
            )
            summaries.append(summary)

            # Cache for future lookups
            self._summary_cache.setdefault(task_id, []).append(summary)

        return summaries

    def get_task_summary(self, task_id: str) -> TaskSummary | None:
        """Get most recent summary for a task."""
        summaries = self._summary_cache.get(task_id, [])
        return summaries[-1] if summaries else None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "digest_count": self._digest_count,
            "cached_tasks": len(self._summary_cache),
            "total_summaries": sum(len(v) for v in self._summary_cache.values()),
        }
