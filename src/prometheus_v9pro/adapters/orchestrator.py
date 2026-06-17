"""Prometheus V9 Orchestrator — Task decomposition, worker assignment, result aggregation.

The missing link between L10 Collaboration (strategy selection) and actual multi-agent execution.
Without this, L10 selects a strategy but has no way to dispatch work.
"""
from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class WorkerState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    DEAD = "dead"


@dataclass
class SubTask:
    """A decomposed subtask."""
    id: str = ""
    parent_id: str = ""
    name: str = ""
    description: str = ""
    required_capabilities: list[str] = field(default_factory=list)
    priority: int = 5
    state: TaskState = TaskState.PENDING
    assigned_worker: str = ""
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    timeout_seconds: float = 300.0
    max_retries: int = 2
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"task_{uuid.uuid4().hex[:8]}"

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMEOUT)


@dataclass
class Worker:
    """A worker agent that can execute subtasks."""
    id: str = ""
    name: str = ""
    role: str = "worker"  # ceo/worker/explorer/judge
    model: str = ""
    model_tier: str = "standard"  # pro/standard/light
    capabilities: list[str] = field(default_factory=list)
    max_concurrent: int = 3
    state: WorkerState = WorkerState.IDLE
    current_tasks: list[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    avg_task_duration: float = 0.0

    @property
    def is_available(self) -> bool:
        return self.state == WorkerState.IDLE and len(self.current_tasks) < self.max_concurrent

    @property
    def success_rate(self) -> float:
        total = self.completed_tasks + self.failed_tasks
        return self.completed_tasks / max(1, total)

    def can_handle(self, required_capabilities: list[str]) -> bool:
        if "*" in self.capabilities:
            return True
        return all(c in self.capabilities for c in required_capabilities)


@dataclass
class TaskResult:
    """Aggregated result from orchestrating a parent task."""
    parent_id: str = ""
    total_subtasks: int = 0
    completed: int = 0
    failed: int = 0
    results: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_duration: float = 0.0
    success_rate: float = 0.0


class Orchestrator:
    """Task orchestrator: decompose → assign → execute → aggregate.

    Strategy modes (aligned with L10 Collaboration):
    - single_agent: Simple delegation, no decomposition
    - parallel_divide: Decompose into parallel subtasks
    - sequential_pipeline: Chain subtasks in sequence
    - debate_vote: Multiple agents debate, vote on result
    """

    def __init__(self, registry=None) -> None:
        self._workers: dict[str, Worker] = {}
        self._tasks: dict[str, SubTask] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._lock = threading.Lock()
        self._registry = registry  # AgentRegistry

    def register_worker(self, worker: Worker) -> None:
        """Register a worker for task assignment."""
        self._workers[worker.id] = worker
        logger.info(f"Registered worker {worker.name} ({worker.role})")

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker."""
        w = self._workers.pop(worker_id, None)
        if w:
            # Cancel assigned tasks
            for tid in w.current_tasks:
                self._cancel_task(tid)

    def decompose_task(self, parent_description: str, strategy: str = "parallel_divide",
                       n_subtasks: int = 3) -> list[SubTask]:
        """Decompose a parent task into subtasks based on strategy."""
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        subtasks = []

        if strategy == "single_agent":
            # No decomposition — one task
            subtasks.append(SubTask(
                parent_id=parent_id,
                name=f"full_task_{parent_id}",
                description=parent_description,
                required_capabilities=["general"],
                priority=5,
            ))
        elif strategy == "parallel_divide":
            # Split into parallel subtasks
            for i in range(n_subtasks):
                subtasks.append(SubTask(
                    parent_id=parent_id,
                    name=f"subtask_{i}",
                    description=f"Part {i+1}/{n_subtasks} of: {parent_description}",
                    required_capabilities=["general"],
                    priority=max(1, 5 - i),
                ))
        elif strategy == "sequential_pipeline":
            # Chain subtasks — each depends on previous
            for i in range(n_subtasks):
                subtasks.append(SubTask(
                    parent_id=parent_id,
                    name=f"step_{i}",
                    description=f"Step {i+1}/{n_subtasks} of pipeline: {parent_description}",
                    required_capabilities=["general"],
                    priority=5,
                    metadata={"depends_on": f"step_{i-1}" if i > 0 else ""},
                ))
        elif strategy == "debate_vote":
            # Multiple perspectives on same task
            for perspective in ["optimist", "critic", "pragmatist"]:
                subtasks.append(SubTask(
                    parent_id=parent_id,
                    name=f"debate_{perspective}",
                    description=f"Analyze from {perspective} perspective: {parent_description}",
                    required_capabilities=["reasoning", "analysis"],
                    priority=5,
                    metadata={"debate_role": perspective},
                ))

        for st in subtasks:
            self._tasks[st.id] = st

        logger.info(f"Decomposed task into {len(subtasks)} subtasks (strategy={strategy})")
        return subtasks

    def assign_task(self, task: SubTask) -> bool:
        """Assign task to best available worker."""
        with self._lock:
            available = [w for w in self._workers.values()
                        if w.is_available and w.can_handle(task.required_capabilities)]
            if not available:
                logger.warning(f"No available worker for task {task.name}")
                return False

            # Pick best worker: highest success_rate among available
            best = max(available, key=lambda w: (w.success_rate, -len(w.current_tasks)))
            task.assigned_worker = best.id
            task.state = TaskState.RUNNING
            task.started_at = time.time()
            best.current_tasks.append(task.id)
            best.state = WorkerState.BUSY
            logger.info(f"Assigned task {task.name} to worker {best.name}")
            return True

    def complete_task(self, task_id: str, result: Any = None, error: str = "") -> None:
        """Mark task as completed or failed."""
        task = self._tasks.get(task_id)
        if not task:
            return
        task.completed_at = time.time()

        if error:
            task.state = TaskState.FAILED
            task.error = error
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                task.state = TaskState.PENDING
                task.assigned_worker = ""
                logger.info(f"Retrying task {task.name} (attempt {task.retry_count})")
                return
        else:
            task.state = TaskState.COMPLETED
            task.result = result

        # Update worker stats
        worker = self._workers.get(task.assigned_worker)
        if worker:
            worker.current_tasks = [t for t in worker.current_tasks if t != task_id]
            if not error:
                worker.completed_tasks += 1
                worker.avg_task_duration = (
                    (worker.avg_task_duration * worker.completed_tasks + task.duration)
                    / max(1, worker.completed_tasks)
                )
            else:
                worker.failed_tasks += 1
            if not worker.current_tasks:
                worker.state = WorkerState.IDLE

    def aggregate_results(self, parent_id: str) -> TaskResult:
        """Aggregate all subtask results for a parent task."""
        subtasks = [t for t in self._tasks.values() if t.parent_id == parent_id]
        completed = [t for t in subtasks if t.state == TaskState.COMPLETED]
        failed = [t for t in subtasks if t.state == TaskState.FAILED]

        result = TaskResult(
            parent_id=parent_id,
            total_subtasks=len(subtasks),
            completed=len(completed),
            failed=len(failed),
            results=[t.result for t in completed],
            errors=[t.error for t in failed],
            total_duration=sum(t.duration for t in completed),
            success_rate=len(completed) / max(1, len(subtasks)),
        )
        self._task_results[parent_id] = result
        return result

    def orchestrate(self, description: str, strategy: str = "parallel_divide",
                    n_subtasks: int = 3, execute_fn: Callable | None = None) -> TaskResult:
        """Full orchestration: decompose → assign → execute → aggregate."""
        subtasks = self.decompose_task(description, strategy, n_subtasks)
        parent_id = subtasks[0].parent_id

        for task in subtasks:
            self.assign_task(task)
            if task.state == TaskState.RUNNING and execute_fn:
                try:
                    result = execute_fn(task.description)
                    self.complete_task(task.id, result=result)
                except Exception as e:
                    self.complete_task(task.id, error=str(e))

        return self.aggregate_results(parent_id)

    @property
    def stats(self) -> dict[str, Any]:
        """Orchestrator statistics."""
        return {
            "registered_workers": len(self._workers),
            "pending_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.PENDING),
            "running_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.RUNNING),
            "completed_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED),
            "failed_tasks": sum(1 for t in self._tasks.values() if t.state == TaskState.FAILED),
        }
