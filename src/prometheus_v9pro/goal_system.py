"""Prometheus V9 Goal System — Evolution goal lifecycle: PENDING→ACTIVE→COMPLETED/FAILED.

Without goals, evolution is directionless drift. Goals provide the fitness targets
that the 12-layer engine optimizes toward.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class GoalState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Goal:
    """Evolution goal with lifecycle tracking."""
    id: str = ""
    name: str = ""
    description: str = ""
    state: GoalState = GoalState.PENDING
    priority: int = 5  # 1=highest
    parent_id: str = ""
    sub_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    fitness_target: float = 0.8
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: float = 0.0
    metadata: dict = field(default_factory=dict)

    def activate(self) -> None:
        self.state = GoalState.ACTIVE
        self.updated_at = time.time()

    def complete(self, progress: float = 1.0) -> None:
        self.state = GoalState.COMPLETED
        self.progress = progress
        self.updated_at = time.time()

    def fail(self, reason: str = "") -> None:
        self.state = GoalState.FAILED
        self.metadata["failure_reason"] = reason
        self.updated_at = time.time()

    def cancel(self) -> None:
        self.state = GoalState.CANCELLED
        self.updated_at = time.time()

    def update_progress(self, progress: float) -> None:
        self.progress = min(1.0, max(0.0, progress))
        self.updated_at = time.time()

    def is_overdue(self) -> bool:
        return self.deadline > 0 and time.time() > self.deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "state": self.state.value,
            "priority": self.priority, "progress": self.progress,
            "fitness_target": self.fitness_target,
        }


class GoalSystem:
    """Goal lifecycle management with priority queue and progress tracking."""

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}
        self._active_goal_id: str | None = None

    def create_goal(self, name: str, description: str = "", priority: int = 5,
                    fitness_target: float = 0.8, deadline: float = 0.0,
                    constraints: list[str] | None = None) -> Goal:
        """Create a new goal."""
        goal_id = f"goal_{len(self._goals)}_{int(time.time())}"
        goal = Goal(
            id=goal_id, name=name, description=description,
            priority=priority, fitness_target=fitness_target,
            deadline=deadline, constraints=constraints or [],
        )
        self._goals[goal_id] = goal
        logger.info(f"Created goal: {name} (priority={priority})")
        return goal

    def activate_goal(self, goal_id: str) -> bool:
        """Activate a goal (deactivate current if any)."""
        goal = self._goals.get(goal_id)
        if not goal or goal.state != GoalState.PENDING:
            return False
        # Complete or cancel current active
        if self._active_goal_id:
            current = self._goals.get(self._active_goal_id)
            if current and current.state == GoalState.ACTIVE:
                current.cancel()
        goal.activate()
        self._active_goal_id = goal_id
        return True

    def get_active_goal(self) -> Goal | None:
        if self._active_goal_id:
            return self._goals.get(self._active_goal_id)
        return None

    def update_progress(self, goal_id: str, progress: float) -> None:
        goal = self._goals.get(goal_id)
        if goal and goal.state == GoalState.ACTIVE:
            goal.update_progress(progress)
            if progress >= 1.0:
                goal.complete()
                self._active_goal_id = None

    def fail_goal(self, goal_id: str, reason: str = "") -> None:
        goal = self._goals.get(goal_id)
        if goal:
            goal.fail(reason)
            if self._active_goal_id == goal_id:
                self._active_goal_id = None

    def auto_activate(self) -> Goal | None:
        """Auto-activate highest priority pending goal."""
        pending = [g for g in self._goals.values()
                   if g.state == GoalState.PENDING and not g.is_overdue()]
        if not pending:
            return None
        best = min(pending, key=lambda g: g.priority)
        self.activate_goal(best.id)
        return best

    @property
    def stats(self) -> dict[str, Any]:
        by_state = {}
        for state in GoalState:
            by_state[state.value] = sum(1 for g in self._goals.values() if g.state == state)
        return {
            "total_goals": len(self._goals),
            "by_state": by_state,
            "active_goal": self._active_goal_id,
        }
