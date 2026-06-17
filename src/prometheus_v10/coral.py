"""Prometheus V9 CORAL Heartbeat — Reflect + Consolidate + Redirect (UCB1)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus_v10.schema import ReflectionNote
from prometheus_v10.layer_infra import UCB1Bandit
from prometheus_v10.utils import jaccard_similarity

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedSkill:
    name: str = ""
    pattern: str = ""
    conditions: list[str] = field(default_factory=list)
    procedure: list[str] = field(default_factory=list)
    source_notes: int = 0
    effectiveness: float = 0.0


class CORALHeartbeat:
    """3-heartbeat mechanism. Redirect uses UCB1 (not random.choice)."""

    REDIRECT_STRATEGIES = ["increase_mutation", "expand_search", "inject_novelty", "lateral_pivot", "rollback"]

    def __init__(self, consolidation_interval: int = 5, stagnation_threshold: int = 10) -> None:
        self._consolidation_interval = consolidation_interval
        self._stagnation_threshold = stagnation_threshold
        self._notes: list[ReflectionNote] = []
        self._skills: list[ConsolidatedSkill] = []
        self._task_count = 0
        self._fitness_history: list[float] = []
        self._last_consolidation = 0
        self._redirect_bandit = UCB1Bandit(self.REDIRECT_STRATEGIES)

    def reflect(self, task: str, outcome: str, insights: list[str] | None = None, mistakes: list[str] | None = None, improvements: list[str] | None = None) -> ReflectionNote:
        """Heartbeat 1: Write reflection note."""
        note = ReflectionNote(task=task, outcome=outcome, insights=insights or [], mistakes=mistakes or [], improvements=improvements or [])
        self._notes.append(note)
        self._task_count += 1
        if self._task_count - self._last_consolidation >= self._consolidation_interval:
            self.consolidate()
        return note

    def consolidate(self) -> list[ConsolidatedSkill]:
        """Heartbeat 2: Merge notes into skills. Uses word overlap (not hardcoded effectiveness)."""
        new_skills = []
        success_notes = [n for n in self._notes if n.outcome == "success"]
        failure_notes = [n for n in self._notes if n.outcome == "failure"]
        # Common insights from successes
        insight_count: dict[str, int] = {}
        for note in success_notes:
            for insight in note.insights:
                key = insight.lower().strip()
                insight_count[key] = insight_count.get(key, 0) + 1
        for key, count in insight_count.items():
            if count >= 2:
                effectiveness = min(1.0, count / len(success_notes)) if success_notes else 0.0
                skill = ConsolidatedSkill(name=f"skill_from_success_{len(self._skills)}", pattern=key, conditions=["repeated_success"], procedure=[key], source_notes=count, effectiveness=effectiveness)
                new_skills.append(skill)
                self._skills.append(skill)
        # Common mistakes from failures
        mistake_count: dict[str, int] = {}
        for note in failure_notes:
            for mistake in note.mistakes:
                key = mistake.lower().strip()
                mistake_count[key] = mistake_count.get(key, 0) + 1
        for key, count in mistake_count.items():
            if count >= 1:
                skill = ConsolidatedSkill(name=f"skill_from_failure_{len(self._skills)}", pattern=f"AVOID: {key}", conditions=["repeated_failure"], procedure=[f"Never {key}"], source_notes=count, effectiveness=0.3)
                new_skills.append(skill)
                self._skills.append(skill)
        self._last_consolidation = self._task_count
        return new_skills

    def redirect(self, fitness_history: list[float]) -> dict[str, Any]:
        """Heartbeat 3: Detect stagnation, select strategy via UCB1 (not random)."""
        self._fitness_history.extend(fitness_history)
        if len(self._fitness_history) < self._stagnation_threshold:
            return {"action": "continue", "reason": "Not enough history"}
        recent = self._fitness_history[-self._stagnation_threshold:]
        improvement = max(recent) - min(recent)
        if improvement < 0.01:
            strategy = self._redirect_bandit.select()  # UCB1, not random.choice
            return {"action": "redirect", "reason": f"Stagnation: {improvement:.4f}", "strategy": strategy}
        return {"action": "continue", "reason": f"Healthy improvement: {improvement:.4f}"}

    def update_redirect_reward(self, strategy: str, reward: float) -> None:
        """Feed back outcome to UCB1 bandit."""
        self._redirect_bandit.update(strategy, reward)

    @property
    def stats(self) -> dict[str, Any]:
        return {"tasks": self._task_count, "notes": len(self._notes), "skills": len(self._skills), "fitness_history_len": len(self._fitness_history)}
