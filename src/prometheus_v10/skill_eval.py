"""Prometheus V10 Skill Evaluator — Automatic task generation + dual scoring.

Inspired by SkillEval (arXiv:2606.17819):
- Auto-generate realistic evaluation tasks from skill content
- Dual scoring: instruction_following + goal_completion
- Skill diagnosis: identify specific weak spots
- Model gap closing: cheap model + skill ≈ frontier model

Key difference from V9PRO:
- V9PRO chain_validator.py: validates code syntax/safety/semantics
- V10 skill_eval.py: evaluates whether a skill actually changes agent behavior
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalTask:
    """An evaluation task generated for a skill."""
    task_id: str = ""
    skill_name: str = ""
    description: str = ""         # realistic user request
    environment_requirements: list[str] = field(default_factory=list)
    input_artifacts: dict[str, str] = field(default_factory=dict)
    rubric_instruction: list[str] = field(default_factory=list)   # hidden instruction-following criteria
    rubric_goal: list[str] = field(default_factory=list)          # hidden goal-completion criteria


@dataclass
class EvalResult:
    """Evaluation result for one task."""
    task_id: str = ""
    skill_name: str = ""
    instruction_following: float = 0.0  # [0,1] how closely skill instructions were followed
    goal_completion: float = 0.0        # [0,1] whether the goal was achieved
    marginal_value: float = 0.0        # improvement over no-skill baseline
    weak_spots: list[str] = field(default_factory=list)
    strong_spots: list[str] = field(default_factory=list)


@dataclass
class SkillUtility:
    """Aggregate utility score for a skill."""
    skill_name: str = ""
    avg_instruction_following: float = 0.0
    avg_goal_completion: float = 0.0
    marginal_value: float = 0.0    # improvement over no-skill baseline
    diagnosis: dict[str, str] = field(default_factory=dict)  # category → weak/strong/neutral


class SkillEvaluator:
    """Evaluate whether a skill actually improves agent behavior.

    Following SkillEval §2:
    1. Analyze skill + user intent
    2. Generate realistic evaluation tasks
    3. Run with-skill and without-skill
    4. Score on instruction-following and goal-completion
    5. Diagnose weak spots
    """

    def __init__(self) -> None:
        self._tasks: dict[str, list[EvalTask]] = {}  # skill_name → tasks
        self._results: dict[str, list[EvalResult]] = {}
        self._task_id_counter: int = 0

    def generate_tasks(self, skill_name: str, skill_content: str,
                       n_tasks: int = 3) -> list[EvalTask]:
        """Generate evaluation tasks from skill content.

        In production, this would use an LLM to generate realistic tasks.
        Here we use a rule-based approach for testability.
        """
        tasks: list[EvalTask] = []

        # Extract key aspects from skill content
        aspects = self._extract_aspects(skill_content)

        for i in range(n_tasks):
            self._task_id_counter += 1
            aspect = aspects[i % len(aspects)] if aspects else "general"

            task = EvalTask(
                task_id=f"task_{self._task_id_counter}",
                skill_name=skill_name,
                description=f"Test task for {skill_name}: verify {aspect} functionality",
                rubric_instruction=[
                    f"Follow {aspect} instructions from the skill",
                    f"Use the specified workflow for {aspect}",
                ],
                rubric_goal=[
                    f"Successfully complete {aspect} operation",
                    f"Produce correct output for {aspect}",
                ],
            )
            tasks.append(task)

        self._tasks[skill_name] = tasks
        return tasks

    def evaluate(self, skill_name: str, with_skill_output: dict[str, Any],
                 without_skill_output: dict[str, Any] | None = None) -> EvalResult:
        """Evaluate a skill by comparing with-skill vs without-skill outputs.

        Args:
            skill_name: The skill being evaluated
            with_skill_output: Output when skill is active
            without_skill_output: Output when skill is not active (for marginal value)
        """
        tasks = self._tasks.get(skill_name, [])
        if not tasks:
            tasks = self.generate_tasks(skill_name, "")

        task = tasks[0]  # evaluate against first task

        # Score instruction following
        instruction_score = self._score_instruction_following(with_skill_output, task)

        # Score goal completion
        goal_score = self._score_goal_completion(with_skill_output, task)

        # Compute marginal value (if baseline available)
        marginal = 0.0
        if without_skill_output is not None:
            baseline_goal = self._score_goal_completion(without_skill_output, task)
            marginal = goal_score - baseline_goal

        # Diagnose weak/strong spots
        weak, strong = self._diagnose(instruction_score, goal_score, task)

        result = EvalResult(
            task_id=task.task_id,
            skill_name=skill_name,
            instruction_following=instruction_score,
            goal_completion=goal_score,
            marginal_value=marginal,
            weak_spots=weak,
            strong_spots=strong,
        )

        if skill_name not in self._results:
            self._results[skill_name] = []
        self._results[skill_name].append(result)
        return result

    def compute_utility(self, skill_name: str) -> SkillUtility:
        """Compute aggregate utility for a skill across all evaluations."""
        results = self._results.get(skill_name, [])
        if not results:
            return SkillUtility(skill_name=skill_name)

        avg_if = sum(r.instruction_following for r in results) / len(results)
        avg_gc = sum(r.goal_completion for r in results) / len(results)
        marginal = sum(r.marginal_value for r in results) / len(results) if results else 0.0

        # Build diagnosis
        diagnosis: dict[str, str] = {}
        for r in results:
            for ws in r.weak_spots:
                diagnosis[ws] = "weak"
            for ss in r.strong_spots:
                diagnosis[ss] = "strong"

        return SkillUtility(
            skill_name=skill_name,
            avg_instruction_following=avg_if,
            avg_goal_completion=avg_gc,
            marginal_value=marginal,
            diagnosis=diagnosis,
        )

    def _extract_aspects(self, content: str) -> list[str]:
        """Extract key aspects from skill content for task generation."""
        aspects: list[str] = []
        # Simple rule-based extraction
        keywords = ["workflow", "validation", "search", "analysis", "generation",
                     "safety", "optimization", "monitoring", "error_handling"]
        content_lower = content.lower()
        for kw in keywords:
            if kw in content_lower:
                aspects.append(kw)
        return aspects or ["general"]

    def _score_instruction_following(self, output: dict[str, Any], task: EvalTask) -> float:
        """Score how closely the output follows skill instructions."""
        score = 0.5  # baseline
        # Check if output mentions key instruction patterns
        output_str = str(output).lower()
        for rubric in task.rubric_instruction:
            rubric_lower = rubric.lower()
            # Simple keyword overlap
            overlap = sum(1 for w in rubric_lower.split() if w in output_str)
            total = len(rubric_lower.split())
            if total > 0:
                score += 0.1 * (overlap / total)
        return min(1.0, score)

    def _score_goal_completion(self, output: dict[str, Any], task: EvalTask) -> float:
        """Score whether the goal was achieved."""
        score = 0.5  # baseline
        output_str = str(output).lower()
        # Check for success indicators
        success_indicators = ["success", "completed", "done", "result", "output"]
        for indicator in success_indicators:
            if indicator in output_str:
                score += 0.1
        return min(1.0, score)

    def _diagnose(self, instruction_score: float, goal_score: float,
                  task: EvalTask) -> tuple[list[str], list[str]]:
        """Diagnose weak and strong spots."""
        weak: list[str] = []
        strong: list[str] = []

        if instruction_score < 0.4:
            weak.append("instruction_following")
        elif instruction_score > 0.7:
            strong.append("instruction_following")

        if goal_score < 0.4:
            weak.append("goal_completion")
        elif goal_score > 0.7:
            strong.append("goal_completion")

        return weak, strong

    def stats(self) -> dict[str, Any]:
        return {
            "skills_evaluated": len(self._results),
            "total_tasks": sum(len(t) for t in self._tasks.values()),
            "total_results": sum(len(r) for r in self._results.values()),
        }
