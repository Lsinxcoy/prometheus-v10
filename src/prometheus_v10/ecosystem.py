"""Prometheus V10 Ecosystem Engine — Lotka-Volterra skill interaction dynamics.

Inspired by SkillSmith (arXiv:2606.01314) §3.2:
- Interaction matrix: pairwise complementarity (+) and conflict (-) from execution traces
- Lotka-Volterra dynamic utility: du/dt = r*u*(1 - u/K) + Σ α_ij * u_i * u_j
- Instance-level Pareto front: non-dominated states per task instance
- Retrieval scoring: semantic relevance + dynamic utility + interaction compatibility + cost

Key difference from V9PRO:
- V9PRO anti_evo.py: only dedup + zero-gain (no interaction modeling)
- V10 ecosystem.py: full ecological dynamics with complementarity/conflict tracking
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────

@dataclass
class SkillObservation:
    """One execution observation for a skill."""
    skill_name: str = ""
    task_category: str = ""
    score: float = 0.0
    normalized_residual: float = 0.0
    co_activated: list[str] = field(default_factory=list)  # other skills active in same execution
    timestamp: float = field(default_factory=time.time)


@dataclass
class InteractionEntry:
    """Pairwise skill interaction estimate."""
    skill_a: str = ""
    skill_b: str = ""
    synergy: float = 0.0       # positive = complementarity, negative = conflict
    co_occurrence_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class DynamicUtility:
    """Per-skill dynamic utility with Lotka-Volterra update."""
    skill_name: str = ""
    raw_utility: float = 0.0       # observed residual
    dynamic_utility: float = 0.5   # after LV update
    growth_rate: float = 0.1       # r in LV equation
    carrying_capacity: float = 1.0  # K in LV equation
    activation_count: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class ParetoInstance:
    """Instance-level Pareto front entry."""
    task_id: str = ""
    state_fingerprint: str = ""  # identifies a system state (skill+tool config)
    score: float = 0.0
    uniquely_best_count: int = 0


# ── Interaction Matrix ─────────────────────────────────────────────

class InteractionMatrix:
    """Estimates pairwise skill complementarity and conflict from execution traces.

    Following SkillSmith §3.2:
    - Individual utility: EMA of (activated_residual - not_activated_residual)
    - Synergy utility: gain of co-activation over better individual
    - Co-occurrence threshold: pairs below min_co_occurrence get zero prior
    """

    MIN_CO_OCCURRENCE = 3  # minimum co-activations before estimating synergy

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], InteractionEntry] = {}
        self._individual_residuals: dict[str, list[float]] = defaultdict(list)
        self._co_activation_residuals: dict[tuple[str, str], list[float]] = defaultdict(list)

    def record_observation(self, obs: SkillObservation) -> None:
        """Record one execution observation."""
        # Track individual residuals
        self._individual_residuals[obs.skill_name].append(obs.normalized_residual)

        # Track co-activation residuals
        for other in obs.co_activated:
            pair = tuple(sorted([obs.skill_name, other]))
            self._co_activation_residuals[pair].append(obs.normalized_residual)

            # Update co-occurrence count
            entry = self._get_or_create(pair[0], pair[1])
            entry.co_occurrence_count += 1
            entry.last_updated = time.time()

    def compute_synergy(self, skill_a: str, skill_b: str) -> float:
        """Compute synergy between two skills.

        Positive = complementarity (co-activation better than either alone).
        Negative = conflict (co-activation worse).
        Zero = insufficient data or no interaction.
        """
        pair = tuple(sorted([skill_a, skill_b]))
        entry = self._entries.get(pair)

        if entry is None or entry.co_occurrence_count < self.MIN_CO_OCCURRENCE:
            return 0.0  # zero prior for insufficient data

        # Co-activation mean residual
        co_residuals = self._co_activation_residuals.get(pair, [])
        if not co_residuals:
            return 0.0
        co_mean = sum(co_residuals[-20:]) / len(co_residuals[-20:])

        # Individual mean residuals
        a_residuals = self._individual_residuals.get(skill_a, [0.0])
        b_residuals = self._individual_residuals.get(skill_b, [0.0])
        a_mean = sum(a_residuals[-20:]) / len(a_residuals[-20:])
        b_mean = sum(b_residuals[-20:]) / len(b_residuals[-20:])

        # Synergy = co-activation gain over better individual
        better_individual = max(a_mean, b_mean)
        synergy = co_mean - better_individual

        # Update entry
        entry.synergy = synergy
        return synergy

    def get_interactions(self, skill_name: str) -> dict[str, float]:
        """Get all interactions for a skill: {other_skill: synergy_value}."""
        result: dict[str, float] = {}
        for (a, b), entry in self._entries.items():
            if a == skill_name:
                result[b] = entry.synergy
            elif b == skill_name:
                result[a] = entry.synergy
        return result

    def get_conflicts(self, skill_name: str, threshold: float = -0.1) -> list[str]:
        """Get skills that conflict with the given skill."""
        interactions = self.get_interactions(skill_name)
        return [s for s, v in interactions.items() if v < threshold]

    def get_complements(self, skill_name: str, threshold: float = 0.1) -> list[str]:
        """Get skills that complement the given skill."""
        interactions = self.get_interactions(skill_name)
        return [s for s, v in interactions.items() if v > threshold]

    def _get_or_create(self, skill_a: str, skill_b: str) -> InteractionEntry:
        pair = tuple(sorted([skill_a, skill_b]))
        if pair not in self._entries:
            self._entries[pair] = InteractionEntry(skill_a=pair[0], skill_b=pair[1])
        return self._entries[pair]

    def stats(self) -> dict[str, Any]:
        total = len(self._entries)
        complementary = sum(1 for e in self._entries.values() if e.synergy > 0.1)
        conflicting = sum(1 for e in self._entries.values() if e.synergy < -0.1)
        neutral = total - complementary - conflicting
        return {
            "total_pairs": total,
            "complementary": complementary,
            "conflicting": conflicting,
            "neutral": neutral,
        }


# ── Lotka-Volterra Updater ─────────────────────────────────────────

class LotkaVolterraUpdater:
    """Dynamic utility update via Lotka-Volterra competition-mutualism equations.

    Following SkillSmith §3.2 Eq.(6):
        du_i/dt = r_i * u_i * (1 - u_i/K) + Σ_j α_ij * u_i * u_j

    Where:
    - u_i = dynamic utility of skill i
    - r_i = growth rate (from observed residual)
    - K = carrying capacity (shared context budget)
    - α_ij = interaction coefficient (from InteractionMatrix)
    """

    def __init__(self, carrying_capacity: float = 1.0, ema_alpha: float = 0.3) -> None:
        self._K = carrying_capacity
        self._ema_alpha = ema_alpha
        self._utilities: dict[str, DynamicUtility] = {}

    def update(self, skill_name: str, observed_residual: float,
               interactions: dict[str, float], population: dict[str, float]) -> float:
        """Update dynamic utility for one skill using LV equation.

        Args:
            skill_name: The skill to update
            observed_residual: Latest observed utility residual
            interactions: {other_skill: synergy_value} from InteractionMatrix
            population: {skill: current_dynamic_utility} of all skills

        Returns:
            Updated dynamic utility value
        """
        if skill_name not in self._utilities:
            self._utilities[skill_name] = DynamicUtility(
                skill_name=skill_name,
                raw_utility=observed_residual,
                dynamic_utility=max(0.01, min(0.99, 0.5 + observed_residual)),
            )

        util = self._utilities[skill_name]

        # EMA update of raw utility
        util.raw_utility = self._ema_alpha * observed_residual + (1 - self._ema_alpha) * util.raw_utility

        # Growth rate from raw utility
        r = 0.1 * util.raw_utility  # scale factor

        # LV competition term: r * u * (1 - u/K)
        u = util.dynamic_utility
        competition = r * u * (1 - u / self._K)

        # LV interaction term: Σ α_ij * u_i * u_j
        interaction_sum = 0.0
        for other_skill, synergy in interactions.items():
            other_u = population.get(other_skill, 0.5)
            interaction_sum += synergy * u * other_u

        # Euler step (dt=1 for discrete rounds)
        du = competition + interaction_sum
        new_u = u + du

        # Clip to [0.01, 0.99] for numerical stability
        new_u = max(0.01, min(0.99, new_u))

        util.dynamic_utility = new_u
        util.activation_count += 1
        util.last_updated = time.time()

        return new_u

    def get_utility(self, skill_name: str) -> float:
        """Get current dynamic utility for a skill."""
        util = self._utilities.get(skill_name)
        return util.dynamic_utility if util else 0.5

    def get_all_utilities(self) -> dict[str, float]:
        """Get all dynamic utilities."""
        return {name: util.dynamic_utility for name, util in self._utilities.items()}

    def get_declining_skills(self, threshold: float = 0.2) -> list[str]:
        """Get skills with utility below threshold (candidates for retirement)."""
        return [name for name, util in self._utilities.items() if util.dynamic_utility < threshold]

    def stats(self) -> dict[str, Any]:
        if not self._utilities:
            return {"total_skills": 0}
        utilities = [u.dynamic_utility for u in self._utilities.values()]
        return {
            "total_skills": len(self._utilities),
            "mean_utility": sum(utilities) / len(utilities),
            "min_utility": min(utilities),
            "max_utility": max(utilities),
            "declining_count": len(self.get_declining_skills()),
        }


# ── Instance-Level Pareto Front ────────────────────────────────────

class InstanceParetoFront:
    """Instance-level Pareto front management.

    Following SkillSmith §3.2 Eq.(7):
    - Track which system state achieves highest score per task instance
    - A state is non-dominated if best on at least one instance
    - Sample from non-dominated set proportional to uniquely-best count
    """

    def __init__(self) -> None:
        # {task_id: {state_fingerprint: score}}
        self._instance_scores: dict[str, dict[str, float]] = defaultdict(dict)
        # {state_fingerprint: set of task_ids where it's uniquely best}
        self._state_best_instances: dict[str, set[str]] = defaultdict(set)

    def record(self, task_id: str, state_fingerprint: str, score: float) -> None:
        """Record a score for a state on a task instance."""
        old_best = self._get_best_state(task_id)
        self._instance_scores[task_id][state_fingerprint] = score

        # Update uniquely-best tracking
        new_best = self._get_best_state(task_id)
        if new_best != old_best:
            if old_best:
                self._state_best_instances[old_best].discard(task_id)
            if new_best:
                self._state_best_instances[new_best].add(task_id)

    def get_non_dominated(self) -> list[str]:
        """Get all non-dominated state fingerprints."""
        non_dominated = set()
        for task_id, states in self._instance_scores.items():
            if states:
                best_fp = max(states, key=lambda fp: states[fp])
                non_dominated.add(best_fp)
        return list(non_dominated)

    def sample_for_mutation(self) -> str | None:
        """Sample a state for mutation, proportional to uniquely-best count."""
        non_dominated = self.get_non_dominated()
        if not non_dominated:
            return None

        # Weight by uniquely-best instance count (min 1 to avoid zero weight)
        weights = [max(1, len(self._state_best_instances.get(fp, set()))) for fp in non_dominated]
        total = sum(weights)

        import random
        r = random.random() * total
        cumulative = 0.0
        for fp, w in zip(non_dominated, weights):
            cumulative += w
            if r <= cumulative:
                return fp
        return non_dominated[-1]

    def _get_best_state(self, task_id: str) -> str | None:
        states = self._instance_scores.get(task_id, {})
        if not states:
            return None
        return max(states, key=lambda fp: states[fp])

    def stats(self) -> dict[str, Any]:
        return {
            "total_instances": len(self._instance_scores),
            "non_dominated_states": len(self.get_non_dominated()),
            "total_states": len(set(fp for states in self._instance_scores.values() for fp in states)),
        }


# ── Retrieval Scorer ───────────────────────────────────────────────

class EcosystemRetrievalScorer:
    """Score skills for retrieval using dynamic utility + interaction compatibility.

    Following SkillSmith §3.2:
    score(s) = α*semantic + β*dynamic_utility + γ*interaction_compat - δ*cost
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.3,
                 gamma: float = 0.2, delta: float = 0.1) -> None:
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta

    def score(self, skill_name: str, semantic_relevance: float,
              dynamic_utility: float, activated_skills: list[str],
              interactions: dict[str, float], execution_cost: float = 0.0) -> float:
        """Compute retrieval score for a skill.

        Args:
            skill_name: Skill to score
            semantic_relevance: Cosine similarity to query [0,1]
            dynamic_utility: From LotkaVolterraUpdater [0,1]
            activated_skills: Currently activated skill set
            interactions: {other_skill: synergy} from InteractionMatrix
            execution_cost: Normalized cost [0,1]
        """
        # Interaction compatibility: sum of synergies with already-activated skills
        compat = 0.0
        for active in activated_skills:
            compat += interactions.get(active, 0.0)
        # Normalize by number of active skills (avoid bias toward more co-activations)
        if activated_skills:
            compat /= len(activated_skills)

        # Normalize cost to [0,1]
        cost = min(1.0, max(0.0, execution_cost))

        score = (self._alpha * semantic_relevance +
                 self._beta * dynamic_utility +
                 self._gamma * compat -
                 self._delta * cost)

        return max(0.0, min(1.0, score))


# ── Main Ecosystem Engine ──────────────────────────────────────────

class SkillEcosystem:
    """Unified skill ecosystem with interaction tracking, LV dynamics, and Pareto management.

    This is the main entry point for V10's ecological governance.
    Replaces V9PRO's simple anti_evo dedup with full ecosystem modeling.
    """

    def __init__(self, carrying_capacity: float = 1.0,
                 retirement_threshold: float = 0.15,
                 retirement_patience: int = 3) -> None:
        self._interaction_matrix = InteractionMatrix()
        self._lv_updater = LotkaVolterraUpdater(carrying_capacity=carrying_capacity)
        self._pareto = InstanceParetoFront()
        self._retrieval_scorer = EcosystemRetrievalScorer()
        self._retirement_threshold = retirement_threshold
        self._retirement_patience = retirement_patience
        self._low_utility_rounds: dict[str, int] = defaultdict(int)
        self._retired_skills: set[str] = set()

    def record_execution(self, skill_name: str, task_category: str,
                         score: float, category_mean: float,
                         co_activated: list[str]) -> None:
        """Record one execution and update ecosystem.

        Args:
            skill_name: The skill that was executed
            task_category: Category of the task
            score: Raw task score
            category_mean: Historical mean score for this category
            co_activated: Other skills active in same execution
        """
        # Compute normalized residual (remove task-difficulty confounding)
        residual = score - category_mean

        obs = SkillObservation(
            skill_name=skill_name,
            task_category=task_category,
            score=score,
            normalized_residual=residual,
            co_activated=co_activated,
        )

        # Update interaction matrix
        self._interaction_matrix.record_observation(obs)

        # Get interactions for this skill
        interactions = self._interaction_matrix.get_interactions(skill_name)

        # Get current population
        population = self._lv_updater.get_all_utilities()

        # Update LV dynamics
        self._lv_updater.update(skill_name, residual, interactions, population)

        # Check retirement
        utility = self._lv_updater.get_utility(skill_name)
        if utility < self._retirement_threshold:
            self._low_utility_rounds[skill_name] += 1
        else:
            self._low_utility_rounds[skill_name] = 0

    def record_pareto(self, task_id: str, state_fingerprint: str, score: float) -> None:
        """Record a Pareto front entry."""
        self._pareto.record(task_id, state_fingerprint, score)

    def should_retire(self, skill_name: str) -> bool:
        """Check if a skill should be retired (utility below threshold for consecutive rounds)."""
        rounds = self._low_utility_rounds.get(skill_name, 0)
        return rounds >= self._retirement_patience and skill_name not in self._retired_skills

    def retire_skill(self, skill_name: str) -> None:
        """Mark a skill as retired."""
        self._retired_skills.add(skill_name)
        logger.info(f"Skill '{skill_name}' retired (utility below {self._retirement_threshold} for {self._retirement_patience} rounds)")

    def score_for_retrieval(self, skill_name: str, semantic_relevance: float,
                            activated_skills: list[str] | None = None,
                            execution_cost: float = 0.0) -> float:
        """Score a skill for retrieval in a task context."""
        if skill_name in self._retired_skills:
            return 0.0

        dynamic_utility = self._lv_updater.get_utility(skill_name)
        interactions = self._interaction_matrix.get_interactions(skill_name)

        return self._retrieval_scorer.score(
            skill_name=skill_name,
            semantic_relevance=semantic_relevance,
            dynamic_utility=dynamic_utility,
            activated_skills=activated_skills or [],
            interactions=interactions,
            execution_cost=execution_cost,
        )

    def get_mutation_candidates(self, top_k: int = 5) -> list[tuple[str, float]]:
        """Get skills prioritized for mutation (low utility + high conflict)."""
        utilities = self._lv_updater.get_all_utilities()
        candidates = []
        for name, util in utilities.items():
            if name in self._retired_skills:
                continue
            conflicts = self._interaction_matrix.get_conflicts(name)
            # Priority: low utility + many conflicts
            priority = (1.0 - util) + 0.3 * len(conflicts)
            candidates.append((name, priority))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def sample_pareto_state(self) -> str | None:
        """Sample a non-dominated state for mutation."""
        return self._pareto.sample_for_mutation()

    @property
    def interaction_matrix(self) -> InteractionMatrix:
        return self._interaction_matrix

    @property
    def lv_updater(self) -> LotkaVolterraUpdater:
        return self._lv_updater

    @property
    def pareto(self) -> InstanceParetoFront:
        return self._pareto

    def stats(self) -> dict[str, Any]:
        return {
            "interaction": self._interaction_matrix.stats(),
            "lv_dynamics": self._lv_updater.stats(),
            "pareto": self._pareto.stats(),
            "retired_skills": list(self._retired_skills),
            "retirement_candidates": [s for s in self._low_utility_rounds
                                      if self._low_utility_rounds[s] > 0 and s not in self._retired_skills],
        }
