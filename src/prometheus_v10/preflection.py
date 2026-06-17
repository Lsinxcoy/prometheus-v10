"""Prometheus V10 Preflection Engine — Proactive failure avoidance before action.

Inspired by EvolveNav (arXiv:2606.18235) §3.4-3.5:
- Self-evolving rule bank: extract actionable rules from trajectories
- UCB retrieval: balance semantic relevance and historical success rate
- Semantic credit assignment: marginal semantic similarity gain per step
- Preflection: evaluate candidate actions BEFORE execution, predict failures

Key difference from V9PRO:
- V9PRO initiative.py: 7-layer governance with post-hoc safety check
- V10 preflection.py: adds pre-action consequence forecasting + UCB rule retrieval
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
class NavigationRule:
    """An actionable rule extracted from past trajectory."""
    rule_id: str = ""
    description: str = ""          # human-readable rule text
    context_keywords: list[str] = field(default_factory=list)  # when this rule applies
    action_type: str = ""          # e.g., "avoid", "prefer", "check_before"
    target: str = ""               # what the rule applies to
    success_rate: float = 0.5      # historical success rate
    support_score: float = 0.0     # initial support from credit assignment
    momentum_score: float = 0.5    # EMA-smoothed utility
    retrieval_count: int = 0       # how many times this rule was retrieved
    episode_count: int = 0         # how many episodes it was applied in
    source_episode: str = ""       # which episode generated this rule
    created_at: float = field(default_factory=time.time)


@dataclass
class PreflectionResult:
    """Result of preflection evaluation for one candidate action."""
    action: str = ""
    predicted_success: float = 0.5
    predicted_risks: list[str] = field(default_factory=list)
    applicable_rules: list[NavigationRule] = field(default_factory=list)
    recommendation: str = ""       # "proceed", "avoid", "caution", "alternative"
    confidence: float = 0.5
    reasoning: str = ""


@dataclass
class CreditAssignment:
    """Semantic-driven credit assignment for one trajectory step."""
    step_index: int = 0
    semantic_similarity: float = 0.0  # composite similarity to target
    marginal_gain: float = 0.0        # gain relative to previous step
    credit_weight: float = 0.0        # normalized weight for rule support


# ── Rule Bank ──────────────────────────────────────────────────────

class RuleBank:
    """Self-evolving agentic rule memory with UCB-based retrieval.

    Following EvolveNav §3.4:
    - Extract rules from post-episode trajectory analysis
    - UCB score = momentum_score + c * sqrt(ln(total_episodes) / retrieval_count)
    - New rules initialized with UCB=∞ (guaranteed initial validation)
    """

    EXPLORATION_COEFFICIENT = 1.5  # c in UCB formula
    MOMENTUM_DECAY = 0.7           # EMA decay for momentum score

    def __init__(self, max_rules: int = 200) -> None:
        self._rules: dict[str, NavigationRule] = {}
        self._total_episodes: int = 0
        self._max_rules = max_rules
        self._rule_id_counter: int = 0

    def add_rule(self, description: str, context_keywords: list[str],
                 action_type: str, target: str, support_score: float,
                 source_episode: str = "") -> NavigationRule:
        """Add a new rule extracted from trajectory analysis."""
        rule_id = f"rule_{self._rule_id_counter}"
        self._rule_id_counter += 1

        rule = NavigationRule(
            rule_id=rule_id,
            description=description,
            context_keywords=context_keywords,
            action_type=action_type,
            target=target,
            support_score=support_score,
            momentum_score=support_score,
            source_episode=source_episode,
        )
        self._rules[rule_id] = rule

        # Evict lowest-UCB rule if over capacity
        if len(self._rules) > self._max_rules:
            self._evict_lowest_ucb()

        logger.info(f"Added rule '{rule_id}': {description[:50]}")
        return rule

    def get_ucb_scores(self) -> dict[str, float]:
        """Compute UCB scores for all rules."""
        scores: dict[str, float] = {}
        for rule_id, rule in self._rules.items():
            if rule.retrieval_count == 0:
                # New rules get ∞ score → guaranteed initial validation
                scores[rule_id] = float('inf')
            else:
                # UCB formula: momentum + c * sqrt(ln(N) / n)
                explore_term = self.EXPLORATION_COEFFICIENT * math.sqrt(
                    math.log(self._total_episodes + 1) / rule.retrieval_count
                )
                scores[rule_id] = rule.momentum_score + explore_term
        return scores

    def retrieve_top_k(self, query_keywords: list[str], k: int = 5) -> list[NavigationRule]:
        """Retrieve top-k rules for a query using UCB scoring + semantic relevance."""
        if not self._rules:
            return []

        ucb_scores = self.get_ucb_scores()
        query_set = set(query_keywords)

        # Combine UCB score with semantic relevance
        combined: list[tuple[float, NavigationRule]] = []
        for rule_id, rule in self._rules.items():
            # Semantic relevance (Jaccard similarity)
            if rule.context_keywords and query_set:
                from prometheus_v10.utils import jaccard_similarity
                sem_rel = jaccard_similarity(query_set, set(rule.context_keywords))
            else:
                sem_rel = 0.0

            # Combined score: 0.4*semantic + 0.6*UCB
            ucb = ucb_scores.get(rule_id, 0.0)
            if ucb == float('inf'):
                combined_score = 1.0  # cap for new rules
            else:
                combined_score = 0.4 * sem_rel + 0.6 * min(1.0, ucb / 3.0)

            combined.append((combined_score, rule))

        combined.sort(key=lambda x: x[0], reverse=True)

        # Update retrieval count for selected rules
        result = [rule for _, rule in combined[:k]]
        for rule in result:
            rule.retrieval_count += 1

        return result

    def update_momentum(self, rule_id: str, new_score: float) -> None:
        """Update momentum score with EMA smoothing."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.momentum_score = self.MOMENTUM_DECAY * rule.momentum_score + \
                                   (1 - self.MOMENTUM_DECAY) * new_score
            rule.episode_count += 1

    def increment_episode(self) -> None:
        """Increment total episode count."""
        self._total_episodes += 1

    def _evict_lowest_ucb(self) -> None:
        """Remove the rule with lowest UCB score."""
        ucb_scores = self.get_ucb_scores()
        if not ucb_scores:
            return
        # Find lowest (excluding ∞ for new rules)
        finite_scores = {k: v for k, v in ucb_scores.items() if v != float('inf')}
        if finite_scores:
            lowest_id = min(finite_scores, key=finite_scores.get)
            del self._rules[lowest_id]
            logger.info(f"Evicted rule '{lowest_id}' (lowest UCB={finite_scores[lowest_id]:.3f})")

    def stats(self) -> dict[str, Any]:
        return {
            "total_rules": len(self._rules),
            "total_episodes": self._total_episodes,
            "avg_momentum": sum(r.momentum_score for r in self._rules.values()) / max(1, len(self._rules)),
        }


# ── Semantic Credit Assignment ─────────────────────────────────────

class SemanticCreditAssigner:
    """Assign credit to trajectory steps based on marginal semantic similarity gain.

    Following EvolveNav §3.4.1:
    - Composite similarity: S_i = α*sim(room, target) + β*sim(objects, target)
    - Initial credit: credit_i = S_i - S_{i-1}  (marginal gain)
    - Clip extreme values, normalize via softmax
    - Rule support score = weighted sum of credits from associated steps
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 clip_min: float = -2.0, clip_max: float = 2.0) -> None:
        self._alpha = alpha
        self._beta = beta
        self._clip_min = clip_min
        self._clip_max = clip_max

    def compute_credits(self, trajectory: list[dict[str, Any]],
                        target_keywords: list[str]) -> list[CreditAssignment]:
        """Compute credit assignments for all steps in a trajectory.

        Args:
            trajectory: list of step dicts with 'room_type', 'visible_objects', etc.
            target_keywords: keywords describing the target
        """
        target_set = set(target_keywords)
        assignments: list[CreditAssignment] = []

        prev_similarity = 0.0
        for i, step in enumerate(trajectory):
            # Compute composite similarity
            room_sim = self._keyword_similarity(step.get("room_type", ""), target_set)
            objects_sim = self._keyword_similarity(step.get("visible_objects", ""), target_set)
            composite = self._alpha * room_sim + self._beta * objects_sim

            # Marginal gain
            marginal = composite - prev_similarity

            # Clip
            marginal = max(self._clip_min, min(self._clip_max, marginal))

            assignments.append(CreditAssignment(
                step_index=i,
                semantic_similarity=composite,
                marginal_gain=marginal,
                credit_weight=0.0,  # will be set after normalization
            ))

            prev_similarity = composite

        # Normalize via softmax on marginal gains
        if assignments:
            gains = [a.marginal_gain for a in assignments]
            weights = self._softmax(gains)
            for i, assignment in enumerate(assignments):
                assignment.credit_weight = weights[i]

        return assignments

    def compute_rule_support(self, rule_step_indices: list[int],
                             credits: list[CreditAssignment]) -> float:
        """Compute support score for a rule from its associated steps.

        Following EvolveNav Eq.(3): support = Σ(credit_weight * marginal_gain) for key steps.
        """
        support = 0.0
        for idx in rule_step_indices:
            if 0 <= idx < len(credits):
                support += credits[idx].credit_weight * credits[idx].marginal_gain
        return max(0.0, support)  # support should be non-negative

    def _keyword_similarity(self, text_or_list: str | list[str],
                            target_set: set[str]) -> float:
        """Compute similarity between step observations and target keywords."""
        if isinstance(text_or_list, list):
            step_set = set(text_or_list)
        elif isinstance(text_or_list, str):
            step_set = set(text_or_list.lower().split())
        else:
            return 0.0

        if not step_set or not target_set:
            return 0.0

        intersection = step_set & target_set
        union = step_set | target_set
        return len(intersection) / len(union) if union else 0.0

    def _softmax(self, values: list[float], temperature: float = 1.0) -> list[float]:
        """Softmax normalization."""
        if not values:
            return []
        max_val = max(values)
        exps = [math.exp((v - max_val) / temperature) for v in values]
        total = sum(exps)
        return [e / total for e in exps]


# ── Preflection Engine ──────────────────────────────────────────────

class PreflectionEngine:
    """Evaluate candidate actions BEFORE execution by predicting consequences.

    Following EvolveNav §3.5:
    - Retrieve top-k rules from RuleBank
    - For each candidate action, evaluate predicted risks
    - Mandatory risk assessment filters out deceptive paths
    - Shift from post-hoc correction to proactive avoidance
    """

    def __init__(self, rule_bank: RuleBank | None = None,
                 risk_threshold: float = 0.3) -> None:
        self._rule_bank = rule_bank or RuleBank()
        self._risk_threshold = risk_threshold
        self._preflection_log: list[PreflectionResult] = []

    def prelect(self, action: str, context_keywords: list[str],
                candidate_details: list[dict[str, Any]] | None = None) -> PreflectionResult:
        """Evaluate an action before execution.

        Args:
            action: The proposed action type (e.g., "read", "search", "navigate")
            context_keywords: Keywords describing current context
            candidate_details: Optional details about candidate targets

        Returns:
            PreflectionResult with predicted success, risks, and recommendation
        """
        # Retrieve relevant rules
        rules = self._rule_bank.retrieve_top_k(context_keywords, k=5)

        # Evaluate risks based on rules
        predicted_risks: list[str] = []
        predicted_success = 0.5
        reasoning_parts: list[str] = []

        for rule in rules:
            if rule.action_type == "avoid" and action in rule.description.lower():
                predicted_risks.append(f"Rule '{rule.rule_id}' suggests avoiding: {rule.description}")
                predicted_success -= 0.15 * rule.momentum_score
                reasoning_parts.append(f"AVOID rule (momentum={rule.momentum_score:.2f})")

            elif rule.action_type == "prefer" and action in rule.description.lower():
                predicted_success += 0.1 * rule.momentum_score
                reasoning_parts.append(f"PREFER rule (momentum={rule.momentum_score:.2f})")

            elif rule.action_type == "check_before":
                predicted_risks.append(f"Rule '{rule.rule_id}' requires check: {rule.description}")
                reasoning_parts.append(f"CHECK_BEFORE rule")

        # Clip predicted success
        predicted_success = max(0.0, min(1.0, predicted_success))

        # Determine recommendation
        if predicted_success >= 0.7 and not predicted_risks:
            recommendation = "proceed"
        elif predicted_success < self._risk_threshold:
            recommendation = "avoid"
        elif predicted_risks:
            recommendation = "caution"
        else:
            recommendation = "proceed"

        result = PreflectionResult(
            action=action,
            predicted_success=predicted_success,
            predicted_risks=predicted_risks,
            applicable_rules=rules,
            recommendation=recommendation,
            confidence=min(1.0, len(rules) * 0.2 + 0.3),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No applicable rules",
        )

        self._preflection_log.append(result)
        return result

    def extract_rules_from_trajectory(self, trajectory: list[dict[str, Any]],
                                       target_keywords: list[str],
                                       episode_id: str = "") -> list[NavigationRule]:
        """Extract new rules from a completed trajectory.

        Following EvolveNav §3.4:
        - Review trajectory to identify pivotal decision steps
        - Synthesize reusable rules from decision contexts
        - Assign initial support score via credit assignment
        """
        # Compute credit assignments
        assigner = SemanticCreditAssigner()
        credits = assigner.compute_credits(trajectory, target_keywords)

        # Find pivotal steps (high positive or negative credit)
        pivotal_steps = []
        for credit in credits:
            if abs(credit.marginal_gain) > 0.1 or credit.credit_weight > 0.15:
                pivotal_steps.append(credit)

        # Generate rules from pivotal steps
        new_rules: list[NavigationRule] = []
        for step_credit in pivotal_steps:
            step_data = trajectory[step_credit.step_index] if step_credit.step_index < len(trajectory) else {}
            step_type = step_data.get("action_type", "unknown")

            # Determine rule type based on credit sign
            if step_credit.marginal_gain > 0:
                rule_action = "prefer"
                desc = f"Prefer {step_type} in contexts like {step_data.get('room_type', 'unknown')}"
            else:
                rule_action = "avoid"
                desc = f"Avoid {step_type} in contexts like {step_data.get('room_type', 'unknown')}"

            # Extract context keywords
            keywords = []
            if step_data.get("room_type"):
                keywords.append(step_data["room_type"])
            if step_data.get("visible_objects"):
                keywords.extend(step_data.get("visible_objects", [])[:3])

            support = step_credit.credit_weight * max(0.1, step_credit.marginal_gain)

            rule = self._rule_bank.add_rule(
                description=desc,
                context_keywords=keywords,
                action_type=rule_action,
                target=step_type,
                support_score=support,
                source_episode=episode_id,
            )
            new_rules.append(rule)

        self._rule_bank.increment_episode()
        return new_rules

    @property
    def rule_bank(self) -> RuleBank:
        return self._rule_bank

    def stats(self) -> dict[str, Any]:
        return {
            "rule_bank": self._rule_bank.stats(),
            "preflections_performed": len(self._preflection_log),
            "avoid_recommendations": sum(1 for r in self._preflection_log if r.recommendation == "avoid"),
            "proceed_recommendations": sum(1 for r in self._preflection_log if r.recommendation == "proceed"),
        }
