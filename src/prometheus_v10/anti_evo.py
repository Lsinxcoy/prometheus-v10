"""Prometheus V9 Anti-Evolution Gate — Dedup, insight, applicability, zero-gain checks."""

from __future__ import annotations

import logging

from prometheus_v10.schema import RawDesign
from prometheus_v10.utils import jaccard_similarity

logger = logging.getLogger(__name__)


class AntiEvolutionGate:
    """Prevent wasteful evolution: dedup, no-insight, inapplicable, zero-gain."""

    def check_deduplication(self, design: RawDesign, history: list[RawDesign]) -> bool:
        """Reject if too similar to existing designs (>0.9 Jaccard on inputs)."""
        new_inputs = set(design.inputs)
        for past in history:
            past_inputs = set(past.inputs)
            if jaccard_similarity(new_inputs, past_inputs) > 0.9:
                return False  # Too similar
        return True

    def check_insight(self, design: RawDesign) -> bool:
        """Design must have at least 1 innovation dimension or novel input."""
        if design.innovation_dimensions and len(design.innovation_dimensions) > 0:
            return True
        if len(set(design.inputs)) > 1:
            return True  # Multiple inputs = potential novel combination
        return False

    def check_applicability(self, design: RawDesign) -> bool:
        """Design type must be applicable."""
        return design.design_type in ("skill", "config", "code")

    def check_zero_gain(self, fitness_history: list[float], window: int = 5) -> bool:
        """If recent fitness deltas are all ~0, stop evolution."""
        if len(fitness_history) < window:
            return False  # Not enough data to decide
        recent = fitness_history[-window:]
        return all(abs(d) < 0.001 for d in recent)

    def gate(self, design: RawDesign, history: list[RawDesign], fitness_history: list[float] | None = None) -> tuple[bool, str]:
        """Run all checks. Returns (pass, reason)."""
        if not self.check_deduplication(design, history):
            return False, "duplicate"
        if not self.check_insight(design):
            return False, "no_insight"
        if not self.check_applicability(design):
            return False, "not_applicable"
        if fitness_history and self.check_zero_gain(fitness_history):
            return False, "zero_gain"
        return True, "passed"
