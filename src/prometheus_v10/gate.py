"""Prometheus V9 Confidence Gate — 3-level decision + OEP defense."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from prometheus_v10.utils import jaccard_similarity

logger = logging.getLogger(__name__)


class GateDecision(Enum):
    PROCEED = "proceed"
    ASK = "ask"
    DEFER = "defer"


@dataclass
class GateResult:
    decision: GateDecision
    reason: str
    confidence: float
    oep_suspect: bool = False


class ConfidenceGate:
    """3-level confidence gate + OEP clean-experience poisoning defense."""

    def __init__(self, ask_threshold: float = 0.7, defer_threshold: float = 0.5, oep_jaccard_threshold: float = 0.3) -> None:
        self._ask_threshold = ask_threshold
        self._defer_threshold = defer_threshold
        self._oep_threshold = oep_jaccard_threshold
        self._known_patterns: list[set[str]] = []

    def evaluate(self, content: str, content_type: str, confidence: float, existing_knowledge: list[str] | None = None) -> GateResult:
        """Evaluate content through confidence gate + OEP check."""
        # Confidence-based decision
        if confidence >= self._ask_threshold:
            decision = GateDecision.PROCEED
        elif confidence >= self._defer_threshold:
            decision = GateDecision.ASK
        else:
            decision = GateDecision.DEFER

        # OEP defense: check if content contradicts existing knowledge
        oep_suspect = False
        if existing_knowledge:
            new_words = set(content.lower().split())
            for existing in existing_knowledge:
                existing_words = set(existing.lower().split())
                similarity = jaccard_similarity(new_words, existing_words)
                # High similarity but different sentiment/stance = suspicious
                if similarity > 0.5 and self._is_contradictory(content, existing):
                    oep_suspect = True
                    break

        if oep_suspect and decision == GateDecision.PROCEED:
            decision = GateDecision.ASK
            logger.warning(f"OEP suspect content detected, downgrading to ASK")

        return GateResult(decision=decision, reason=f"confidence={confidence:.2f}", confidence=confidence, oep_suspect=oep_suspect)

    def _is_contradictory(self, a: str, b: str) -> bool:
        """Simple contradiction detection: opposing sentiment words."""
        positive = {"good", "great", "excellent", "should", "must", "always", "best"}
        negative = {"bad", "terrible", "never", "avoid", "worst", "forbidden", "harmful"}
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        a_pos = bool(a_words & positive)
        a_neg = bool(a_words & negative)
        b_pos = bool(b_words & positive)
        b_neg = bool(b_words & negative)
        return (a_pos and b_neg) or (a_neg and b_pos)
