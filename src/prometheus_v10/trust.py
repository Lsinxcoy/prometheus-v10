"""Prometheus V9 Trust — 3-level trust + Moat 3-layer evaluation + SSGM consistency + semantic drift."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.schema import TrustLevel
from prometheus_v10.utils import jaccard_similarity

logger = logging.getLogger(__name__)


@dataclass
class MoatAssessment:
    data_layer: float       # Source reliability, data quality
    structure_layer: float  # Internal consistency, logical coherence
    governance_layer: float # Policy compliance, safety alignment
    overall: float
    issues: list[str] = field(default_factory=list)


class TrustManager:
    """3-level trust (PENDING→HIGH_SIGNAL→VERIFIED) + Moat + SSGM + drift detection."""

    TRUST_TRANSITIONS = {
        (TrustLevel.PENDING, TrustLevel.HIGH_SIGNAL): {"min_uses": 5, "min_success_rate": 0.7},
        (TrustLevel.HIGH_SIGNAL, TrustLevel.VERIFIED): {"min_uses": 20, "min_success_rate": 0.9},
    }

    def __init__(self) -> None:
        self._trust_records: dict[str, dict] = {}
        self._semantic_baselines: dict[str, str] = {}

    def get_trust(self, entity_id: str) -> TrustLevel:
        record = self._trust_records.get(entity_id)
        if not record:
            return TrustLevel.PENDING
        return TrustLevel(record.get("level", "pending"))

    def record_outcome(self, entity_id: str, success: bool) -> TrustLevel:
        """Record action outcome and potentially upgrade/downgrade trust."""
        if entity_id not in self._trust_records:
            self._trust_records[entity_id] = {"level": "pending", "uses": 0, "successes": 0}
        rec = self._trust_records[entity_id]
        rec["uses"] += 1
        if success:
            rec["successes"] += 1
        # Check for upgrade
        current = TrustLevel(rec["level"])
        rate = rec["successes"] / rec["uses"]
        if current == TrustLevel.PENDING and rate >= 0.7 and rec["uses"] >= 5:
            rec["level"] = TrustLevel.HIGH_SIGNAL.value
        elif current == TrustLevel.HIGH_SIGNAL and rate >= 0.9 and rec["uses"] >= 20:
            rec["level"] = TrustLevel.VERIFIED.value
        # Check for downgrade
        elif rate < 0.3 and rec["uses"] >= 5:
            rec["level"] = TrustLevel.PENDING.value
        return TrustLevel(rec["level"])

    def moat_assess(self, content: str, source: str, context: dict[str, Any] | None = None) -> MoatAssessment:
        """Moat 3-layer evaluation: data + structure + governance."""
        issues = []
        # Data layer: source reliability
        source_trust = self.get_trust(source)
        data_score = {TrustLevel.VERIFIED: 0.9, TrustLevel.HIGH_SIGNAL: 0.7, TrustLevel.PENDING: 0.3}.get(source_trust, 0.3)
        if data_score < 0.5:
            issues.append(f"Low data reliability: source trust={source_trust.value}")

        # Structure layer: internal consistency
        words = content.lower().split()
        unique_ratio = len(set(words)) / max(1, len(words))
        structure_score = min(1.0, unique_ratio * 2)  # More unique words = more structured
        if structure_score < 0.3:
            issues.append("Low structural coherence: repetitive content")

        # Governance layer: policy compliance
        governance_score = 0.8  # Default
        dangerous = ["rm -rf", "drop table", "delete from", "format c:"]
        for d in dangerous:
            if d in content.lower():
                governance_score -= 0.3
                issues.append(f"Governance violation: dangerous pattern '{d}'")

        overall = 0.4 * data_score + 0.3 * structure_score + 0.3 * governance_score
        return MoatAssessment(data_layer=data_score, structure_layer=structure_score, governance_layer=governance_score, overall=overall, issues=issues)

    def ssgm_consistency_check(self, old_content: str, new_content: str) -> bool:
        """SSGM consistency verification: modification must not contradict existing knowledge."""
        old_words = set(old_content.lower().split())
        new_words = set(new_content.lower().split())
        # High overlap = consistent modification
        overlap = jaccard_similarity(old_words, new_words)
        if overlap < 0.1:
            return False  # Completely different = suspicious
        # Check for direct contradictions
        removed = old_words - new_words
        added = new_words - old_words
        # Simple contradiction: removed positive, added negative (or vice versa)
        positive = {"good", "should", "must", "always", "best", "correct"}
        negative = {"bad", "never", "avoid", "worst", "wrong", "forbidden"}
        if (removed & positive and added & negative) or (removed & negative and added & positive):
            return False
        return True

    def detect_semantic_drift(self, entity_id: str, current_content: str, threshold: float = 0.5) -> bool:
        """Detect if entity's content has drifted too far from baseline."""
        if entity_id not in self._semantic_baselines:
            self._semantic_baselines[entity_id] = current_content
            return False
        baseline = self._semantic_baselines[entity_id]
        similarity = jaccard_similarity(set(baseline.lower().split()), set(current_content.lower().split()))
        if similarity < threshold:
            return True  # Significant drift
        self._semantic_baselines[entity_id] = current_content
        return False
