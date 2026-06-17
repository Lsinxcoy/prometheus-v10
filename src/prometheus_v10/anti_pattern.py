"""Prometheus V10 Anti-Pattern Memory — Failure signatures + causal attribution + veto + epitaph.

Inspired by SkillSmith (arXiv:2606.01314) §E (Anti-Pattern Memory):
- Record verified failure modes with signatures, attributions, and remedies
- Diagnostic acceleration: retrieve similar past failures during reflection
- Proposal veto: reject proposals that repeat known mistakes
- Retirement→epitaph: retired skills transfer lessons to anti-pattern memory

Key difference from V9PRO:
- V9PRO anti_evo.py: only dedup + zero-gain (no failure memory)
- V10 anti_pattern.py: structured failure knowledge base with veto power
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.utils import jaccard_similarity

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────

@dataclass
class FailureSignature:
    """Structured failure signature for matching and retrieval."""
    failure_type: str = ""         # e.g., "tool_error", "skill_conflict", "validation_failed"
    component: str = ""            # e.g., "search_tool", "qa_skill", "code_validator"
    error_pattern: str = ""        # e.g., "KeyError: 'results'", "timeout>30s"
    context_keywords: list[str] = field(default_factory=list)  # keywords from the execution context
    layer: int = 0                 # evolution layer where failure occurred

    def fingerprint(self) -> str:
        """Deterministic fingerprint for deduplication."""
        raw = f"{self.failure_type}:{self.component}:{self.error_pattern}:{sorted(self.context_keywords)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class CausalAttribution:
    """Why did the failure happen?"""
    root_cause: str = ""           # e.g., "tool_missing_capability", "skill_orchestration_bug"
    affected_skills: list[str] = field(default_factory=list)
    affected_tools: list[str] = field(default_factory=list)
    confidence: float = 0.5        # how confident we are in this attribution [0,1]
    evidence: list[str] = field(default_factory=list)  # supporting evidence from traces


@dataclass
class Remedy:
    """How to fix or avoid this failure."""
    action_type: str = ""          # "edit_tool", "wrap_tool", "retire_skill", "add_skill", "split_tool"
    target: str = ""               # component to modify
    description: str = ""          # human-readable description
    attempted: bool = False        # whether this remedy was attempted
    successful: bool | None = None # whether it worked (None = not yet evaluated)


@dataclass
class AntiPatternRecord:
    """One complete anti-pattern entry."""
    signature: FailureSignature = field(default_factory=FailureSignature)
    attribution: CausalAttribution = field(default_factory=CausalAttribution)
    remedies: list[Remedy] = field(default_factory=list)
    occurrence_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    vetted: bool = False           # whether this pattern has been verified by a second occurrence

    @property
    def severity(self) -> float:
        """Severity score based on occurrence count and attribution confidence."""
        return min(1.0, self.occurrence_count * 0.2) * self.attribution.confidence


@dataclass
class Epitaph:
    """Lessons transferred from a retired skill/tool."""
    component_name: str = ""
    component_type: str = ""       # "skill" or "tool"
    retirement_reason: str = ""
    key_failures: list[str] = field(default_factory=list)  # failure fingerprints
    lessons_learned: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ── Anti-Pattern Memory ────────────────────────────────────────────

class AntiPatternMemory:
    """Structured failure knowledge base with diagnostic acceleration and proposal veto.

    Three mechanisms from SkillSmith §E:
    1. Diagnostic acceleration: retrieve similar failures during reflection
    2. Proposal veto: reject proposals matching known anti-patterns
    3. Retirement→epitaph: transfer lessons from retired components
    """

    def __init__(self, max_records: int = 5000, similarity_threshold: float = 0.6) -> None:
        self._records: list[AntiPatternRecord] = []
        self._fingerprint_index: dict[str, int] = {}  # fingerprint → index
        self._type_index: dict[str, list[int]] = {}    # failure_type → [indices]
        self._epitaphs: list[Epitaph] = []
        self._max_records = max_records
        self._similarity_threshold = similarity_threshold

    # ── Recording ───────────────────────────────────────────────

    def record_failure(self, signature: FailureSignature, attribution: CausalAttribution,
                       remedies: list[Remedy] | None = None) -> AntiPatternRecord:
        """Record a failure occurrence. If similar pattern exists, increment count."""
        fp = signature.fingerprint()

        # Check for existing record with same fingerprint
        if fp in self._fingerprint_index:
            idx = self._fingerprint_index[fp]
            record = self._records[idx]
            record.occurrence_count += 1
            record.last_seen = time.time()
            # Promote to vetted after 2+ occurrences
            if record.occurrence_count >= 2:
                record.vetted = True
            # Update attribution confidence with more evidence
            if attribution.evidence:
                record.attribution.evidence.extend(attribution.evidence[-3:])
            logger.debug(f"Updated anti-pattern {fp} (count={record.occurrence_count})")
            return record

        # New record
        record = AntiPatternRecord(
            signature=signature,
            attribution=attribution,
            remedies=remedies or [],
            vetted=False,
        )
        idx = len(self._records)
        self._records.append(record)
        self._fingerprint_index[fp] = idx

        # Type index
        ftype = signature.failure_type
        if ftype not in self._type_index:
            self._type_index[ftype] = []
        self._type_index[ftype].append(idx)

        # Evict oldest if over capacity
        if len(self._records) > self._max_records:
            self._evict_oldest()

        logger.info(f"Recorded new anti-pattern: {signature.failure_type} in {signature.component}")
        return record

    # ── Diagnostic Acceleration ─────────────────────────────────

    def retrieve_similar(self, signature: FailureSignature, top_k: int = 5) -> list[AntiPatternRecord]:
        """Retrieve anti-patterns similar to a new failure.

        Used during reflection to inject prior attributions into diagnostic context,
        helping the system avoid previously failed configurations.
        """
        if not self._records:
            return []

        # Score each record by similarity
        scored: list[tuple[float, AntiPatternRecord]] = []
        query_keywords = set(signature.context_keywords)

        for record in self._records:
            sim = self._compute_similarity(signature, record.signature, query_keywords)
            if sim >= self._similarity_threshold:
                # Boost by severity (prefer well-established patterns)
                boost = 1.0 + 0.3 * record.severity
                scored.append((sim * boost, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def get_diagnostic_context(self, signature: FailureSignature) -> dict[str, Any]:
        """Get diagnostic context for a failure: similar past failures + their attributions + remedies."""
        similar = self.retrieve_similar(signature, top_k=3)
        if not similar:
            return {"known_patterns": [], "suggested_remedies": []}

        return {
            "known_patterns": [
                {
                    "type": r.signature.failure_type,
                    "component": r.signature.component,
                    "root_cause": r.attribution.root_cause,
                    "occurrence_count": r.occurrence_count,
                    "vetted": r.vetted,
                }
                for r in similar
            ],
            "suggested_remedies": [
                {"action": rem.action_type, "target": rem.target, "description": rem.description}
                for r in similar
                for rem in r.remedies
                if rem.successful is not False  # include untested and successful remedies
            ],
        }

    # ── Proposal Veto ───────────────────────────────────────────

    def should_veto(self, proposal_type: str, target: str,
                    proposal_keywords: list[str] | None = None) -> tuple[bool, str]:
        """Check if a proposal should be vetoed because it repeats a known anti-pattern.

        Returns (should_veto, reason).
        """
        # Check if this target was the subject of a vetted failure
        for record in self._records:
            if not record.vetted:
                continue

            # Direct match on target component
            if target in record.attribution.affected_skills or target in record.attribution.affected_tools:
                # Check if the proposal type matches a known failed remedy
                for rem in record.remedies:
                    if rem.action_type == proposal_type and rem.attempted and rem.successful is False:
                        return True, (f"Proposal '{proposal_type}' on '{target}' matches failed pattern: "
                                      f"{record.signature.failure_type} — {record.attribution.root_cause}")

            # Keyword similarity match
            if proposal_keywords:
                record_keywords = set(record.signature.context_keywords)
                proposal_set = set(proposal_keywords)
                if record_keywords and proposal_set:
                    sim = jaccard_similarity(record_keywords, proposal_set)
                    if sim > 0.7 and record.severity > 0.5:
                        return True, (f"Proposal keywords overlap with anti-pattern: "
                                      f"{record.signature.failure_type} (similarity={sim:.2f})")

        return False, ""

    # ── Retirement → Epitaph ────────────────────────────────────

    def record_epitaph(self, component_name: str, component_type: str,
                       retirement_reason: str, key_failures: list[str],
                       lessons_learned: list[str]) -> Epitaph:
        """Transfer lessons from a retired skill/tool into anti-pattern memory.

        This prevents the system from regenerating retired components
        that embody known failure modes.
        """
        epitaph = Epitaph(
            component_name=component_name,
            component_type=component_type,
            retirement_reason=retirement_reason,
            key_failures=key_failures,
            lessons_learned=lessons_learned,
        )
        self._epitaphs.append(epitaph)

        logger.info(f"Recorded epitaph for {component_type} '{component_name}': {retirement_reason}")
        return epitaph

    def is_retired(self, component_name: str) -> bool:
        """Check if a component has been retired (exists in epitaphs)."""
        return any(e.component_name == component_name for e in self._epitaphs)

    def get_epitaph(self, component_name: str) -> Epitaph | None:
        """Get the epitaph for a retired component."""
        for e in self._epitaphs:
            if e.component_name == component_name:
                return e
        return None

    # ── Similarity Computation ──────────────────────────────────

    def _compute_similarity(self, query: FailureSignature, candidate: FailureSignature,
                            query_keywords: set[str]) -> float:
        """Compute similarity between two failure signatures."""
        score = 0.0

        # Exact type match
        if query.failure_type == candidate.failure_type:
            score += 0.3

        # Component match
        if query.component == candidate.component:
            score += 0.3

        # Error pattern substring match
        if query.error_pattern and candidate.error_pattern:
            if query.error_pattern in candidate.error_pattern or candidate.error_pattern in query.error_pattern:
                score += 0.2

        # Keyword overlap
        if query_keywords and candidate.context_keywords:
            candidate_keywords = set(candidate.context_keywords)
            sim = jaccard_similarity(query_keywords, candidate_keywords)
            score += 0.2 * sim

        return min(1.0, score)

    def _evict_oldest(self) -> None:
        """Evict the oldest non-vetted record to stay under capacity."""
        for i, record in enumerate(self._records):
            if not record.vetted:
                fp = record.signature.fingerprint()
                del self._fingerprint_index[fp]
                self._type_index.get(record.signature.failure_type, []).remove(i)
                self._records.pop(i)
                return

    def stats(self) -> dict[str, Any]:
        vetted_count = sum(1 for r in self._records if r.vetted)
        type_counts = {k: len(v) for k, v in self._type_index.items()}
        return {
            "total_records": len(self._records),
            "vetted_patterns": vetted_count,
            "epitaphs": len(self._epitaphs),
            "by_type": type_counts,
        }
