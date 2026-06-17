"""Prometheus V9 Daily Learning Cycle — 5-step: Learn→Reflect→Reason→Derive→Apply.

From MiMo V5 dropped feature, restored and enhanced.
Each round produces a LearningRound with scores, stored as knowledge nodes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v10.schema import Node, NodeType, NodePayload, MemoryLayer

logger = logging.getLogger(__name__)


@dataclass
class LearningRound:
    """One round of daily learning."""
    id: str = ""
    topic: str = ""
    learned: str = ""
    reflected: str = ""
    reasoned: str = ""
    derived: str = ""
    applied: str = ""
    score: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DailyLearningCycle:
    """5-step daily learning cycle with quota management.

    1. LEARN: Acquire new knowledge from exploration
    2. REFLECT: Review what was learned, identify gaps
    3. REASON: Apply reasoning to derive implications
    4. DERIVE: Extract actionable principles
    5. APPLY: Apply derived principles to current tasks

    Quota: max 20 rounds/day, revision every 5 rounds.
    """

    def __init__(self, store=None, llm=None, daily_quota: int = 20,
                 revision_interval: int = 5) -> None:
        self._store = store
        self._llm = llm
        self._daily_quota = daily_quota
        self._revision_interval = revision_interval
        self._rounds_today = 0
        self._today = time.strftime("%Y-%m-%d")
        self._all_rounds: list[LearningRound] = []
        self._revision_count = 0

    def run_cycle(self, topic: str, content: str) -> LearningRound:
        """Run one 5-step learning cycle."""
        self._check_day_reset()
        if self._rounds_today >= self._daily_quota:
            logger.info("Daily learning quota reached")
            return LearningRound(topic=topic, score=0.0)

        round_id = f"lr_{self._rounds_today}_{int(time.time())}"
        lr = LearningRound(id=round_id, topic=topic)

        lr.learned = self._learn(topic, content)
        lr.reflected = self._reflect(lr.learned)
        lr.reasoned = self._reason(lr.learned, lr.reflected)
        lr.derived = self._derive(lr.reasoned)
        lr.applied = self._apply(lr.derived)
        lr.score = self._score_round(lr)

        # Store as knowledge node
        node = Node(
            payload=NodePayload(content=f"[{topic}] {lr.derived}"),
            type=NodeType.FACT,
            layer=MemoryLayer.SEMANTIC,
            tags=["daily_learning", topic],
            importance=lr.score,
        )
        if self._store:
            self._store.add_node(node)

        self._rounds_today += 1
        self._all_rounds.append(lr)

        # Revision every N rounds
        if self._rounds_today % self._revision_interval == 0:
            self._revise()

        return lr

    def _learn(self, topic: str, content: str) -> str:
        """Step 1: Acquire knowledge."""
        if self._llm and self._llm.has_real_llm:
            return self._llm.generate(
                f"Learn about {topic} from this content:\n{content}",
                system="You are a learning agent. Extract key facts and patterns.",
                heuristic_fn=lambda p: f"Learned: {content[:200]}",
            )
        return f"Learned: {content[:200]}"

    def _reflect(self, learned: str) -> str:
        """Step 2: Reflect on what was learned."""
        if self._llm and self._llm.has_real_llm:
            return self._llm.generate(
                f"Reflect on what was learned:\n{learned}\nIdentify gaps and contradictions.",
                heuristic_fn=lambda p: f"Reflected: gaps in {learned[:100]}",
            )
        return f"Reflected: gaps in understanding of {learned[:100]}"

    def _reason(self, learned: str, reflected: str) -> str:
        """Step 3: Apply reasoning."""
        if self._llm and self._llm.has_real_llm:
            return self._llm.generate(
                f"Reason about:\nLearned: {learned}\nGaps: {reflected}",
                heuristic_fn=lambda p: f"Reasoned: implications of {learned[:100]}",
            )
        return f"Reasoned: logical implications of {learned[:100]}"

    def _derive(self, reasoned: str) -> str:
        """Step 4: Extract principles."""
        if self._llm and self._llm.has_real_llm:
            return self._llm.generate(
                f"Extract actionable principles from:\n{reasoned}",
                heuristic_fn=lambda p: f"Derived: principle from {reasoned[:100]}",
            )
        return f"Derived: general principle from {reasoned[:100]}"

    def _apply(self, derived: str) -> str:
        """Step 5: Apply to current context."""
        if self._llm and self._llm.has_real_llm:
            return self._llm.generate(
                f"How to apply this principle:\n{derived}",
                heuristic_fn=lambda p: f"Applied: {derived[:100]}",
            )
        return f"Applied: {derived[:100]}"

    def _score_round(self, lr: LearningRound) -> float:
        """Score learning round quality."""
        scores = []
        # Each step gets 0.2 max for content length > 0
        for attr in ["learned", "reflected", "reasoned", "derived", "applied"]:
            val = getattr(lr, attr, "")
            scores.append(min(1.0, len(val) / 50.0) * 0.2)
        return sum(scores)

    def _revise(self) -> None:
        """Revise learning: consolidate recent rounds."""
        self._revision_count += 1
        recent = self._all_rounds[-self._revision_interval:]
        avg_score = sum(r.score for r in recent) / max(1, len(recent))
        logger.info(f"Revision {self._revision_count}: avg score = {avg_score:.2f}")

    def _check_day_reset(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self._rounds_today = 0

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "rounds_today": self._rounds_today,
            "total_rounds": len(self._all_rounds),
            "revisions": self._revision_count,
            "avg_score": sum(r.score for r in self._all_rounds) / max(1, len(self._all_rounds)),
        }
