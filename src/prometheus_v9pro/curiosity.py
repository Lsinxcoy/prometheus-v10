"""Prometheus V9 Curiosity Queue — Priority-based exploration queue.

Implements T4 (hypothesis > fitting): structured curiosity drives autonomous exploration.
Without this, the system only explores what it already knows it needs.
"""
from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CuriosityItem:
    """An item in the curiosity queue."""
    question: str = ""
    priority: int = 5  # 1=highest, 10=lowest
    source: str = ""
    created_at: float = field(default_factory=time.time)
    explored: bool = False
    result: str = ""
    exploration_count: int = 0

    def __lt__(self, other: CuriosityItem) -> bool:
        return self.priority < other.priority


class CuriosityQueue:
    """Priority-based curiosity queue for autonomous exploration.

    Implements T4 (hypothesis > fitting): 
    - High priority: contradictory observations, anomalies
    - Medium priority: gaps in knowledge
    - Low priority: "what if" explorations (Gagarin channel)

    Quota management: max 50 questions, re-evaluate every 10 explorations.
    """

    def __init__(self, max_size: int = 50, reeval_interval: int = 10) -> None:
        self._max_size = max_size
        self._reeval_interval = reeval_interval
        self._queue: list[CuriosityItem] = []
        self._lock = threading.RLock()
        self._explored_count = 0
        self._generated_count = 0

    def add(self, question: str, priority: int = 5, source: str = "") -> CuriosityItem:
        """Add a question to the curiosity queue."""
        with self._lock:
            item = CuriosityItem(question=question, priority=priority, source=source)
            heapq.heappush(self._queue, item)
            self._generated_count += 1
            while len(self._queue) > self._max_size:
                heapq.heappop(self._queue)
            return item

    def pop(self) -> CuriosityItem | None:
        """Get the highest priority unexplored item."""
        with self._lock:
            while self._queue:
                item = heapq.heappop(self._queue)
                if not item.explored:
                    self._explored_count += 1
                    return item
            return None

    def peek(self) -> CuriosityItem | None:
        """Peek at the highest priority item without removing."""
        with self._lock:
            for item in sorted(self._queue):
                if not item.explored:
                    return item
            return None

    def mark_explored(self, question: str, result: str = "") -> None:
        """Mark a question as explored."""
        with self._lock:
            for item in self._queue:
                if item.question == question:
                    item.explored = True
                    item.result = result
                    item.exploration_count += 1

    def reprioritize(self, question: str, new_priority: int) -> bool:
        """Change priority of an existing question."""
        with self._lock:
            for item in self._queue:
                if item.question == question and not item.explored:
                    item.priority = new_priority
                    heapq.heapify(self._queue)
                    return True
            return False

    def auto_generate(self, knowledge_gaps: list[str], anomalies: list[str]) -> int:
        """Auto-generate curiosity questions from gaps and anomalies."""
        added = 0
        for gap in knowledge_gaps:
            self.add(f"How does {gap} work?", priority=3, source="gap_detection")
            added += 1
        for anomaly in anomalies:
            self.add(f"Why does {anomaly} behave unexpectedly?", priority=1, source="anomaly")
            added += 1
        # Gagarin channel: "what if" questions
        self.add("What if we try a completely different approach?", priority=8, source="gagarin")
        return added + 1

    def _reevaluate(self) -> None:
        """Re-evaluate priorities based on exploration results."""
        explored = [i for i in self._queue if i.explored]
        unexplored = [i for i in self._queue if not i.explored]
        # Boost priority of questions near successful explorations
        for u in unexplored:
            for e in explored:
                if e.result and any(w in u.question for w in e.result.split()):
                    u.priority = max(1, u.priority - 1)
        heapq.heapify(self._queue)

    @property
    def stats(self) -> dict[str, Any]:
        unexplored = [i for i in self._queue if not i.explored]
        return {
            "queue_size": len(self._queue),
            "unexplored": len(unexplored),
            "explored_count": self._explored_count,
            "generated_count": self._generated_count,
            "top_question": unexplored[0].question if unexplored else None,
        }
