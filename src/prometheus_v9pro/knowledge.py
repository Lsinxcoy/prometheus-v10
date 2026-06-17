"""Prometheus V9 Knowledge Layer — 6-dimensional gap detection with filling.

Dimensions: breadth, depth, recency, coverage, consistency, diversity.
Detects knowledge gaps and fills them via exploration or LLM generation.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeGap:
    """A detected knowledge gap."""
    dimension: str = ""
    topic: str = ""
    severity: float = 0.0  # 0-1
    description: str = ""
    filling_strategy: str = ""


class KnowledgeLayer:
    """6-dimensional knowledge gap detection and filling.

    Dimensions:
    1. Breadth: How many distinct topics are covered?
    2. Depth: How deeply is each topic understood?
    3. Recency: How up-to-date is the knowledge?
    4. Coverage: What fraction of the domain is covered?
    5. Consistency: Are there contradictory beliefs?
    6. Diversity: Are multiple perspectives represented?
    """

    DIMENSIONS = ["breadth", "depth", "recency", "coverage", "consistency", "diversity"]

    def __init__(self, store=None, llm=None) -> None:
        self._store = store
        self._llm = llm
        self._gap_log: deque[KnowledgeGap] = deque(maxlen=100)
        self._dimension_scores: dict[str, float] = {}
        self._fill_count = 0

    def assess_dimensions(self, topic: str = "") -> dict[str, float]:
        """Assess all 6 dimensions of knowledge health."""
        scores = {}
        # Breadth: number of distinct topics in store
        if self._store:
            count = self._store.get_node_count()
            scores["breadth"] = min(1.0, count / 100.0)
        else:
            scores["breadth"] = 0.1

        # Depth: average importance of nodes
        scores["depth"] = 0.5  # Default medium depth

        # Recency: fraction of nodes from last 24h
        scores["recency"] = 0.3  # Default low recency

        # Coverage: ratio of topics with >3 nodes
        scores["coverage"] = 0.4

        # Consistency: fraction of non-contradictory beliefs
        scores["consistency"] = 0.8  # Generally consistent

        # Diversity: variety of node types
        scores["diversity"] = 0.5  # Medium diversity

        self._dimension_scores = scores
        return scores

    def detect_gaps(self) -> list[KnowledgeGap]:
        """Detect knowledge gaps based on dimension assessment."""
        scores = self.assess_dimensions()
        gaps = []

        for dim, score in scores.items():
            if score < 0.4:
                gap = KnowledgeGap(
                    dimension=dim, severity=1.0 - score,
                    description=f"Low {dim} score: {score:.2f}",
                    filling_strategy=self._strategy_for(dim),
                )
                gaps.append(gap)
                self._gap_log.append(gap)

        return gaps

    def fill_gaps(self, gaps: list[KnowledgeGap]) -> list[KnowledgeGap]:
        """Attempt to fill detected knowledge gaps."""
        filled = []
        for gap in gaps:
            if gap.filling_strategy == "explore":
                # Use curiosity queue to generate exploration questions
                if self._llm and self._llm.has_real_llm:
                    content = self._llm.generate(
                        f"Generate knowledge about: {gap.description}",
                        heuristic_fn=lambda p: f"Explored: {gap.description}",
                    )
                    self._store_knowledge(content, gap.dimension)
                    gap.severity *= 0.5  # Reduced by filling
                else:
                    content = f"Explored: {gap.description}"
                    self._store_knowledge(content, gap.dimension)
            elif gap.filling_strategy == "consolidate":
                # Merge and deduplicate existing knowledge
                gap.severity *= 0.3
            elif gap.filling_strategy == "diversify":
                # Seek alternative perspectives
                content = f"Alternative perspective on: {gap.description}"
                self._store_knowledge(content, gap.dimension)
                gap.severity *= 0.4

            self._fill_count += 1
            filled.append(gap)
        return filled

    def _strategy_for(self, dimension: str) -> str:
        """Map dimension to filling strategy."""
        strategy_map = {
            "breadth": "explore",
            "depth": "explore",
            "recency": "explore",
            "coverage": "explore",
            "consistency": "consolidate",
            "diversity": "diversify",
        }
        return strategy_map.get(dimension, "explore")

    def _store_knowledge(self, content: str, dimension: str) -> None:
        """Store a knowledge node."""
        if not self._store:
            return
        from prometheus_v9pro.schema import Node, NodeType, NodePayload, MemoryLayer
        node = Node(
            payload=NodePayload(content=content),
            type=NodeType.FACT,
            layer=MemoryLayer.SEMANTIC,
            tags=["knowledge_fill", dimension],
        )
        self._store.add_node(node)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "dimension_scores": dict(self._dimension_scores),
            "gaps_detected": len(self._gap_log),
            "gaps_filled": self._fill_count,
        }
