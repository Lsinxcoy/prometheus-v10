"""Prometheus V9 Memory Compression — Semantic compression, summarization, merging."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import Node

logger = logging.getLogger(__name__)


class MemoryCompressor:
    """Compress memories: merge similar, summarize long, prune redundant.

    Compression strategies:
    1. Merge: Nodes with Jaccard similarity > 0.8 → keep higher importance
    2. Summarize: Nodes > 500 chars → extract key points
    3. Prune: Nodes with importance < 0.1 and age > 30 days → delete
    """

    def __init__(self, store=None, llm=None, similarity_threshold: float = 0.8) -> None:
        self._store = store
        self._llm = llm
        self._threshold = similarity_threshold
        self._stats = {"merged": 0, "summarized": 0, "pruned": 0}

    def compress(self, nodes: list[Node] | None = None) -> dict[str, int]:
        """Run compression cycle."""
        if nodes is None and self._store:
            nodes = self._store.search_fts("*", limit=10000)
        if not nodes:
            return self._stats

        # 1. Merge similar nodes
        self._merge_similar(nodes)

        # 2. Summarize long nodes
        self._summarize_long(nodes)

        # 3. Prune low-value nodes
        self._prune_redundant(nodes)

        return self._stats

    def _merge_similar(self, nodes: list[Node]) -> None:
        from prometheus_v9pro.utils import jaccard_similarity
        merged_ids = set()
        for i, n1 in enumerate(nodes):
            if n1.id in merged_ids:
                continue
            for j, n2 in enumerate(nodes):
                if j <= i or n2.id in merged_ids:
                    continue
                sim = jaccard_similarity(n1.payload.content, n2.payload.content)
                if sim > self._threshold:
                    # Keep the higher importance one
                    if n1.importance >= n2.importance:
                        merged_ids.add(n2.id)
                    else:
                        merged_ids.add(n1.id)
                    self._stats["merged"] += 1

        # Delete merged nodes from store
        if self._store:
            for nid in merged_ids:
                self._store.delete_node(nid)

    def _summarize_long(self, nodes: list[Node]) -> None:
        for node in nodes:
            if len(node.payload.content) > 500:
                # Simple truncation-based summary (LLM version if available)
                summary = node.payload.content[:200] + "..."
                if self._llm and self._llm.has_real_llm:
                    summary = self._llm.generate(
                        f"Summarize in 200 chars: {node.payload.content[:1000]}",
                        heuristic_fn=lambda p: node.payload.content[:200] + "...",
                    )
                node.payload.content = summary
                self._stats["summarized"] += 1

    def _prune_redundant(self, nodes: list[Node]) -> None:
        import time
        now = time.time()
        for node in nodes:
            age_days = (now - node.created_at) / 86400.0
            if node.importance < 0.1 and age_days > 30:
                if self._store:
                    self._store.delete_node(node.id)
                self._stats["pruned"] += 1
