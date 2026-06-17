"""Prometheus V9 Hermes Plugin Adapter — Bridge between Prometheus and Hermes Agent.

This is the interface that allows Hermes to use Prometheus as a memory provider.
Hermes calls: save/recall/forget → Prometheus handles the evolution side.
"""
from __future__ import annotations

import logging

from prometheus_v9pro.schema import Node, NodeType, NodePayload, MemoryLayer

logger = logging.getLogger(__name__)


class HermesPluginAdapter:
    """Adapter for Hermes Agent plugin integration.

    Allows Prometheus V9 to be used as a Hermes memory provider:
    - save(key, value) → store as fact node
    - recall(query) → FTS search
    - forget(key) → delete matching nodes
    - evolve(steps=N) → run evolution cycle
    - status() → get system status
    """

    def __init__(self, life=None) -> None:
        self._life = life
        self._name = "prometheus-v9"

    @property
    def name(self) -> str:
        return self._name

    def save(self, key: str, value: str, metadata: dict | None = None) -> bool:
        """Save a memory entry (Hermes plugin interface)."""
        if not self._life:
            return False
        node = Node(
            payload=NodePayload(content=f"{key}: {value}"),
            type=NodeType.FACT,
            layer=MemoryLayer.SEMANTIC,
            tags=[key],
        )
        if metadata:
            node.metadata = metadata
        nid = self._life.store.add_node(node)
        return nid is not None

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Recall memories matching query (Hermes plugin interface)."""
        if not self._life:
            return []
        nodes = self._life.store.search_fts(query, limit)
        return [n.payload.content for n in nodes]

    def forget(self, key: str) -> bool:
        """Remove a memory by key."""
        if not self._life:
            return False
        nodes = self._life.store.search_fts(key, limit=10)
        removed = 0
        for node in nodes:
            if key in node.tags or key in node.payload.content:
                self._life.store.delete_node(node.id)
                removed += 1
        return removed > 0

    def evolve(self, steps: int = 1) -> list[dict]:
        """Run evolution cycle."""
        if not self._life:
            return []
        results = self._life.evolve(steps=steps)
        return [{"step": i, "layers": len(r["layer_results"])} for i, r in enumerate(results)]

    def status(self) -> dict:
        """Get system status."""
        if not self._life:
            return {"version": "9.0.0", "status": "not_initialized"}
        return self._life.status()
