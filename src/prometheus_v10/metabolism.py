"""Prometheus V9 Metabolism — Memory gravity + triage + decay + audit.

The life-or-death decision engine for memories. Every memory has a "gravity" score
that determines whether it should be promoted, kept, decayed, or deleted.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from prometheus_v10.schema import MemoryLayer, Node

logger = logging.getLogger(__name__)


class TriageDecision(str, Enum):
    PROMOTE = "promote"    # Upgrade to higher layer
    KEEP = "keep"          # Maintain current layer
    DECAY = "decay"        # Reduce importance
    DELETE = "delete"      # Remove from store
    ARCHIVE = "archive"    # Move to archive


@dataclass
class TriageResult:
    node_id: bytes = b""
    decision: TriageDecision = TriageDecision.KEEP
    reason: str = ""
    score: float = 0.0
    target_layer: MemoryLayer | None = None


class MetabolismEngine:
    """Memory metabolism: gravity + triage + decay + consolidate + audit.

    Memory Gravity Formula:
        G(node) = importance × (1 + log(1 + access_count)) × retention × freshness
        where freshness = exp(-(age_days / λ))
    """

    def __init__(self, store=None, decay_rate: float = 0.95,
                 min_importance_keep: float = 0.3) -> None:
        self._store = store
        self._decay_rate = decay_rate
        self._min_importance_keep = min_importance_keep
        self._audit_log: deque[dict] = deque(maxlen=1000)
        self._triage_count = 0

    def compute_gravity(self, node: Node) -> float:
        """Compute memory gravity score (higher = more valuable)."""
        importance = node.importance
        access_factor = 1.0 + math.log(1 + node.access_count)
        retention = node.retention
        age_days = max(0.001, (time.time() - node.created_at) / 86400.0)
        lam = getattr(node, 'weibull', None)
        lam_val = lam.lambda_ if lam else 86400.0
        freshness = math.exp(-age_days / max(1.0, lam_val / 86400.0))
        return importance * access_factor * retention * freshness

    def triage(self, node: Node) -> TriageResult:
        """Decide what to do with a memory node."""
        gravity = self.compute_gravity(node)
        self._triage_count += 1

        result = TriageResult(node_id=node.id, score=gravity)

        if gravity > 0.7 and node.layer == MemoryLayer.EPISODIC:
            result.decision = TriageDecision.PROMOTE
            result.target_layer = MemoryLayer.SEMANTIC
            result.reason = f"High gravity ({gravity:.3f}) warrants promotion"
        elif gravity > 0.3:
            result.decision = TriageDecision.KEEP
            result.reason = f"Gravity ({gravity:.3f}) above threshold"
        elif gravity > 0.1:
            result.decision = TriageDecision.DECAY
            result.reason = f"Low gravity ({gravity:.3f}), decaying importance"
        elif node.importance < self._min_importance_keep and node.access_count < 2:
            result.decision = TriageDecision.DELETE
            result.reason = f"Very low gravity ({gravity:.3f}) and rarely accessed"
        else:
            result.decision = TriageDecision.ARCHIVE
            result.reason = f"Gravity ({gravity:.3f}) below keep threshold"

        self._audit_log.append({
            "node_id": node.id.hex(), "gravity": gravity,
            "decision": result.decision.value, "reason": result.reason,
            "timestamp": time.time(),
        })
        return result

    def run_metabolism_cycle(self, nodes: list[Node] | None = None) -> dict[str, int]:
        """Run a full metabolism cycle over all or specified nodes."""
        if nodes is None and self._store:
            # Get all nodes from store — use FTS wildcard
            nodes = self._store.search_fts("*", limit=10000)

        stats = {"promoted": 0, "kept": 0, "decayed": 0, "deleted": 0, "archived": 0}

        for node in nodes:
            result = self.triage(node)

            if result.decision == TriageDecision.DECAY:
                node.importance *= self._decay_rate
                node.retention *= self._decay_rate
                if self._store:
                    self._store.update_node(node)
                stats["decayed"] += 1
            elif result.decision == TriageDecision.PROMOTE:
                if result.target_layer and self._store:
                    node.layer = result.target_layer
                    self._store.update_node(node)
                stats["promoted"] += 1
            elif result.decision == TriageDecision.DELETE:
                if self._store:
                    self._store.delete_node(node.id)
                stats["deleted"] += 1
            elif result.decision == TriageDecision.ARCHIVE:
                node.metadata["archived"] = True
                if self._store:
                    self._store.update_node(node)
                stats["archived"] += 1
            else:
                stats["kept"] += 1

        logger.info(f"Metabolism cycle: {stats}")
        return stats

    @property
    def audit_trail(self) -> list[dict]:
        return list(self._audit_log)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "triage_count": self._triage_count,
            "audit_entries": len(self._audit_log),
        }
