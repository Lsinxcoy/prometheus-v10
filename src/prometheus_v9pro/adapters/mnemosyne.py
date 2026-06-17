"""Prometheus V9 Mnemosyne Adapter — Mnemosyne V3 memory store adapter with Hallway transfer.

Mnemosyne is one of the memory systems you researched (E:/hermes生成). 
This adapter allows Prometheus to:
- Store/retrieve memories in Mnemosyne format
- Transfer memories between agents via Hallway protocol
- Import/export in Mnemosyne V3 JSON format
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus_v9pro.schema import Node, NodeType, MemoryLayer, Provenance, ProvenanceType

logger = logging.getLogger(__name__)


@dataclass
class HallwayTransfer:
    """A hallway transfer record — moving memories between agents."""
    transfer_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    node_ids: list[bytes] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending/completed/failed
    transferred_count: int = 0


class MnemosyneAdapter:
    """Adapter for Mnemosyne V3 memory store.

    Enables:
    1. Direct access to Mnemosyne database for shared memory
    2. Hallway transfer between agents (cross-agent memory sharing)
    3. JSON import/export in Mnemosyne V3 format
    """

    def __init__(self, db_path: str = "data/mnemosyne_v3.db", store=None) -> None:
        self._db_path = db_path
        self._store = store
        self._connected = False
        self._hallway_log: list[HallwayTransfer] = []

    def connect(self) -> bool:
        """Connect to Mnemosyne store."""
        try:
            from prometheus_v9pro.store import SQLiteStore
            self._store = SQLiteStore(self._db_path)
            self._connected = True
            logger.info(f"Connected to Mnemosyne at {self._db_path}")
            return True
        except Exception as e:
            logger.warning(f"Mnemosyne connection failed: {e}")
            return False

    def store_node(self, node: Node) -> bool:
        """Store a node in Mnemosyne."""
        if not self._connected and not self.connect():
            return False
        try:
            self._store.add_node(node)
            return True
        except Exception as e:
            logger.warning(f"Mnemosyne store error: {e}")
            return False

    def retrieve_node(self, node_id: bytes) -> Node | None:
        """Retrieve a node from Mnemosyne."""
        if not self._connected and not self.connect():
            return None
        return self._store.get_node(node_id)

    def search(self, query: str, limit: int = 10) -> list[Node]:
        """Search Mnemosyne memories."""
        if not self._connected and not self.connect():
            return []
        return self._store.search_fts(query, limit)

    def transfer_hallway(self, from_agent: str, to_agent: str,
                         node_ids: list[bytes], filter_fn=None) -> HallwayTransfer:
        """Transfer hallway nodes between agents.

        For each node:
        1. Retrieve the node
        2. Apply filter (if provided) — e.g. only transfer HIGH trust level
        3. Update owner_agent metadata
        4. Store the modified node
        5. Log in hallway audit trail
        """
        transfer = HallwayTransfer(
            from_agent=from_agent, to_agent=to_agent, node_ids=node_ids,
        )
        logger.info(f"Hallway transfer: {from_agent} → {to_agent}, {len(node_ids)} nodes")

        for nid in node_ids:
            node = self.retrieve_node(nid)
            if not node:
                logger.warning(f"Node {nid.hex()} not found for hallway transfer")
                continue
            if filter_fn and not filter_fn(node):
                continue  # Filter rejects this node
            # Update owner
            node.metadata["owner_agent"] = to_agent
            node.metadata["previous_owner"] = from_agent
            node.metadata["transfer_timestamp"] = time.time()
            if self.store_node(node):
                transfer.transferred_count += 1

        transfer.status = "completed" if transfer.transferred_count > 0 else "failed"
        self._hallway_log.append(transfer)
        logger.info(f"Hallway complete: {transfer.transferred_count}/{len(node_ids)} transferred")
        return transfer

    def export_json(self, path: str, agent_filter: str | None = None) -> int:
        """Export memories in Mnemosyne V3 JSON format."""
        if not self._connected and not self.connect():
            return 0
        nodes = self._store.search_fts("*", limit=10000)  # Get all
        if agent_filter:
            nodes = [n for n in nodes if n.metadata.get("owner_agent") == agent_filter]
        data = [self._node_to_mnemosyne(n) for n in nodes]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(data)

    def import_json(self, path: str) -> int:
        """Import memories from Mnemosyne V3 JSON file."""
        p = Path(path)
        if not p.exists():
            logger.error(f"File not found: {path}")
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        count = 0
        for item in items:
            node = self._mnemosyne_to_node(item)
            if node and self._store:
                self._store.add_node(node)
                count += 1
        logger.info(f"Imported {count} items from {path}")
        return count

    def _node_to_mnemosyne(self, node: Node) -> dict[str, Any]:
        """Convert Prometheus Node to Mnemosyne V3 format."""
        return {
            "content": node.payload.content,
            "type": node.type.value,
            "importance": node.importance,
            "tags": node.tags,
            "layer": node.layer.value,
            "trust_level": node.trust_level.value if node.trust_level else "unverified",
            "provenance": {
                "source": node.provenance.source.value if node.provenance else "unknown",
                "agent_id": node.provenance.agent_id if node.provenance else "",
                "confidence": node.provenance.confidence if node.provenance else 0.0,
            },
            "metadata": node.metadata,
            "created_at": node.created_at,
        }

    def _mnemosyne_to_node(self, item: dict) -> Node | None:
        """Convert Mnemosyne V3 format to Prometheus Node."""
        content = item.get("content", "") or item.get("text", "") or json.dumps(item, ensure_ascii=False)
        if not content:
            return None
        node_type_str = item.get("type", "fact")
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            node_type = NodeType.FACT
        node = Node(
            payload=NodePayload(content=content[:2000]),
            type=node_type,
            layer=MemoryLayer.SEMANTIC,
            importance=float(item.get("importance", 0.5)),
            tags=item.get("tags", []),
        )
        node.provenance = Provenance(
            source=ProvenanceType.IMPORTED,
            agent_id=item.get("source_agent", ""),
            confidence=float(item.get("confidence", 0.5)),
        )
        return node

    @property
    def hallway_stats(self) -> dict[str, Any]:
        """Hallway transfer statistics."""
        completed = [t for t in self._hallway_log if t.status == "completed"]
        return {
            "total_transfers": len(self._hallway_log),
            "completed_transfers": len(completed),
            "total_nodes_transferred": sum(t.transferred_count for t in completed),
        }
