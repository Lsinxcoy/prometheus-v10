"""Prometheus V9 Minerva Import Adapter — Import data from Minerva V2 format.

Minerva V2 is one of the memory systems you researched (E:/hermes生成/Minerva_V2_Architecture.md).
Its data model has semantic nodes, episodic traces, and procedural patterns.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from prometheus_v9pro.schema import Node, NodeType, MemoryLayer, Provenance, ProvenanceType, NodePayload

logger = logging.getLogger(__name__)


class MinervaImportAdapter:
    """Import data from Minerva V2 format.

    Minerva V2 data model:
    - Semantic nodes: concept definitions, relationships
    - Episodic traces: event sequences with temporal markers
    - Procedural patterns: learned sequences and automations
    """

    def __init__(self, store=None) -> None:
        self._store = store
        self._imported_count = 0

    def import_file(self, path: str) -> int:
        """Import a Minerva V2 JSON file."""
        p = Path(path)
        if not p.exists():
            logger.error(f"File not found: {path}")
            return 0
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return 0
        return self._import_data(data)

    def import_directory(self, dir_path: str) -> int:
        """Import all Minerva V2 files from a directory."""
        total = 0
        for p in Path(dir_path).rglob("*.json"):
            total += self.import_file(str(p))
        return total

    def _import_data(self, data: dict | list) -> int:
        """Process imported data."""
        count = 0
        items = data if isinstance(data, list) else [data]
        for item in items:
            node = self._convert_item(item)
            if node and self._store:
                self._store.add_node(node)
                count += 1
        self._imported_count += count
        return count

    def _convert_item(self, item: dict) -> Node | None:
        """Convert Minerva V2 item to Prometheus Node."""
        content = item.get("content", "") or item.get("text", "") or json.dumps(item, ensure_ascii=False)
        if not content:
            return None
        # Determine node type
        item_type = item.get("type", "semantic")
        type_map = {
            "semantic": NodeType.FACT,
            "episodic": NodeType.EVENT,
            "procedural": NodeType.SKILL,
            "concept": NodeType.FACT,
            "trace": NodeType.EVENT,
            "pattern": NodeType.SKILL,
        }
        node_type = type_map.get(item_type, NodeType.FACT)
        # Determine layer
        layer_map = {
            "semantic": MemoryLayer.SEMANTIC,
            "episodic": MemoryLayer.EPISODIC,
            "procedural": MemoryLayer.PROCEDURAL,
        }
        layer = layer_map.get(item_type, MemoryLayer.SEMANTIC)

        node = Node(
            payload=NodePayload(content=content[:2000]),
            type=node_type,
            layer=layer,
            importance=float(item.get("importance", 0.5)),
            tags=item.get("tags", []),
        )
        node.provenance = Provenance(
            source=ProvenanceType.IMPORTED,
            agent_id=item.get("source_agent", "minerva_v2"),
            confidence=float(item.get("confidence", 0.5)),
        )
        return node

    @property
    def imported_count(self) -> int:
        return self._imported_count
