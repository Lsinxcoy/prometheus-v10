"""Prometheus V9 Graph — GraphProtocol + SQLGraphBackend (1 backend)."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Protocol, runtime_checkable

from pathlib import Path

logger = logging.getLogger(__name__)


@runtime_checkable
class GraphProtocol(Protocol):
    """Graph store interface — extendable, but only 1 backend implemented."""

    def add_node(self, node_id: str, attributes: dict | None = None) -> None: ...
    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0, metadata: dict | None = None) -> None: ...
    def get_neighbors(self, node_id: str, edge_type: str | None = None, direction: str = "outgoing") -> list[tuple[str, str, float]]: ...
    def find_paths(self, source: str, target: str, max_depth: int = 5) -> list[list[str]]: ...
    def community_detection(self) -> dict[str, int]: ...
    def get_node_attrs(self, node_id: str) -> dict | None: ...
    def remove_node(self, node_id: str) -> None: ...


class SQLGraphBackend:
    """SQL-backed graph with recursive CTE path finding. Only 1 backend — name matches."""

    def __init__(self, db_path: str = "data/prometheus_v9_graph.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                attributes TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source_type ON graph_edges(source, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_target_type ON graph_edges(target, edge_type);
        """)
        self._conn.commit()

    def add_node(self, node_id: str, attributes: dict | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO graph_nodes (node_id, attributes) VALUES (?, ?)",
            (node_id, json.dumps(attributes or {})),
        )
        self._conn.commit()

    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0, metadata: dict | None = None) -> None:
        self._conn.execute(
            "INSERT INTO graph_edges (source, target, edge_type, weight, metadata) VALUES (?, ?, ?, ?, ?)",
            (source, target, edge_type, weight, json.dumps(metadata or {})),
        )
        self._conn.commit()

    def get_neighbors(self, node_id: str, edge_type: str | None = None, direction: str = "outgoing") -> list[tuple[str, str, float]]:
        if direction == "outgoing":
            query = "SELECT target, edge_type, weight FROM graph_edges WHERE source = ?"
            params: list = [node_id]
        elif direction == "incoming":
            query = "SELECT source, edge_type, weight FROM graph_edges WHERE target = ?"
            params = [node_id]
        else:
            query = "SELECT source, edge_type, weight FROM graph_edges WHERE target = ? UNION SELECT target, edge_type, weight FROM graph_edges WHERE source = ?"
            params = [node_id, node_id]
        if edge_type:
            query += " AND edge_type = ?"
            params.append(edge_type)
        rows = self._conn.execute(query, params).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def find_paths(self, source: str, target: str, max_depth: int = 5) -> list[list[str]]:
        """Recursive CTE path finding."""
        rows = self._conn.execute("""
            WITH RECURSIVE paths(path, last_node, depth) AS (
                VALUES (?, ?, 0)
                UNION ALL
                SELECT path || ',' || ge.target, ge.target, p.depth + 1
                FROM paths p
                JOIN graph_edges ge ON p.last_node = ge.source
                WHERE p.depth < ? AND ge.target NOT IN (SELECT value FROM json_each('[' || replace(path, ',', ',') || ']'))
            )
            SELECT path FROM paths WHERE last_node = ? AND depth > 0
        """, (source, source, max_depth, target)).fetchall()
        return [r[0].split(",") for r in rows]

    def community_detection(self) -> dict[str, int]:
        """Label propagation community detection (2 iterations)."""
        # Initialize each node with its own label
        nodes = self._conn.execute("SELECT node_id FROM graph_nodes").fetchall()
        labels: dict[str, int] = {n[0]: i for i, n in enumerate(nodes)}
        # 2 iterations of label propagation
        for _ in range(2):
            for node_id in labels:
                neighbors = self.get_neighbors(node_id, direction="both")
                if not neighbors:
                    continue
                neighbor_labels = [labels.get(n[0], labels[node_id]) for n in neighbors]
                # Most common label
                from collections import Counter
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                labels[node_id] = most_common
        return labels

    def get_node_attrs(self, node_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT attributes FROM graph_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def remove_node(self, node_id: str) -> None:
        self._conn.execute("DELETE FROM graph_edges WHERE source = ? OR target = ?", (node_id, node_id))
        self._conn.execute("DELETE FROM graph_nodes WHERE node_id = ?", (node_id,))
        self._conn.commit()
