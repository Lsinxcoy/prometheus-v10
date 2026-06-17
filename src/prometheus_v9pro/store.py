"""Prometheus V9 Store — SQLite WAL + FTS5 + version control + DualTrack."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from prometheus_v9pro.schema import (
    Edge, EdgeType, MemoryLayer, MemoryScope, Node, NodePayload, NodeType, TrustLevel,
)
from prometheus_v9pro.utils import deterministic_rowid

logger = logging.getLogger(__name__)


class SQLiteStore:
    """SQLite-backed store with WAL, FTS5, versioning, and DualTrack isolation."""

    def __init__(self, db_path: str = "data/prometheus_v9pro.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        self._dual_track = DualTrackMemory(self)

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id BLOB PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                layer TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                trust_level TEXT NOT NULL DEFAULT 'pending',
                tags TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                action_hook TEXT,
                fingerprint TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB
            );
            CREATE TABLE IF NOT EXISTS edges (
                id BLOB PRIMARY KEY,
                source BLOB NOT NULL,
                target BLOB NOT NULL,
                type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE INDEX IF NOT EXISTS idx_nodes_layer ON nodes(layer);
            CREATE INDEX IF NOT EXISTS idx_nodes_scope ON nodes(scope);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_content USING fts5(
                node_id, content, tags,
                content='nodes', content_rowid='rowid'
            );
            CREATE TABLE IF NOT EXISTS node_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id BLOB NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_versions_node ON node_versions(node_id);
            CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
                INSERT INTO fts_content(rowid, node_id, content, tags)
                VALUES (new.rowid, hex(new.id), new.content, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
                INSERT INTO fts_content(fts_content, rowid, node_id, content, tags)
                VALUES ('delete', old.rowid, hex(old.id), old.content, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
                INSERT INTO fts_content(fts_content, rowid, node_id, content, tags)
                VALUES ('delete', old.rowid, hex(old.id), old.content, old.tags);
                INSERT INTO fts_content(rowid, node_id, content, tags)
                VALUES (new.rowid, hex(new.id), new.content, new.tags);
            END;
        """)
        self._conn.commit()

    # ── Node CRUD ──────────────────────────────────────────────

    def add_node(self, node: Node) -> bytes:
        rowid = deterministic_rowid(node.id)
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, type, content, layer, scope, trust_level, tags, importance,
                access_count, created_at, updated_at, action_hook, fingerprint, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node.id, node.type.value, node.payload.content, node.layer.value,
             node.scope.value, node.trust_level.value, json.dumps(node.tags),
             node.importance, node.access_count, node.created_at, node.updated_at,
             node.action_hook, node.fingerprint, json.dumps(node.payload.metadata)),
        )
        self._conn.commit()
        return node.id

    def get_node(self, node_id: bytes) -> Node | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_node(row)

    def update_node(self, node: Node) -> bool:
        existing = self.get_node(node.id)
        if not existing:
            return False
        # Create version snapshot
        self._conn.execute(
            "INSERT INTO node_versions (node_id, content, importance, timestamp) VALUES (?, ?, ?, ?)",
            (node.id, existing.payload.content, existing.importance, time.time()),
        )
        node.updated_at = time.time()
        self._conn.execute(
            """UPDATE nodes SET type=?, content=?, layer=?, scope=?, trust_level=?,
               tags=?, importance=?, access_count=?, updated_at=?,
               action_hook=?, fingerprint=?, metadata=? WHERE id=?""",
            (node.type.value, node.payload.content, node.layer.value, node.scope.value,
             node.trust_level.value, json.dumps(node.tags), node.importance,
             node.access_count, node.updated_at, node.action_hook, node.fingerprint,
             json.dumps(node.payload.metadata), node.id),
        )
        self._conn.commit()
        return True

    def delete_node(self, node_id: bytes) -> bool:
        cursor = self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_nodes_by_layer(self, layer: MemoryLayer, limit: int = 100) -> list[Node]:
        rows = self._conn.execute(
            "SELECT * FROM nodes WHERE layer = ? ORDER BY importance DESC LIMIT ?",
            (layer.value, limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_node_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0] if row else 0

    # ── FTS5 Search ────────────────────────────────────────────

    def search_fts(self, query: str, limit: int = 10) -> list[Node]:
        rows = self._conn.execute(
            """SELECT n.* FROM nodes n
               JOIN fts_content f ON f.rowid = n.rowid
               WHERE fts_content MATCH ?
               ORDER BY rank LIMIT ?""",
            (query, limit),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    # ── Edge CRUD ──────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> bytes:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (id, source, target, type, weight, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (edge.id, edge.source, edge.target, edge.type.value, edge.weight, json.dumps(edge.metadata)),
        )
        self._conn.commit()
        return edge.id

    def get_edges(self, node_id: bytes) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source = ? OR target = ?",
            (node_id, node_id),
        ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ── Row Mappers ────────────────────────────────────────────

    def _row_to_node(self, row: tuple) -> Node:
        return Node(
            id=row[0],
            type=NodeType(row[1]),
            payload=NodePayload(content=row[2], metadata=json.loads(row[13])),
            layer=MemoryLayer(row[3]),
            scope=MemoryScope(row[4]),
            trust_level=TrustLevel(row[5]),
            tags=json.loads(row[6]),
            importance=row[7],
            access_count=row[8],
            created_at=row[9],
            updated_at=row[10],
            action_hook=row[11],
            fingerprint=row[12],
        )

    @staticmethod
    def _row_to_edge(row: tuple) -> Edge:
        return Edge(
            id=row[0], source=row[1], target=row[2],
            type=EdgeType(row[3]), weight=row[4],
            metadata=json.loads(row[5]),
        )

    # ── DualTrack Accessor ─────────────────────────────────────

    @property
    def dual_track(self) -> DualTrackMemory:
        return self._dual_track


class DualTrackMemory:
    """Agent/User memory isolation with trust-gated bridging (T2 collective)."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def add_agent_memory(self, node: Node) -> bytes:
        node.scope = MemoryScope.AGENT
        return self._store.add_node(node)

    def add_user_memory(self, node: Node) -> bytes:
        node.scope = MemoryScope.USER
        return self._store.add_node(node)

    def get_agent_memories(self, query: str, limit: int = 10) -> list[Node]:
        nodes = self._store.search_fts(query, limit * 3)
        return [n for n in nodes if n.scope in (MemoryScope.AGENT, MemoryScope.GLOBAL)][:limit]

    def get_user_memories(self, query: str, limit: int = 10) -> list[Node]:
        nodes = self._store.search_fts(query, limit * 3)
        return [n for n in nodes if n.scope in (MemoryScope.USER, MemoryScope.GLOBAL)][:limit]

    def bridge_to_user(self, node_id: bytes) -> bool:
        """Bridge agent knowledge to user track. Requires HIGH_SIGNAL or VERIFIED trust."""
        node = self._store.get_node(node_id)
        if not node or node.trust_level not in (TrustLevel.HIGH_SIGNAL, TrustLevel.VERIFIED):
            return False
        node.scope = MemoryScope.USER
        self._store.update_node(node)
        return True
