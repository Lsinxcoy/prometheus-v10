"""Prometheus V9 Vector — NumpyVecBackend (1 backend, deterministic rowid)."""

from __future__ import annotations

import logging

import numpy as np

from prometheus_v10.utils import deterministic_rowid

logger = logging.getLogger(__name__)


class NumpyVecBackend:
    """NumPy-backed vector search. Only 1 backend — name matches implementation."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self._vectors: np.ndarray | None = None  # shape: (N, dim)
        self._ids: list[int] = []      # rowid -> index mapping
        self._id_to_idx: dict[int, int] = {}

    def add(self, node_id: bytes, embedding: list[float]) -> None:
        """Add vector for a node. Deterministic rowid ensures cross-process consistency."""
        rowid = deterministic_rowid(node_id)
        vec = np.array(embedding, dtype=np.float32)
        if vec.shape[0] != self._dim:
            raise ValueError(f"Embedding dimension {vec.shape[0]} != expected {self._dim}")
        # Normalize for cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        if self._vectors is None:
            self._vectors = vec.reshape(1, -1)
        else:
            self._vectors = np.vstack([self._vectors, vec])
        idx = len(self._ids)
        self._ids.append(rowid)
        self._id_to_idx[rowid] = idx

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[int, float]]:
        """Cosine similarity search. Returns (rowid, similarity_score) pairs."""
        if self._vectors is None or len(self._ids) == 0:
            return []
        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        # Cosine similarity (vectors already normalized)
        scores = np.dot(self._vectors, query)
        top_indices = np.argsort(scores)[-min(top_k, len(scores)):][::-1]
        return [(self._ids[i], float(scores[i])) for i in top_indices]

    def delete(self, node_id: bytes) -> bool:
        """Delete vector by node_id."""
        rowid = deterministic_rowid(node_id)
        if rowid not in self._id_to_idx:
            return False
        idx = self._id_to_idx[rowid]
        # Remove from arrays
        mask = np.ones(len(self._ids), dtype=bool)
        mask[idx] = False
        self._vectors = self._vectors[mask]
        self._ids = [self._ids[i] for i in range(len(self._ids)) if i != idx]
        self._id_to_idx = {rid: new_idx for new_idx, rid in enumerate(self._ids)}
        return True

    def get(self, node_id: bytes) -> list[float] | None:
        """Get embedding by node_id."""
        rowid = deterministic_rowid(node_id)
        if rowid not in self._id_to_idx:
            return None
        idx = self._id_to_idx[rowid]
        return self._vectors[idx].tolist()

    def count(self) -> int:
        return len(self._ids)
