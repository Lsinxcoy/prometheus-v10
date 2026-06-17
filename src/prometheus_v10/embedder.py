"""Prometheus V9PRO Embedder — Semantic embedding with hash fallback.

V8 had core/embedder.py (71 lines) with sentence-transformers support.
V9PRO adds: hash fallback when no model available, embedding cache.
"""
from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class Embedder:
    """Semantic embedding with hash fallback.

    Priority:
    1. sentence-transformers (if available) → real semantic embeddings
    2. Hash-based pseudo-embeddings → deterministic but not semantic
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 dimension: int = 384,
                 cache_size: int = 1000) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model = None
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._embed_count = 0
        self._fallback_count = 0
        self._model_loaded = False

        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._model_loaded = True
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.info("sentence-transformers not available, using hash fallback")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")

    @property
    def using_semantic(self) -> bool:
        return self._model_loaded and self._model is not None

    def embed(self, text: str) -> list[float]:
        """Embed text into a vector. Uses model if available, hash fallback otherwise."""
        self._embed_count += 1

        # Check cache
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        # Generate embedding
        if self._model_loaded and self._model is not None:
            try:
                vector = self._model.encode(text).tolist()
                self._cache[text] = vector
                self._trim_cache()
                return vector
            except Exception as e:
                logger.warning(f"Model embedding failed: {e}, using hash fallback")

        # Hash fallback
        self._fallback_count += 1
        vector = self._hash_embed(text)
        self._cache[text] = vector
        self._trim_cache()
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Uses model batch encoding if available."""
        if self._model_loaded and self._model is not None:
            try:
                vectors = self._model.encode(texts).tolist()
                for text, vector in zip(texts, vectors):
                    self._cache[text] = vector
                    self._embed_count += 1
                self._trim_cache()
                return vectors
            except Exception:
                pass
        return [self.embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic hash-based pseudo-embedding (not semantic)."""
        result: list[float] = []
        for i in range(self._dimension):
            chunk = f"{text}:{i}"
            h = hashlib.sha256(chunk.encode()).hexdigest()
            result.append(int(h[:8], 16) / 0xFFFFFFFF)
        return result

    def _trim_cache(self) -> None:
        """Evict oldest entries when cache exceeds limit."""
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "using_semantic": self.using_semantic,
            "dimension": self._dimension,
            "embed_count": self._embed_count,
            "fallback_count": self._fallback_count,
            "cache_size": len(self._cache),
        }
