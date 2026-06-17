"""Prometheus V9 Utilities — Deterministic IDs, entropy, similarity, fingerprinting."""

from __future__ import annotations

import hashlib
import math
import struct
import time
import uuid


def generate_uuidv7() -> bytes:
    """Generate UUIDv7-like bytes: 48-bit timestamp + 74-bit random."""
    now_ms = int(time.time() * 1000)
    rand_bytes = uuid.uuid4().bytes[6:]  # 10 random bytes
    # Pack: 6 bytes timestamp + 10 bytes random
    ts_bytes = struct.pack(">Q", now_ms)[2:]  # 6 bytes from 8-byte big-endian
    return ts_bytes + rand_bytes


def deterministic_rowid(data: bytes) -> int:
    """Deterministic integer ID from bytes. Fixes V8 hash(id) cross-process bug.
    
    Uses first 8 bytes interpreted as unsigned 64-bit integer.
    Same bytes → same int, guaranteed across processes and restarts.
    """
    padded = data[:8].ljust(8, b"\x00")
    return struct.unpack(">Q", padded)[0] % (2**53)  # Safe for SQLite INTEGER


def compute_entropy(values: list[float]) -> float:
    """Shannon entropy: H = -sum(p * log2(p)) for p in values.
    
    Values are treated as a probability distribution (normalized if needed).
    """
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity: |intersection| / |union|."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


uuidv7 = generate_uuidv7  # Alias for convenience
def compute_fingerprint(code: str) -> str:
    """SHA-256 fingerprint of code, first 16 hex chars."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]
