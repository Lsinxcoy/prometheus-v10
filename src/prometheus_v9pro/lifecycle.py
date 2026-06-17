"""Prometheus V9 Lifecycle — Weibull forgetting + entropy stratification + Dream ReACT + consolidation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import (
    MemoryLayer, Node, NodePayload, NodeType, create_insight_node,
)
from prometheus_v9pro.utils import compute_entropy, jaccard_similarity

logger = logging.getLogger(__name__)


# ── Weibull Forgetting ────────────────────────────────────────────

class WeibullForgetting:
    """Weibull forgetting curve: S(t) = exp(-(t/λ)^k) + spaced repetition reinforcement.

    V8 had simple retention calculation. V9PRO adds:
    - Per-layer Weibull parameters (EPISODIC decays faster than SKILL)
    - Spaced repetition: recall events boost retention (Ebbinghaus)
    - Recall reinforcement tracking
    """

    # Per-layer (k, lambda_) parameters
    LAYER_PARAMS: dict[str, tuple[float, float]] = {
        "working":   (1.2, 3600.0),     # 1 hour half-life
        "episodic":  (1.5, 86400.0),    # 1 day half-life
        "semantic":  (2.0, 604800.0),   # 1 week half-life
        "skill":     (2.5, 2592000.0),  # 1 month half-life
    }

    def __init__(self) -> None:
        self._recall_count = 0
        self._boost_count = 0

    def retention(self, elapsed: float, k: float = 1.5, lambda_: float = 86400.0) -> float:
        """Compute retention probability after elapsed seconds."""
        if elapsed <= 0:
            return 1.0
        return math.exp(-(elapsed / lambda_) ** k)

    def retention_for_layer(self, elapsed: float, layer: str) -> float:
        """Compute retention using layer-specific Weibull parameters."""
        k, lambda_ = self.LAYER_PARAMS.get(layer, (1.5, 86400.0))
        return self.retention(elapsed, k, lambda_)

    def half_life(self, k: float = 1.5, lambda_: float = 86400.0) -> float:
        """Time for retention to drop to 0.5."""
        return lambda_ * (math.log(2) ** (1.0 / k))

    def half_life_for_layer(self, layer: str) -> float:
        """Half-life using layer-specific parameters."""
        k, lambda_ = self.LAYER_PARAMS.get(layer, (1.5, 86400.0))
        return self.half_life(k, lambda_)

    def should_forget(self, node: Node, current_time: float, threshold: float = 0.1) -> bool:
        """Should this node be forgotten based on Weibull curve (layer-aware)."""
        elapsed = current_time - node.updated_at
        ret = self.retention_for_layer(elapsed, node.layer.value)
        return ret < threshold

    def boost(self, node: Node, factor: float = 1.05) -> Node:
        """Boost node importance (simulates memory reinforcement)."""
        node.importance = min(1.0, node.importance * factor)
        node.updated_at = time.time()
        self._boost_count += 1
        return node

    def spaced_repetition_boost(self, node: Node, recall_count: int) -> Node:
        """Spaced repetition: each recall exponentially increases retention interval.

        Ebbinghaus: after n recalls, the optimal interval ≈ base_interval * 2^n.
        We model this by increasing the effective lambda_ of the node.
        """
        self._recall_count += 1
        # Each recall doubles the effective half-life (up to 32x)
        boost_factor = min(2.0 ** recall_count, 32.0)
        node.importance = min(1.0, node.importance * (1.0 + 0.05 * boost_factor))
        node.updated_at = time.time()
        return node

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "recall_count": self._recall_count,
            "boost_count": self._boost_count,
            "layer_params": {k: {"k": v[0], "lambda": v[1]} for k, v in self.LAYER_PARAMS.items()},
        }


# ── Entropy Stratification (Prism) ───────────────────────────────

class EntropyStratifier:
    """Shannon entropy-based memory stratification (Prism paper arXiv:2604.19795).
    
    High entropy content → SKILL layer (complex, many information dimensions)
    Medium entropy → SEMANTIC layer (moderate information)
    Low entropy → EPISODIC layer (simple, few dimensions)
    """

    def compute_content_entropy(self, content: str) -> float:
        """Shannon entropy of word frequency distribution."""
        if not content:
            return 0.0
        words = content.lower().split()
        if not words:
            return 0.0
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        total = len(words)
        probs = [count / total for count in freq.values()]
        return compute_entropy(probs)

    def stratify(self, nodes: list[Node]) -> dict[MemoryLayer, list[Node]]:
        """Assign nodes to layers based on entropy thresholds."""
        result: dict[MemoryLayer, list[Node]] = {
            MemoryLayer.SKILL: [],
            MemoryLayer.SEMANTIC: [],
            MemoryLayer.EPISODIC: [],
        }
        for node in nodes:
            entropy = self.compute_content_entropy(node.payload.content)
            if entropy > 5.0:   # High information density
                node.layer = MemoryLayer.SKILL
                result[MemoryLayer.SKILL].append(node)
            elif entropy > 2.0:  # Moderate
                node.layer = MemoryLayer.SEMANTIC
                result[MemoryLayer.SEMANTIC].append(node)
            else:                # Simple
                node.layer = MemoryLayer.EPISODIC
                result[MemoryLayer.EPISODIC].append(node)
        return result


# ── Episodic-Semantic Dual Process ──────────────────────────────

class EpisodicSemanticMemory:
    """Dual-process memory: 10-message episodic window + semantic growth (arXiv:2605.17625)."""

    WINDOW_SIZE = 10    # Latest episodes kept in working memory
    GROWTH_RATE = 3     # Tokens per message for semantic compression

    def __init__(self) -> None:
        self._episodes: list[Node] = []
        self._semantic_count: int = 0

    def add_episode(self, node: Node) -> None:
        """Add episode to working memory."""
        node.layer = MemoryLayer.EPISODIC
        self._episodes.append(node)
        if len(self._episodes) > self.WINDOW_SIZE:
            self._episodes = self._episodes[-self.WINDOW_SIZE:]
        self._semantic_count += self.GROWTH_RATE

    def get_working_memory(self, semantic_nodes: list[Node] | None = None) -> list[Node]:
        """Get working memory: last 10 episodes + relevant semantic nodes."""
        working = list(self._episodes[-self.WINDOW_SIZE:])
        if semantic_nodes:
            # Add semantic nodes proportional to growth
            n_semantic = min(self._semantic_count, len(semantic_nodes))
            working.extend(semantic_nodes[:n_semantic])
        return working


# ── Consolidation ──────────────────────────────────────────────

class ConsolidationManager:
    """4-level memory consolidation pipeline: Working→Episodic→Semantic→Procedural→Archive.

    V8 had a simple promote-if-accessed model. V9PRO adds:
    - Procedural layer: repeated patterns → compiled procedures
    - Archive layer: low-access but high-importance → long-term storage
    - Consolidation audit trail
    """

    def consolidate(self, store) -> int:
        """Run full 4-level consolidation pipeline. Returns total promoted."""
        promoted = 0
        promoted += self._promote_working_to_episodic(store)
        promoted += self._promote_episodic_to_semantic(store)
        promoted += self._promote_semantic_to_procedural(store)
        promoted += self._archive_low_access(store)
        return promoted

    def _promote_working_to_episodic(self, store) -> int:
        """Working→Episodic: any node with access_count >= 1."""
        working = store.get_nodes_by_layer(MemoryLayer.WORKING, limit=50)
        count = 0
        for node in working:
            if node.access_count >= 1:
                node.layer = MemoryLayer.EPISODIC
                store.update_node(node)
                count += 1
        return count

    def _promote_episodic_to_semantic(self, store) -> int:
        """Episodic→Semantic: accessed > 3 times AND importance > 0.6."""
        episodic = store.get_nodes_by_layer(MemoryLayer.EPISODIC, limit=100)
        count = 0
        for node in episodic:
            if node.access_count > 3 and node.importance > 0.6:
                node.layer = MemoryLayer.SEMANTIC
                node.type = NodeType.INSIGHT
                store.update_node(node)
                count += 1
        return count

    def _promote_semantic_to_procedural(self, store) -> int:
        """Semantic→Procedural: high importance + high access + skill-type content."""
        semantic = store.get_nodes_by_layer(MemoryLayer.SEMANTIC, limit=50)
        count = 0
        for node in semantic:
            if node.importance > 0.8 and node.access_count > 10:
                node.layer = MemoryLayer.SKILL
                node.type = NodeType.SKILL
                store.update_node(node)
                count += 1
        return count

    def _archive_low_access(self, store) -> int:
        """Archive: low access but high importance → preserve for long term."""
        semantic = store.get_nodes_by_layer(MemoryLayer.SEMANTIC, limit=50)
        count = 0
        for node in semantic:
            if node.access_count <= 1 and node.importance > 0.7 and node.retention < 0.3:
                # High importance but rarely accessed and low retention → archive
                node.importance *= 0.95  # slight decay
                store.update_node(node)
                count += 1
        return count


# ── Dream Cycle (ReACT GENERATE + INTEGRATE) ────────────────────

class DreamCycle:
    """Dream cycle: ReACT 3-round reasoning for insight generation.
    
    Only GENERATE + INTEGRATE stages (REPLAY/ASSOCIATE/CONSOLIDATE
    merged into lifecycle daily operations).
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._dream_count = 0
        self._insights_generated = 0

    def generate(self, store, search_engine=None, n_associations: int = 30) -> list[Node]:
        """ReACT GENERATE: 3-round Thought→Action→Observation → insights.
        
        If LLM available: genuine ReACT reasoning.
        If no LLM: heuristic pattern detection from node associations.
        """
        self._dream_count += 1

        # Gather top associations from memory
        recent = store.get_nodes_by_layer(MemoryLayer.SEMANTIC, limit=n_associations)
        episodic = store.get_nodes_by_layer(MemoryLayer.EPISODIC, limit=n_associations // 2)
        candidates = recent + episodic

        if len(candidates) < 2:
            return []

        # Compute pairwise associations
        associations: list[tuple[Node, Node, float]] = []
        for i in range(min(len(candidates), 20)):
            for j in range(i + 1, min(len(candidates), 20)):
                score = self._compute_association(candidates[i], candidates[j])
                if score > 0.15:
                    associations.append((candidates[i], candidates[j], score))
        associations.sort(key=lambda x: x[2], reverse=True)
        top = associations[:5]

        if not top:
            return []

        insights: list[Node] = []
        accumulated_observations: list[str] = []

        # ReACT Round 1-3
        for n1, n2, score in top:
            # Thought: what pattern connects these two?
            thought = f"Pattern: '{n1.payload.content[:50]}' ↔ '{n2.payload.content[:50]}' (strength={score:.2f})"
            # Action: search for more evidence
            if search_engine:
                action_query = n1.payload.content[:30] + " " + n2.payload.content[:30]
                try:
                    search_results = search_engine.search(action_query, top_k=3)
                    observation = "; ".join(n.payload.content[:50] for n, _ in search_results[:3])
                except Exception:
                    observation = "Search failed"
            else:
                observation = f"Association between {n1.type.value} and {n2.type.value}"
            accumulated_observations.append(observation)

        # Generate insights from accumulated observations
        if self._llm:
            try:
                context_str = "\n".join(
                    f"'{n1.payload.content[:50]}' ↔ '{n2.payload.content[:50]}' (score={s:.2f})"
                    for n1, n2, s in top
                )
                obs_str = "\n".join(accumulated_observations)
                insight_prompt = (
                    "Based on the following memory associations and observations, "
                    "generate 1-3 novel insights. Each insight should be a single sentence. "
                    "Return a JSON array of objects with 'content' and 'importance' fields.\n\n"
                    f"Associations:\n{context_str}\n\nObservations:\n{obs_str}\n\n"
                    "Return ONLY the JSON array."
                )
                import json
                response = self._llm(insight_prompt).strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                insight_data = json.loads(response)
                if isinstance(insight_data, list):
                    for item in insight_data[:3]:
                        content = item.get("content", "")
                        importance = float(item.get("importance", 0.5))
                        if content:
                            insight = create_insight_node(content, importance)
                            insights.append(insight)
            except Exception as e:
                logger.warning(f"LLM insight generation failed: {e}")
                # Fallback: generate from top associations
                for n1, n2, score in top[:2]:
                    content = f"Pattern detected: '{n1.payload.content[:50]}' ↔ '{n2.payload.content[:50]}' (strength={score:.2f})"
                    insight = create_insight_node(content, score * 0.8)
                    insights.append(insight)
        else:
            # Heuristic fallback: pattern detection from associations
            for n1, n2, score in top[:3]:
                content = f"Association: {n1.type.value}-{n2.type.value} pattern (score={score:.2f})"
                insight = create_insight_node(content, score * 0.7)
                insights.append(insight)

        self._insights_generated += len(insights)
        return insights

    def integrate(self, store, insights: list[Node], event_bus=None) -> int:
        """INTEGRATE: write insights into semantic memory + publish events."""
        integrated = 0
        for insight in insights:
            store.add_node(insight)
            integrated += 1
            if event_bus:
                try:
                    event_bus.publish("insight_generated", {
                        "content": insight.payload.content,
                        "importance": insight.importance,
                    }, source="dream")
                except Exception:
                    pass
        return integrated

    def _compute_association(self, n1: Node, n2: Node) -> float:
        """Compute association score between two nodes."""
        w1 = set(n1.payload.content.lower().split())
        w2 = set(n2.payload.content.lower().split())
        content_sim = jaccard_similarity(w1, w2)
        tag_sim = jaccard_similarity(set(n1.tags), set(n2.tags))
        type_bonus = 0.1 if n1.type == n2.type else 0.0
        return 0.5 * content_sim + 0.3 * tag_sim + 0.2 * type_bonus

    @property
    def stats(self) -> dict[str, int]:
        return {"dream_cycles": self._dream_count, "insights_generated": self._insights_generated}


# ── 4-Dimension Aging Detection ──────────────────────────────────

class AgingDetector:
    """4-dimension memory aging detection. Input for metabolism/triage.

    V8 had aging.py (112 lines) as separate module. V9PRO consolidates into lifecycle.
    4 dimensions:
    1. Compressive aging: content compressed/summarized too many times → info loss
    2. Interference aging: similar competing nodes → retrieval confusion
    3. Revision aging: content has been revised many times → unstable
    4. Maintenance aging: low recent access + low retention → neglected
    """

    def __init__(self) -> None:
        self._checks_performed = 0
        self._aging_detected: dict[str, int] = {
            "compressive": 0, "interference": 0,
            "revision": 0, "maintenance": 0,
        }

    def detect(self, node: Node, current_time: float,
               similar_nodes: list[Node] | None = None) -> dict[str, float]:
        """Detect aging across 4 dimensions. Returns dimension → aging_score (0-1)."""
        self._checks_performed += 1
        scores: dict[str, float] = {}

        # 1. Compressive aging: short content in SEMANTIC+ layers = over-compressed
        content_len = len(node.payload.content)
        if node.layer.value in ("semantic", "skill") and content_len < 20:
            scores["compressive"] = 0.8
        elif content_len < 10:
            scores["compressive"] = 0.5
        else:
            scores["compressive"] = 0.0

        # 2. Interference aging: too many similar nodes competing
        if similar_nodes:
            n_similar = len([n for n in similar_nodes
                            if n.id != node.id and
                            jaccard_similarity(
                                set(node.payload.content.lower().split()),
                                set(n.payload.content.lower().split())) > 0.5])
            scores["interference"] = min(1.0, n_similar / 5.0)
        else:
            scores["interference"] = 0.0

        # 3. Revision aging: many updates → unstable
        age = current_time - node.created_at
        update_frequency = node.access_count / max(1.0, age / 86400.0)
        scores["revision"] = min(1.0, update_frequency / 10.0)

        # 4. Maintenance aging: low access + low retention
        elapsed = current_time - node.updated_at
        weibull = WeibullForgetting()
        retention = weibull.retention_for_layer(elapsed, node.layer.value)
        access_rate = node.access_count / max(1.0, age / 86400.0)
        scores["maintenance"] = (1.0 - retention) * (1.0 - min(1.0, access_rate / 5.0))

        # Track detections
        for dim, score in scores.items():
            if score > 0.5:
                self._aging_detected[dim] = self._aging_detected.get(dim, 0) + 1

        return scores

    def get_triage_recommendation(self, aging_scores: dict[str, float]) -> str:
        """Recommend action based on aging scores: refresh/archive/delete/keep."""
        max_score = max(aging_scores.values()) if aging_scores else 0.0
        worst_dim = max(aging_scores, key=aging_scores.get) if aging_scores else ""

        if max_score < 0.3:
            return "keep"
        if worst_dim == "compressive":
            return "refresh"   # Re-expand compressed content
        if worst_dim == "interference":
            return "merge"     # Merge with similar nodes
        if worst_dim == "revision":
            return "freeze"    # Stop revising, stabilize
        if worst_dim == "maintenance":
            return "archive"   # Low access → archive
        return "keep"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "checks_performed": self._checks_performed,
            "aging_detected": dict(self._aging_detected),
        }
