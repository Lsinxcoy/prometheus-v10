"""Prometheus V9 — Self-evolving AI agent memory platform.

T2: 集体>个体 — Only expose `Life` entry point. 5 organs born together.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_v9pro.schema import Genome, MemoryLayer, NodeType
from prometheus_v9pro.config import PrometheusConfig
from prometheus_v9pro.events import EventBus
from prometheus_v9pro.store import SQLiteStore
from prometheus_v9pro.vector import NumpyVecBackend
from prometheus_v9pro.graph import SQLGraphBackend
from prometheus_v9pro.search import HybridSearchEngine
from prometheus_v9pro.lifecycle import (
    WeibullForgetting, EntropyStratifier, EpisodicSemanticMemory,
    ConsolidationManager, DreamCycle,
)
from prometheus_v9pro.engine import UnifiedEvolutionEngine
from prometheus_v9pro.organs import OrganPipeline
from prometheus_v9pro.anti_evo import AntiEvolutionGate
from prometheus_v9pro.coral import CORALHeartbeat
from prometheus_v9pro.breaker import PerToolCircuitBreaker
from prometheus_v9pro.gate import ConfidenceGate
from prometheus_v9pro.validator import ChainValidator
from prometheus_v9pro.manager import SafetyManager
from prometheus_v9pro.autonomy import AutonomyManager
from prometheus_v9pro.trust import TrustManager
from prometheus_v9pro.initiative import InitiativeLayer

logger = logging.getLogger(__name__)


class Life:
    """T2: 集体>个体 — Single entry point. All subsystems born together.

    Importing individual organs directly is discouraged.
    Life is the only public API.
    """

    def __init__(self, db_path: str = "data/prometheus_v9pro.db", config: PrometheusConfig | None = None) -> None:
        self._config = config or PrometheusConfig()

        # L1: Memory
        self._store = SQLiteStore(db_path)
        self._vector = NumpyVecBackend(dim=getattr(self._config, "vector_dim", 128))
        self._graph = SQLGraphBackend(db_path.replace(".db", "_graph.db"))
        self._search = HybridSearchEngine(self._store, self._vector, self._graph)
        self._weibull = WeibullForgetting()
        self._entropy = EntropyStratifier()
        self._episodic = EpisodicSemanticMemory()
        self._consolidation = ConsolidationManager()
        self._dream = DreamCycle()

        # L2: Evolution
        self._engine = UnifiedEvolutionEngine(store=self._store)
        self._pipeline = OrganPipeline(EventBus(), self._config)
        self._anti_evo = AntiEvolutionGate()
        self._coral = CORALHeartbeat()

        # L3: Safety
        self._safety = SafetyManager(
            breaker=PerToolCircuitBreaker(),
            gate=ConfidenceGate(),
            validator=ChainValidator(),
        )

        # L4: Governance
        self._autonomy = AutonomyManager()
        self._trust = TrustManager()
        self._initiative = InitiativeLayer(self._autonomy, self._trust, self._safety)

        # Current genome
        self._current_genome = Genome(
            code="",
            config={"mutation_rate": 0.3, "crossover_rate": 0.7, "elite_ratio": 0.1, "population_size": 20},
            skills=[],
            prompts=["You are a helpful AI assistant."],
            tools=["search", "code_generation"],
        )

    # ── Public Properties ─────────────────────────────────────

    @property
    def store(self) -> SQLiteStore:
        return self._store

    @property
    def vector(self) -> NumpyVecBackend:
        return self._vector

    @property
    def graph(self) -> SQLGraphBackend:
        return self._graph

    @property
    def search(self) -> HybridSearchEngine:
        return self._search

    @property
    def engine(self) -> UnifiedEvolutionEngine:
        return self._engine

    @property
    def pipeline(self) -> OrganPipeline:
        return self._pipeline

    @property
    def anti_evo(self) -> AntiEvolutionGate:
        return self._anti_evo

    @property
    def coral(self) -> CORALHeartbeat:
        return self._coral

    @property
    def dream(self) -> DreamCycle:
        return self._dream

    @property
    def safety(self) -> SafetyManager:
        return self._safety

    @property
    def autonomy(self) -> AutonomyManager:
        return self._autonomy

    @property
    def trust(self) -> TrustManager:
        return self._trust

    @property
    def initiative(self) -> InitiativeLayer:
        return self._initiative

    @property
    def genome(self) -> Genome:
        return self._current_genome

    # ── High-Level Operations ──────────────────────────────────

    def evolve(self, steps: int = 1, test_path: str | None = None) -> list[dict]:
        """Run evolution for N steps."""
        results = []
        for _ in range(steps):
            self._current_genome, step_results = self._engine.evolve_single_step(self._current_genome, test_path)
            results.append({
                "generation": self._current_genome.generation,
                "fitness": self._current_genome.fitness,
                "layers": [{"id": r.layer, "name": r.layer_name, "success": r.success} for r in step_results],
            })
        return results

    def remember(self, content: str, node_type: str = "note", layer: str = "episodic", tags: list[str] | None = None) -> str:
        """Add a memory node."""
        node = NodeType(node_type) if node_type in [e.value for e in NodeType] else NodeType.NOTE
        from prometheus_v9pro.schema import Node, NodePayload, MemoryLayer as ML
        mem_layer = ML(layer) if layer in [e.value for e in ML] else ML.EPISODIC
        n = Node(payload=NodePayload(content=content), type=node, layer=mem_layer, tags=tags or [])
        nid = self._store.add_node(n)
        return nid.hex()

    def recall(self, query: str, limit: int = 10) -> list[dict]:
        """Search memory."""
        nodes = self._store.search_fts(query, limit)
        return [{"id": n.id.hex(), "type": n.type.value, "content": n.payload.content[:200], "importance": n.importance} for n in nodes]

    def dream_cycle(self) -> dict:
        """Run dream cycle."""
        insights = self._dream.generate(self._store)
        integrated = self._dream.integrate(self._store, insights)
        return {"insights": len(insights), "integrated": integrated}

    def reflect(self, task: str, outcome: str, insights: list[str] | None = None, mistakes: list[str] | None = None) -> dict:
        """Record CORAL reflection."""
        note = self._coral.reflect(task, outcome, insights, mistakes)
        return {"task": note.task, "outcome": note.outcome}

    def status(self) -> dict:
        """Full system status."""
        return {
            "version": "9.0.0",
            "nodes": self._store.get_node_count(),
            "generation": self._current_genome.generation,
            "fitness": self._current_genome.fitness,
            "autonomy": f"L{self._autonomy.current_level.value}",
            "safety_violations": self._safety.violation_count,
            "evolution_layers": len(self._engine._layers),
            "dream_stats": self._dream.stats,
            "coral_stats": self._coral.stats,
        }
