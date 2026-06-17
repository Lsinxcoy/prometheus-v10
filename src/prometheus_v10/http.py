"""Prometheus V9 HTTP API — FastAPI 15 endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# FastAPI is optional — graceful degradation if not installed
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(life=None) -> Any:
    """Create FastAPI app. Returns None if FastAPI not installed."""
    if not HAS_FASTAPI:
        logger.warning("FastAPI not installed. HTTP API unavailable.")
        return None

    app = FastAPI(title="Prometheus V9", version="9.0.0")

    @app.get("/health")
    async def health():
        return {"status": "alive", "version": "9.0.0", "timestamp": time.time()}

    @app.get("/memory/nodes")
    async def list_nodes(layer: str | None = None, limit: int = 20):
        if not life or not life.store:
            raise HTTPException(500, "Store not initialized")
        from prometheus_v10.schema import MemoryLayer
        if layer:
            nodes = life.store.get_nodes_by_layer(MemoryLayer(layer), limit)
        else:
            nodes = []
        return {"nodes": [{"id": n.id.hex(), "type": n.type.value, "content": n.payload.content[:100], "importance": n.importance} for n in nodes], "count": len(nodes)}

    @app.get("/memory/nodes/{node_id}")
    async def get_node(node_id: str):
        if not life or not life.store:
            raise HTTPException(500, "Store not initialized")
        node = life.store.get_node(bytes.fromhex(node_id))
        if not node:
            raise HTTPException(404, "Node not found")
        return {"id": node.id.hex(), "type": node.type.value, "content": node.payload.content, "importance": node.importance, "layer": node.layer.value}

    @app.post("/memory/nodes")
    async def add_node(body: dict):
        if not life or not life.store:
            raise HTTPException(500, "Store not initialized")
        from prometheus_v10.schema import Node, NodePayload, NodeType, MemoryLayer
        node = Node(payload=NodePayload(content=body.get("content", "")), type=NodeType(body.get("type", "note")), layer=MemoryLayer(body.get("layer", "episodic")))
        nid = life.store.add_node(node)
        return {"id": nid.hex(), "status": "created"}

    @app.get("/memory/search")
    async def search_memory(q: str, limit: int = 10):
        if not life or not life.store:
            raise HTTPException(500, "Store not initialized")
        nodes = life.store.search_fts(q, limit)
        return {"results": [{"id": n.id.hex(), "content": n.payload.content[:100], "score": 0.0} for n in nodes]}

    @app.get("/evolution/status")
    async def evolution_status():
        if not life or not life.engine:
            raise HTTPException(500, "Engine not initialized")
        layers = [{"id": l.layer_id, "name": l.name, "executions": l._execution_count} for l in life.engine._layers]
        return {"layers": layers, "total_layers": len(layers)}

    @app.post("/evolution/step")
    async def evolution_step(body: dict):
        if not life or not life.engine:
            raise HTTPException(500, "Engine not initialized")
        from prometheus_v10.schema import Genome
        genome = life._current_genome or Genome(code="", config={}, skills=[], prompts=[], tools=[])
        genome, results = life.engine.evolve_single_step(genome)
        life._current_genome = genome
        return {"generation": genome.generation, "results": [{"layer": r.layer, "name": r.layer_name, "success": r.success, "delta": r.fitness_delta} for r in results]}

    @app.get("/evolution/genome")
    async def get_genome():
        if not life or not life._current_genome:
            raise HTTPException(404, "No genome")
        g = life._current_genome
        return {"generation": g.generation, "fitness": g.fitness, "skills": g.skills, "config": g.config}

    @app.get("/safety/status")
    async def safety_status():
        if not life or not life.safety:
            raise HTTPException(500, "Safety not initialized")
        return {"breaker_states": life.safety._breaker.get_all_states(), "violation_count": life.safety.violation_count, "audit_hash": life.safety.audit_hash}

    @app.post("/safety/check")
    async def safety_check(body: dict):
        if not life or not life.safety:
            raise HTTPException(500, "Safety not initialized")
        return life.safety.check(body.get("tool", "unknown"), body.get("code", ""), body.get("confidence", 0.8))

    @app.get("/autonomy/level")
    async def autonomy_level():
        if not life or not life.autonomy:
            raise HTTPException(500, "Autonomy not initialized")
        return {"level": life.autonomy.current_level.value, "budget": life.autonomy.budget_remaining}

    @app.post("/autonomy/set")
    async def set_autonomy(body: dict):
        if not life or not life.autonomy:
            raise HTTPException(500, "Autonomy not initialized")
        from prometheus_v10.autonomy import AutonomyLevel
        level = AutonomyLevel(body.get("level", 2))
        life.autonomy.set_level(level)
        return {"level": level.value}

    @app.get("/organs/status")
    async def organs_status():
        if not life or not life.pipeline:
            raise HTTPException(500, "Pipeline not initialized")
        organs = {"taotie": life.pipeline.taotie.is_alive, "nuwa": life.pipeline.nuwa.is_alive, "darwin": life.pipeline.darwin.is_alive, "pool": life.pipeline.pool.is_alive, "guard": life.pipeline.guard.is_alive}
        return {"organs": organs}

    @app.post("/organs/degrade")
    async def degrade_organ(body: dict):
        if not life or not life.pipeline:
            raise HTTPException(500, "Pipeline not initialized")
        life.pipeline.degrade_organ(body.get("organ", ""))
        return {"status": "degraded"}

    @app.get("/coral/stats")
    async def coral_stats():
        if not life or not life.coral:
            raise HTTPException(500, "CORAL not initialized")
        return life.coral.stats

    @app.get("/dream/stats")
    async def dream_stats():
        if not life or not life.dream:
            raise HTTPException(500, "Dream not initialized")
        return life.dream.stats

    return app
