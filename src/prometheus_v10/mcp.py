"""Prometheus V9 MCP — FastMCP 10 tools."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


def create_mcp(life=None) -> Any:
    """Create MCP server. Returns None if FastMCP not installed."""
    if not HAS_MCP:
        logger.warning("FastMCP not installed. MCP tools unavailable.")
        return None

    mcp = FastMCP("prometheus-v9")

    @mcp.tool()
    def memory_search(query: str, limit: int = 10) -> str:
        """Search memory for relevant information."""
        if not life or not life.store:
            return "Error: Store not initialized"
        nodes = life.store.search_fts(query, limit)
        if not nodes:
            return "No results found"
        return "\n".join(f"[{n.type.value}] {n.payload.content[:100]}" for n in nodes)

    @mcp.tool()
    def memory_add(content: str, node_type: str = "note", layer: str = "episodic") -> str:
        """Add a memory node."""
        if not life or not life.store:
            return "Error: Store not initialized"
        from prometheus_v10.schema import Node, NodePayload, NodeType, MemoryLayer
        node = Node(payload=NodePayload(content=content), type=NodeType(node_type), layer=MemoryLayer(layer))
        nid = life.store.add_node(node)
        return f"Added node {nid.hex()}"

    @mcp.tool()
    def evolution_step() -> str:
        """Run one evolution step."""
        if not life or not life.engine:
            return "Error: Engine not initialized"
        genome, results = life.engine.evolve_single_step(life._current_genome)
        life._current_genome = genome
        return f"Generation {genome.generation}: {sum(1 for r in results if r.success)}/12 layers succeeded"

    @mcp.tool()
    def evolution_status() -> str:
        """Get evolution engine status."""
        if not life or not life.engine:
            return "Error: Engine not initialized"
        lines = [f"L{l.layer_id} {l.name}: {l._execution_count} executions" for l in life.engine._layers]
        return "\n".join(lines)

    @mcp.tool()
    def safety_check(code: str, tool: str = "test") -> str:
        """Run safety check on code."""
        if not life or not life.safety:
            return "Error: Safety not initialized"
        result = life.safety.check(tool, code)
        return f"Allowed: {result.get('allowed', False)}, Reason: {result.get('reason', 'unknown')}"

    @mcp.tool()
    def autonomy_level() -> str:
        """Get current autonomy level."""
        if not life or not life.autonomy:
            return "Error: Autonomy not initialized"
        return f"Level: L{life.autonomy.current_level.value}, Budget: {life.autonomy.budget_remaining}"

    @mcp.tool()
    def dream_cycle() -> str:
        """Run dream cycle for insight generation."""
        if not life or not life.dream:
            return "Error: Dream not initialized"
        insights = life.dream.generate(life.store)
        integrated = life.dream.integrate(life.store, insights)
        return f"Generated {len(insights)} insights, integrated {integrated}"

    @mcp.tool()
    def coral_reflect(task: str, outcome: str, insights: str = "", mistakes: str = "") -> str:
        """Record CORAL reflection."""
        if not life or not life.coral:
            return "Error: CORAL not initialized"
        note = life.coral.reflect(task=task, outcome=outcome, insights=insights.split(",") if insights else [], mistakes=mistakes.split(",") if mistakes else [])
        return f"Reflected: {note.task} → {note.outcome}"

    @mcp.tool()
    def organ_status() -> str:
        """Get organ pipeline status."""
        if not life or not life.pipeline:
            return "Error: Pipeline not initialized"
        organs = {"taotie": life.pipeline.taotie.is_alive, "nuwa": life.pipeline.nuwa.is_alive, "darwin": life.pipeline.darwin.is_alive, "pool": life.pipeline.pool.is_alive, "guard": life.pipeline.guard.is_alive}
        return "\n".join(f"{k}: {'alive' if v else 'dead'}" for k, v in organs.items())

    @mcp.tool()
    def system_status() -> str:
        """Get full system status."""
        parts = []
        if life and life.store:
            parts.append(f"Nodes: {life.store.get_node_count()}")
        if life and life.autonomy:
            parts.append(f"Autonomy: L{life.autonomy.current_level.value}")
        if life and life.safety:
            parts.append(f"Safety violations: {life.safety.violation_count}")
        if life and life.engine:
            parts.append(f"Evolution layers: {len(life.engine._layers)}")
        return " | ".join(parts) if parts else "System not initialized"

    return mcp
