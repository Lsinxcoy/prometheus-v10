"""Prometheus V9 CLI — Typer 8 commands."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

try:
    import typer
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False


def create_cli() -> Any:
    """Create Typer CLI app. Returns None if Typer not installed."""
    if not HAS_TYPER:
        logger.warning("Typer not installed. CLI unavailable.")
        return None

    app = typer.Typer(name="prometheus-v9", help="Prometheus V9 — Self-evolving AI agent memory platform")

    @app.command()
    def init(db_path: str = "data/prometheus_v10.db"):
        """Initialize Prometheus V9 database."""
        from prometheus_v10.store import SQLiteStore
        store = SQLiteStore(db_path)
        typer.echo(f"Initialized store at {db_path} with {store.get_node_count()} nodes")

    @app.command()
    def evolve(steps: int = 10, test_path: str | None = None):
        """Run evolution engine for N steps."""
        from prometheus_v10 import Life
        life = Life()
        for step in range(steps):
            genome, results = life.engine.evolve_single_step(life._current_genome, test_path=test_path)
            life._current_genome = genome
            successes = sum(1 for r in results if r.success)
            typer.echo(f"Step {step+1}: {successes}/12 layers succeeded, fitness={genome.fitness:.3f}")

    @app.command()
    def search(query: str, limit: int = 10, db_path: str = "data/prometheus_v10.db"):
        """Search memory."""
        from prometheus_v10.store import SQLiteStore
        store = SQLiteStore(db_path)
        nodes = store.search_fts(query, limit)
        for node in nodes:
            typer.echo(f"[{node.type.value}] {node.payload.content[:80]}... (importance={node.importance:.2f})")

    @app.command()
    def status(db_path: str = "data/prometheus_v10.db"):
        """Show system status."""
        from prometheus_v10 import Life
        life = Life(db_path=db_path)
        typer.echo(f"Nodes: {life.store.get_node_count()}")
        typer.echo(f"Autonomy: L{life.autonomy.current_level.value}")
        typer.echo(f"Safety violations: {life.safety.violation_count}")

    @app.command()
    def dream(db_path: str = "data/prometheus_v10.db"):
        """Run dream cycle."""
        from prometheus_v10 import Life
        life = Life(db_path=db_path)
        insights = life.dream.generate(life.store)
        integrated = life.dream.integrate(life.store, insights)
        typer.echo(f"Generated {len(insights)} insights, integrated {integrated}")

    @app.command()
    def coral(task: str, outcome: str = "success"):
        """Record CORAL reflection."""
        from prometheus_v10.coral import CORALHeartbeat
        hb = CORALHeartbeat()
        note = hb.reflect(task=task, outcome=outcome)
        typer.echo(f"Reflected: {note.task} → {note.outcome}")

    @app.command()
    def safety_check(code: str, tool: str = "test"):
        """Run safety check on code."""
        from prometheus_v10.manager import SafetyManager
        sm = SafetyManager()
        result = sm.check(tool, code)
        typer.echo(json.dumps(result, indent=2, default=str))

    @app.command()
    def serve(port: int = 8000):
        """Start HTTP API server."""
        from prometheus_v10 import Life
        life = Life()
        app = create_http_app(life)
        if app is None:
            typer.echo("FastAPI not installed. Cannot start server.")
            return
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)

    return app


def create_http_app(life=None):
    from prometheus_v10.http import create_app
    return create_app(life)
