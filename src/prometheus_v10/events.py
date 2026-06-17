"""Prometheus V9 Events — 25 event types, EventBus with pub/sub."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(Enum):
    # Organ lifecycle
    TAOITE_DONE = "taotie_done"
    NUWA_DONE = "nuwa_done"
    DARWIN_DONE = "darwin_done"
    POOL_DONE = "pool_done"
    GUARD_DONE = "guard_done"
    # Evolution
    EVOLUTION_STEP = "evolution_step"
    EVOLUTION_COMPLETE = "evolution_complete"
    STAGNATION_DETECTED = "stagnation_detected"
    FITNESS_REPORT = "fitness_report"
    # Memory
    NODE_ADDED = "node_added"
    NODE_UPDATED = "node_updated"
    NODE_DELETED = "node_deleted"
    CONSOLIDATION = "consolidation"
    DREAM_CYCLE = "dream_cycle"
    INSIGHT_GENERATED = "insight_generated"
    # Safety
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    SAFETY_VIOLATION = "safety_violation"
    OEP_SUSPECT = "oep_suspect"
    # Governance
    TRUST_CHANGE = "trust_change"
    AUTONOMY_CHANGE = "autonomy_change"
    INITIATIVE_PROPOSED = "initiative_proposed"
    INITIATIVE_APPROVED = "initiative_approved"
    # System
    ORGAN_DEGRADED = "organ_degraded"
    SYSTEM_ERROR = "system_error"


@dataclass
class Event:
    type: EventType = EventType.SYSTEM_ERROR
    channel: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""


class EventBus:
    """Typed event bus with pub/sub, history, and metrics."""

    def __init__(self, history_size: int = 1000) -> None:
        self._subscribers: dict[str, list[Callable]] = {}
        self._history: deque[Event] = deque(maxlen=history_size)
        self._publish_count: dict[str, int] = {}

    def publish(self, channel: str, payload: dict[str, Any] | None = None, source: str = "") -> None:
        """Publish event to channel, notify all subscribers."""
        event = Event(
            channel=channel,
            payload=payload or {},
            source=source,
        )
        self._history.append(event)
        self._publish_count[channel] = self._publish_count.get(channel, 0) + 1
        for callback in self._subscribers.get(channel, []):
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Subscriber error on {channel}: {e}")

    def subscribe(self, channel: str, callback: Callable[[Event], None]) -> None:
        """Subscribe callback to channel."""
        self._subscribers.setdefault(channel, []).append(callback)

    def unsubscribe(self, channel: str, callback: Callable[[Event], None]) -> bool:
        """Remove callback from channel."""
        subs = self._subscribers.get(channel, [])
        if callback in subs:
            subs.remove(callback)
            return True
        return False

    def get_history(self, channel: str | None = None, limit: int = 50) -> list[Event]:
        """Get event history, optionally filtered by channel."""
        events = list(self._history)
        if channel:
            events = [e for e in events if e.channel == channel]
        return events[-limit:]

    def get_metrics(self) -> dict[str, Any]:
        """Get bus metrics: publish counts, subscriber counts."""
        return {
            "publish_counts": dict(self._publish_count),
            "subscriber_counts": {ch: len(subs) for ch, subs in self._subscribers.items()},
            "total_events": len(self._history),
        }


# ── Agent Registry (V8 communication/registry.py consolidated) ────

@dataclass
class AgentDescriptor:
    """Agent registration record with heartbeat monitoring."""
    agent_id: str = ""
    capabilities: list[str] = field(default_factory=list)
    status: str = "active"  # active | idle | zombie | dead
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """Agent registration + heartbeat monitoring + zombie reaping.

    V8 had communication/registry.py (141 lines). V9PRO consolidates into events.py
    since registry is fundamentally about event-driven agent lifecycle.
    """

    ZOMBIE_TIMEOUT = 300.0  # 5 minutes without heartbeat → zombie
    DEAD_TIMEOUT = 3600.0   # 1 hour → dead

    def __init__(self) -> None:
        self._agents: dict[str, AgentDescriptor] = {}
        self._reap_count = 0

    def register(self, agent_id: str, capabilities: list[str] | None = None,
                 metadata: dict[str, Any] | None = None) -> AgentDescriptor:
        """Register a new agent."""
        desc = AgentDescriptor(
            agent_id=agent_id,
            capabilities=capabilities or [],
            metadata=metadata or {},
        )
        self._agents[agent_id] = desc
        logger.info(f"Agent registered: {agent_id} with {len(desc.capabilities)} capabilities")
        return desc

    def heartbeat(self, agent_id: str) -> bool:
        """Record heartbeat from agent. Returns False if agent not registered."""
        if agent_id not in self._agents:
            return False
        self._agents[agent_id].last_heartbeat = time.time()
        self._agents[agent_id].status = "active"
        return True

    def find_by_capability(self, capability: str) -> list[AgentDescriptor]:
        """Find active agents with a specific capability."""
        return [a for a in self._agents.values()
                if capability in a.capabilities and a.status == "active"]

    def reap_zombies(self) -> list[str]:
        """Mark agents as zombie/dead based on heartbeat timeout. Returns reaped IDs."""
        now = time.time()
        reaped: list[str] = []
        for agent_id, desc in self._agents.items():
            elapsed = now - desc.last_heartbeat
            if elapsed > self.DEAD_TIMEOUT and desc.status != "dead":
                desc.status = "dead"
                reaped.append(agent_id)
                logger.warning(f"Agent {agent_id} marked DEAD (no heartbeat for {elapsed:.0f}s)")
            elif elapsed > self.ZOMBIE_TIMEOUT and desc.status == "active":
                desc.status = "zombie"
                reaped.append(agent_id)
                logger.warning(f"Agent {agent_id} marked ZOMBIE (no heartbeat for {elapsed:.0f}s)")
        self._reap_count += len(reaped)
        return reaped

    def unregister(self, agent_id: str) -> bool:
        """Remove agent from registry."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    @property
    def stats(self) -> dict[str, Any]:
        status_counts: dict[str, int] = {}
        for desc in self._agents.values():
            status_counts[desc.status] = status_counts.get(desc.status, 0) + 1
        return {
            "total_agents": len(self._agents),
            "status_counts": status_counts,
            "zombies_reaped": self._reap_count,
        }
