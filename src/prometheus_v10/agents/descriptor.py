"""Prometheus V9 Agent Descriptor — Agent metadata, capabilities, lifecycle management.

Multi-agent architecture foundation: who each agent is, what it can do, 
how it"s doing. Without this, L10 Collaboration is an empty shell.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent roles in the Prometheus ecosystem."""
    CEO = "ceo"           # Strategic decisions, uses pro model
    WORKER = "worker"     # Task execution, standard model
    EXPLORER = "explorer"  # Knowledge exploration and gap filling
    JUDGE = "judge"       # Fitness evaluation and quality assessment
    GUARDIAN = "guardian"  # Safety monitoring and enforcement


class AgentState(str, Enum):
    """Agent runtime state."""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    DEAD = "dead"
    OFFLINE = "offline"


@dataclass
class AgentDescriptor:
    """Complete agent descriptor with identity, capabilities, and runtime state."""
    id: str = ""
    name: str = ""
    role: AgentRole = AgentRole.WORKER
    model: str = ""
    model_tier: str = "standard"  # pro/standard/light
    capabilities: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3
    priority: int = 5
    state: AgentState = AgentState.IDLE
    current_tasks: list[str] = field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"agent_{uuid.uuid4().hex[:8]}"

    @property
    def is_available(self) -> bool:
        return (
            self.state in (AgentState.IDLE, AgentState.WAITING)
            and len(self.current_tasks) < self.max_concurrent_tasks
        )

    @property
    def success_rate(self) -> float:
        total = self.completed_tasks + self.failed_tasks
        return self.completed_tasks / max(1, total)

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.capabilities or "*" in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "role": self.role.value,
            "model": self.model, "model_tier": self.model_tier,
            "capabilities": self.capabilities, "state": self.state.value,
            "priority": self.priority, "is_available": self.is_available,
            "success_rate": round(self.success_rate, 3),
            "current_tasks": len(self.current_tasks),
        }


class AgentRegistry:
    """Registry of all known agents with capability-based lookup."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDescriptor] = {}
        self._capability_index: dict[str, set[str]] = defaultdict(set)
        self._role_index: dict[AgentRole, set[str]] = defaultdict(set)

    def register(self, agent: AgentDescriptor) -> None:
        self._agents[agent.id] = agent
        for cap in agent.capabilities:
            self._capability_index[cap].add(agent.id)
            self._capability_index["*"].add(agent.id)  # wildcard
        self._role_index[agent.role].add(agent.id)
        logger.info(f"Registered agent {agent.name} ({agent.role.value})")

    def unregister(self, agent_id: str) -> None:
        agent = self._agents.pop(agent_id, None)
        if agent:
            for cap in agent.capabilities:
                self._capability_index[cap].discard(agent_id)
            self._role_index[agent.role].discard(agent_id)

    def get(self, agent_id: str) -> AgentDescriptor | None:
        return self._agents.get(agent_id)

    def find_by_capability(self, capability: str) -> list[AgentDescriptor]:
        ids = self._capability_index.get(capability, set())
        return [a for a in (self._agents[i] for i in ids) if a.is_available]

    def find_by_role(self, role: AgentRole) -> list[AgentDescriptor]:
        ids = self._role_index.get(role, set())
        return [self._agents[i] for i in ids if i in self._agents]

    def find_available(self, capability: str | None = None) -> list[AgentDescriptor]:
        agents = list(self._agents.values())
        if capability:
            agents = [a for a in agents if a.can_handle(capability)]
        return [a for a in agents if a.is_available]

    def update_heartbeat(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent:
            agent.last_heartbeat = time.time()

    def mark_dead(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent:
            agent.state = AgentState.DEAD
            logger.warning(f"Agent {agent.name} marked DEAD")

    @property
    def all_agents(self) -> list[AgentDescriptor]:
        return list(self._agents.values())

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.state != AgentState.DEAD)
