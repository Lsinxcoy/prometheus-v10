"""Prometheus V9 EVO Bus — Agent communication bus with graceful degradation.

Redis Stream based inter-agent communication with InMemory fallback.
This is the backbone of multi-agent coordination: organs, workers, and 
external agents all communicate through this bus.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@dataclass
class BusMessage:
    """A message on the bus."""
    id: str = ""
    topic: str = ""
    sender: str = ""
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: str = ""
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"msg_{uuid.uuid4().hex[:8]}"

    def serialize(self) -> dict[str, str]:
        return {
            "id": self.id, "topic": self.topic, "sender": self.sender,
            "payload": json.dumps(self.payload, ensure_ascii=False),
            "timestamp": str(self.timestamp),
            "reply_to": self.reply_to, "priority": str(self.priority),
        }

    @classmethod
    def deserialize(cls, data: dict[str, str]) -> BusMessage:
        return cls(
            id=data.get("id", ""), topic=data.get("topic", ""),
            sender=data.get("sender", ""),
            payload=json.loads(data.get("payload", "{}")),
            timestamp=float(data.get("timestamp", 0)),
            reply_to=data.get("reply_to", ""),
            priority=int(data.get("priority", 0)),
        )


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    id: str = ""
    name: str = ""
    role: str = "worker"
    model: str = ""
    capabilities: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "online"


class InMemoryBus:
    """In-memory fallback message bus when Redis is unavailable."""

    def __init__(self, max_history: int = 1000) -> None:
        self._subscribers: dict[str, list[Callable[[BusMessage], None]]] = defaultdict(list)
        self._history: dict[str, deque[BusMessage]] = defaultdict(lambda: deque(maxlen=max_history))
        self._agents: dict[str, AgentInfo] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, message: BusMessage) -> int:
        """Publish message to topic. Returns subscriber count."""
        self._history[topic].append(message)
        count = 0
        for callback in self._subscribers.get(topic, []):
            try:
                callback(message)
                count += 1
            except Exception as e:
                logger.warning(f"Subscriber error on {topic}: {e}")
        return count

    def subscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> None:
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> bool:
        subs = self._subscribers.get(topic, [])
        if callback in subs:
            subs.remove(callback)
            return True
        return False

    def get_history(self, topic: str, limit: int = 50) -> list[BusMessage]:
        msgs = list(self._history.get(topic, []))
        return msgs[-limit:]

    def register_agent(self, agent: AgentInfo) -> None:
        self._agents[agent.id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get_agents(self, role: str | None = None) -> list[AgentInfo]:
        agents = list(self._agents.values())
        if role:
            agents = [a for a in agents if a.role == role]
        return agents

    def update_heartbeat(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent:
            agent.last_heartbeat = time.time()


class RedisBus:
    """Redis Stream based message bus for production multi-agent communication."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 prefix: str = "prometheus:") -> None:
        if not HAS_REDIS:
            raise ImportError("Redis not available")
        self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._prefix = prefix
        self._subscribers: dict[str, list[Callable[[BusMessage], None]]] = defaultdict(list)
        self._agents: dict[str, AgentInfo] = {}

    def publish(self, topic: str, message: BusMessage) -> int:
        stream_key = f"{self._prefix}{topic}"
        return self._client.xadd(stream_key, message.serialize())

    def subscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> None:
        self._subscribers[topic].append(callback)

    def read_stream(self, topic: str, count: int = 10, block_ms: int = 1000) -> list[BusMessage]:
        stream_key = f"{self._prefix}{topic}"
        results = self._client.xread({stream_key: "0"}, count=count, block=block_ms)
        messages = []
        for _, entries in results:
            for entry_id, data in entries:
                messages.append(BusMessage.deserialize(data))
        return messages

    def register_agent(self, agent: AgentInfo) -> None:
        key = f"{self._prefix}agents:{agent.id}"
        self._client.hset(key, mapping={
            "name": agent.name, "role": agent.role, "model": agent.model,
            "capabilities": json.dumps(agent.capabilities),
            "status": agent.status,
        })
        self._agents[agent.id] = agent

    def get_agents(self, role: str | None = None) -> list[AgentInfo]:
        agents = list(self._agents.values())
        if role:
            agents = [a for a in agents if a.role == role]
        return agents


class EVOBus:
    """Unified communication bus with graceful degradation.

    Tries Redis first; falls back to InMemory when unavailable.
    This implements T5 (anti-fragile): communication failure → degrade, don't crash.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379) -> None:
        self._impl: InMemoryBus | RedisBus | None = None
        self._is_redis = False

        if HAS_REDIS:
            try:
                self._impl = RedisBus(host=redis_host, port=redis_port)
                self._impl._client.ping()
                self._is_redis = True
                logger.info("EVO Bus: Redis connected")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to in-memory: {e}")

        if not self._is_redis:
            self._impl = InMemoryBus()
            logger.info("EVO Bus: using in-memory fallback")

    @property
    def backend(self) -> str:
        return "redis" if self._is_redis else "inmemory"

    def publish(self, topic: str, payload: dict | None = None, sender: str = "",
                reply_to: str = "", priority: int = 0) -> int:
        """Publish a message to a topic."""
        msg = BusMessage(
            topic=topic, sender=sender, payload=payload or {},
            reply_to=reply_to, priority=priority,
        )
        return self._impl.publish(topic, msg)

    def subscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> None:
        """Subscribe to a topic."""
        self._impl.subscribe(topic, callback)

    def unsubscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> bool:
        """Unsubscribe from a topic."""
        if hasattr(self._impl, "unsubscribe"):
            return self._impl.unsubscribe(topic, callback)
        return False

    def get_history(self, topic: str, limit: int = 50) -> list[BusMessage]:
        """Get message history for a topic."""
        if hasattr(self._impl, "get_history"):
            return self._impl.get_history(topic, limit)
        return []

    def register_agent(self, agent: AgentInfo) -> None:
        self._impl.register_agent(agent)

    def get_agents(self, role: str | None = None) -> list[AgentInfo]:
        return self._impl.get_agents(role)

    # ── Prometheus Organ Channels ──
    # Pre-defined channels for the 5-organ pipeline
    ORGAN_CHANNELS = ["taotie_out", "nuwa_out", "darwin_out", "pool_out", "guard_out"]

    def subscribe_organ_channel(self, organ_name: str, callback: Callable[[BusMessage], None]) -> None:
        """Subscribe to an organ's output channel."""
        channel = f"{organ_name}_out"
        self.subscribe(channel, callback)

    def publish_organ_output(self, organ_name: str, payload: dict) -> int:
        """Publish an organ's output to its channel."""
        return self.publish(f"{organ_name}_out", payload, sender=organ_name)
