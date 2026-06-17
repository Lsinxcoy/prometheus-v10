"""Prometheus V9 Schema — All domain types, 12-layer configs, evolution primitives."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from prometheus_v10.utils import generate_uuidv7


# ── Node & Edge Types ──────────────────────────────────────────────

class NodeType(Enum):
    AGENT = "agent"
    USER = "user"
    EPISODE = "episode"
    FACT = "fact"
    SKILL = "skill"
    PATTERN = "pattern"
    INSIGHT = "insight"
    BELIEF = "belief"
    FORESIGHT = "foresight"
    HYPOTHESIS = "hypothesis"
    CODE_UNIT = "code_unit"
    DOMAIN = "domain"
    TOOL = "tool"
    CONFIG = "config"
    NOTE = "note"
    ARCHITECTURE = "architecture"


class EdgeType(Enum):
    DERIVES_FROM = "derives_from"
    CAUSES = "causes"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    SUPERSEDES = "supersedes"
    BELONGS_TO = "belongs_to"
    REFERENCES = "references"
    FORBIDS = "forbids"
    TRUSTS = "trusts"
    RELATED = "relates_to"  # alias for RELATES_TO


# ── Memory Layers & Scopes ─────────────────────────────────────────

class MemoryLayer(Enum):
    SENSORY = "sensory"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SKILL = "skill"
    ARCHIVAL = "archival"


class MemoryScope(Enum):
    AGENT = "agent"
    USER = "user"
    GLOBAL = "global"


class TrustLevel(Enum):
    PENDING = "pending"
    HIGH_SIGNAL = "high_signal"
    VERIFIED = "verified"


class AutonomyLevel(Enum):
    L0_OBSERVE = 0
    L1_SUGGEST = 1
    L2_ACT_WITH_APPROVAL = 2
    L3_ACT_AND_REPORT = 3
    L4_FULL_AUTONOMY = 4


class EvolutionDirection(Enum):
    FORWARD = "forward"
    LATERAL = "lateral"
    REVERSE = "reverse"


# ── Core Data Classes ──────────────────────────────────────────────

class ProvenanceType(Enum):
    UNKNOWN = "unknown"
    OBSERVED = "observed"
    INFERRED = "inferred"
    IMPORTED = "imported"
    GENERATED = "generated"

@dataclass
class Provenance:
    source: ProvenanceType = ProvenanceType.UNKNOWN
    agent_id: str = ""
    confidence: float = 0.5

@dataclass
class WeibullParams:
    k: float = 1.5        # shape parameter
    lambda_: float = 86400.0  # scale (seconds, default 1 day)
    t0: float = 0.0       # location

@dataclass
class NodePayload:
    content: str = ""
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    id: bytes = field(default_factory=generate_uuidv7)
    type: NodeType = NodeType.FACT
    payload: NodePayload = field(default_factory=NodePayload)
    layer: MemoryLayer = MemoryLayer.EPISODIC
    scope: MemoryScope = MemoryScope.GLOBAL
    trust_level: TrustLevel = TrustLevel.PENDING
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    retention: float = 1.0  # 0-1, how well this memory is retained
    weibull: WeibullParams = field(default_factory=WeibullParams)
    provenance: Provenance | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    action_hook: str | None = None
    fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.access_count += 1
        self.updated_at = time.time()


@dataclass
class Edge:
    id: bytes = field(default_factory=generate_uuidv7)
    source: bytes = b""
    target: bytes = b""
    type: EdgeType = EdgeType.RELATES_TO
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Genome:
    code: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    fitness: float = 0.0
    fingerprint: str = ""
    generation: int = 0


# ── Evolution Primitives ───────────────────────────────────────────

@dataclass
class EvolutionLayerConfig:
    layer_id: str = ""
    name: str = ""
    description: str = ""
    risk_level: float = 0.1
    cycle: str = "day"       # minute/hour/day/week/month
    target_types: list[str] = field(default_factory=list)
    parallel_safe: bool = True


EVOLUTION_LAYERS: list[EvolutionLayerConfig] = [
    EvolutionLayerConfig("L0", "meta_params", "Thompson Sampling hyperparameter optimization", 0.1, "minute", ["mutation_rate", "crossover_rate", "elite_ratio", "population_size"]),
    EvolutionLayerConfig("L1", "strategy", "UCB1 direction selection", 0.2, "hour", ["direction"]),
    EvolutionLayerConfig("L2", "skill", "Skill acquisition via gap analysis", 0.2, "day", ["skill"]),
    EvolutionLayerConfig("L3", "config", "Configuration optimization via runtime benchmark", 0.3, "day", ["config"]),
    EvolutionLayerConfig("L4", "code", "AST mutation + human approval gate", 0.8, "week", ["code"]),
    EvolutionLayerConfig("L5", "meta_evolution", "UCB1 strategy selection on stagnation", 0.9, "month", ["evolution_strategy"]),
    EvolutionLayerConfig("L6", "prompt", "LLM rewrite + heuristic fallback", 0.3, "day", ["prompt"]),
    EvolutionLayerConfig("L7", "tool", "Tool utility tracking + recommendation", 0.4, "day", ["tool"]),
    EvolutionLayerConfig("L8", "memory", "Memory strategy selection + entropy stratification", 0.3, "day", ["memory_strategy"]),
    EvolutionLayerConfig("L9", "knowledge", "6-dim gap detection + filling", 0.4, "week", ["knowledge"]),
    EvolutionLayerConfig("L10", "collaboration", "UCB1 collaboration strategy selection", 0.5, "week", ["collaboration_mode"]),
    EvolutionLayerConfig("L11", "architecture", "Multi-dim health monitoring + repair", 0.7, "month", ["architecture"]),
]


# ── Organ Primitives ───────────────────────────────────────────────

@dataclass
class CausalSignal:
    entity: str = ""
    claim: str = ""
    direction: str = "positive"  # positive/negative/neutral
    source: str = ""
    source_quality: float = 1.0
    ingested_at: float | None = None


@dataclass
class RawDesign:
    design_type: str = "config"   # skill / config / code
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] | None = None
    innovation_dimensions: list[str] | None = None
    resource_impact: float | None = None
    parallelization_potential: float | None = None
    pattern: str | None = None
    target_modules: list[str] | None = None
    change_scope: float | None = None


@dataclass
class ReflectionNote:
    task: str = ""
    outcome: str = ""     # success / partial / failure
    insights: list[str] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class FitnessReport:
    experiment_id: str = ""
    canary_score: float = 0.0
    evolved_score: float = 0.0
    verdict: str = "died"   # survived / died
    metrics_delta: dict[str, Any] = field(default_factory=dict)


# ── Helper Factories ───────────────────────────────────────────────

def create_episode_node(content: str, tags: list[str] | None = None, importance: float = 0.5) -> Node:
    return Node(
        type=NodeType.EPISODE,
        payload=NodePayload(content=content),
        layer=MemoryLayer.EPISODIC,
        tags=tags or [],
        importance=importance,
    )


def create_skill_node(content: str, tags: list[str] | None = None, importance: float = 0.7) -> Node:
    return Node(
        type=NodeType.SKILL,
        payload=NodePayload(content=content),
        layer=MemoryLayer.SKILL,
        tags=tags or [],
        importance=importance,
        trust_level=TrustLevel.HIGH_SIGNAL,
    )


def create_insight_node(content: str, importance: float = 0.6) -> Node:
    return Node(
        type=NodeType.INSIGHT,
        payload=NodePayload(content=content),
        layer=MemoryLayer.SEMANTIC,
        importance=importance,
    )


def create_hypothesis_node(content: str, tags: list[str] | None = None, importance: float = 0.4) -> Node:
    return Node(
        type=NodeType.HYPOTHESIS,
        payload=NodePayload(content=content),
        layer=MemoryLayer.SEMANTIC,
        tags=tags or [],
        importance=importance,
    )


# ── HarnessX Integration (§3 Composition + §3.3 Nine Dimensions) ───

class HookPoint(str, Enum):
    """HarnessX §3: 8 lifecycle hook points for Processor attachment."""
    TASK_START = "task_start"
    STEP_START = "step_start"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    STEP_END = "step_end"       # read-only
    TASK_END = "task_end"       # read-only


class HarnessDimension(str, Enum):
    """HarnessX §3.3: 9-dimensional behavioral taxonomy."""
    D1_MODEL = "model"              # model selection
    D2_CONTEXT = "context"          # context assembly
    D3_MEMORY = "memory"            # memory management
    D4_TOOL = "tool"                # tool ecosystem
    D5_SANDBOX = "sandbox"          # execution environment
    D6_EVAL = "eval"                # evaluation & reward
    D7_CONTROL = "control"          # control & safety
    D8_OBSERVE = "observe"          # observability
    D9_TRAIN = "train"              # training bridge


@dataclass
class ProcessorEvent:
    """Event flowing through a hook point. HarnessX §3.2."""
    hook: HookPoint = HookPoint.STEP_START
    payload: dict[str, Any] = field(default_factory=dict)
    read_only: bool = False


@dataclass
class HarnessEdit:
    """Typed harness modification — e: H → H preserving type contracts. HarnessX §3.2."""
    action: str = "replace"    # insert | replace | remove
    hook: HookPoint = HookPoint.BEFORE_MODEL
    target_singleton_group: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    # manifest keys: changed_components, intended_effect,
    # tasks_expected_improve, tasks_expected_regress, risk_factors, smoke_test_passed


@dataclass
class HarnessConfig:
    """H = (M, C) — Harness as first-class object. HarnessX §3.1.
    M = model_config, C = (P, S) where P: Hook→List[Processor], S = shared slots.
    First-class: independently serializable, comparable, hashable, substitutable.
    """
    model_config: dict[str, Any] = field(default_factory=dict)
    processors: dict[str, list] = field(default_factory=dict)  # HookPoint.value → list
    slots: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Deterministic hash for comparison and variant routing."""
        import hashlib
        data = json.dumps({
            "model": self.model_config,
            "hooks": {k: [str(p) for p in v] for k, v in self.processors.items()},
            "slots": sorted(self.slots.keys()),
        }, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def diff(self, other: HarnessConfig) -> list[HarnessEdit]:
        """Compute typed edits to transform self into other."""
        edits: list[HarnessEdit] = []
        # Model config changes
        if self.model_config != other.model_config:
            edits.append(HarnessEdit(
                action="replace", hook=HookPoint.TASK_START,
                manifest={"changed_components": ["model_config"],
                          "intended_effect": "model selection change"},
            ))
        # Processor changes per hook
        all_hooks = set(list(self.processors.keys()) + list(other.processors.keys()))
        for hook_key in all_hooks:
            self_procs = self.processors.get(hook_key, [])
            other_procs = other.processors.get(hook_key, [])
            if self_procs != other_procs:
                edits.append(HarnessEdit(
                    action="replace", hook=HookPoint(hook_key) if hook_key in [h.value for h in HookPoint] else HookPoint.STEP_START,
                    manifest={"changed_components": [f"processors.{hook_key}"],
                              "intended_effect": f"processor pipeline change at {hook_key}"},
                ))
        return edits


# ── Layer→Dimension Mapping (12 layers × 9 dimensions) ─────────────

LAYER_TO_DIMENSIONS: dict[str, list[HarnessDimension]] = {
    "L0": [HarnessDimension.D1_MODEL, HarnessDimension.D8_OBSERVE],
    "L1": [HarnessDimension.D7_CONTROL],
    "L2": [HarnessDimension.D2_CONTEXT, HarnessDimension.D4_TOOL],
    "L3": [HarnessDimension.D6_EVAL],
    "L4": [HarnessDimension.D3_MEMORY, HarnessDimension.D8_OBSERVE],
    "L5": [HarnessDimension.D1_MODEL, HarnessDimension.D2_CONTEXT,
           HarnessDimension.D4_TOOL, HarnessDimension.D7_CONTROL],
    "L6": [HarnessDimension.D2_CONTEXT, HarnessDimension.D3_MEMORY,
           HarnessDimension.D8_OBSERVE],
    "L7": [HarnessDimension.D3_MEMORY, HarnessDimension.D4_TOOL,
           HarnessDimension.D5_SANDBOX],
    "L8": [HarnessDimension.D3_MEMORY, HarnessDimension.D7_CONTROL],
    "L9": [HarnessDimension.D2_CONTEXT],
    "L10": [HarnessDimension.D2_CONTEXT, HarnessDimension.D4_TOOL],
    "L11": [HarnessDimension.D1_MODEL, HarnessDimension.D2_CONTEXT,
            HarnessDimension.D3_MEMORY, HarnessDimension.D4_TOOL,
            HarnessDimension.D5_SANDBOX, HarnessDimension.D6_EVAL,
            HarnessDimension.D7_CONTROL, HarnessDimension.D8_OBSERVE,
            HarnessDimension.D9_TRAIN],
}

# Reverse: which layers touch a given dimension
DIMENSION_TO_LAYERS: dict[HarnessDimension, list[str]] = {}
for _layer, _dims in LAYER_TO_DIMENSIONS.items():
    for _dim in _dims:
        DIMENSION_TO_LAYERS.setdefault(_dim, []).append(_layer)
