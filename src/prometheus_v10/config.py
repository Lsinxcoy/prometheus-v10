"""Prometheus V9 Configuration — All configurable parameters, philosophy toggles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus_v10.schema import EVOLUTION_LAYERS, WeibullParams


@dataclass
class PhilosophyConfig:
    """8 tenets from CIP design — each can be toggled for testing."""
    t1_emergence: bool = True      # 涌现 > 设计
    t2_collective: bool = True     # 集体 > 个体
    t3_quantity: bool = True       # 数量 > 质量 (candidate layer)
    t4_hypothesis: bool = True     # 假设 > 拟合
    t5_antifragile: bool = True    # 抗脆弱 > 健壮
    t6_fit_over_novel: bool = True # 适配 > 新颖
    t7_business_metric: bool = True # 业务指标 > 内部 metric
    t8_creator_sovereignty: bool = True  # 造物主主权 > AI自治


@dataclass
class DegradationConfig:
    """T5 antifragile: degradation strategies per organ."""
    taotie_mode: str = "low_observe"     # ingest only existing DNA
    nuwa_mode: str = "ledger_cache"      # store designs for later
    darwin_mode: str = "original_only"   # no mutation, pass-through
    pool_mode: str = "pending_queue"     # block promotions
    guard_mode: str = "fail_closed"      # reject all promotions


@dataclass
class OrganConfig:
    """5-organ pipeline configuration."""
    taotie_sources: list[str] = field(default_factory=lambda: ["local", "github", "arxiv"])
    nuwa_designs_per_batch: int = 10
    nuwa_exploratory_ratio: float = 0.2   # T4: 2/10 = 0.2
    darwin_max_concurrent: int = 10
    pool_timeout_seconds: int = 300
    guard_auto_apply_code: bool = False   # T8: code changes need approval


@dataclass
class EvolutionConfig:
    """Evolution engine configuration."""
    population_size: int = 20
    elite_ratio: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    stagnation_threshold: int = 10
    stagnation_delta: float = 0.01
    max_generations: int = 100
    budget_tokens: int = 4000
    budget_time: int = 240


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    db_path: str = "data/prometheus_v10.db"
    vector_dim: int = 128
    weibull: WeibullParams = field(default_factory=WeibullParams)
    forget_threshold: float = 0.1
    consolidation_threshold: float = 0.6
    episodic_window: int = 10
    semantic_growth_rate: int = 3  # tokens per message


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    breaker_failure_threshold: int = 5
    breaker_recovery_timeout: float = 60.0
    confidence_threshold_ask: float = 0.7
    confidence_threshold_defer: float = 0.5
    oep_jaccard_threshold: float = 0.3


@dataclass
class PrometheusConfig:
    """Top-level configuration with convenience field forwarding."""
    philosophy: PhilosophyConfig = field(default_factory=PhilosophyConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    organs: OrganConfig = field(default_factory=OrganConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    layer_configs: list = field(default_factory=lambda: list(EVOLUTION_LAYERS))

    # Convenience fields — forwarded to sub-configs via __post_init__
    vector_dim: int | None = field(default=None, repr=False)
    exploratory_ratio: float | None = field(default=None, repr=False)
    stagnation_threshold: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Forward convenience fields to sub-configs; when not set, inherit from sub-config defaults
        if self.vector_dim is not None:
            self.memory.vector_dim = self.vector_dim
        else:
            self.vector_dim = self.memory.vector_dim
        if self.exploratory_ratio is not None:
            self.organs.nuwa_exploratory_ratio = self.exploratory_ratio
        else:
            self.exploratory_ratio = self.organs.nuwa_exploratory_ratio
        if self.stagnation_threshold is not None:
            self.evolution.stagnation_threshold = self.stagnation_threshold
        else:
            self.stagnation_threshold = self.evolution.stagnation_threshold

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj
        return _convert(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrometheusConfig:
        """Deserialize from dict."""
        cfg = cls()
        if "philosophy" in data:
            for k, v in data["philosophy"].items():
                if hasattr(cfg.philosophy, k):
                    setattr(cfg.philosophy, k, v)
        if "evolution" in data:
            for k, v in data["evolution"].items():
                if hasattr(cfg.evolution, k):
                    setattr(cfg.evolution, k, v)
        if "memory" in data:
            for k, v in data["memory"].items():
                if hasattr(cfg.memory, k):
                    setattr(cfg.memory, k, v)
        return cfg

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str) -> PrometheusConfig:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
