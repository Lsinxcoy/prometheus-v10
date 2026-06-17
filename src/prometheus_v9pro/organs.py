"""Prometheus V9 Organs — 5-organ pipeline with T4 Gagarin channel + T5 degradation."""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import CausalSignal, RawDesign, FitnessReport, Genome
from prometheus_v9pro.events import EventBus, Event
from prometheus_v9pro.utils import jaccard_similarity

logger = logging.getLogger(__name__)


class OrganBase(ABC):
    """Base class for organs. T5: each organ can degrade without killing the system."""

    def __init__(self, name: str, event_bus: EventBus | None = None) -> None:
        self.name = name
        self._event_bus = event_bus
        self._alive = True
        self._degraded = False

    @abstractmethod
    def process(self, data: Any) -> Any: ...

    def degrade(self) -> None:
        """T5 antifragile: degrade capability but stay alive."""
        self._degraded = True
        logger.warning(f"Organ {self.name} degraded (T5)")

    @property
    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self._alive = False


class TaotieOrgan(OrganBase):
    """饕餮: Ingest signals, T6 dual-scoring (fit + novelty)."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        super().__init__("taotie", event_bus)
        self._signals: list[CausalSignal] = []

    def process(self, data: list[CausalSignal] | None = None) -> list[CausalSignal]:
        if not self._alive:
            return []
        if self._degraded:
            # T5: low observe mode — only process existing signals
            return self._signals[:5]
        signals = data or []
        for s in signals:
            s.ingested_at = time.time()
        self._signals.extend(signals)
        if self._event_bus:
            self._event_bus.publish("taotie_done", {"signals": len(signals)}, source="taotie")
        return signals

    def dual_score(self, signal: CausalSignal, context_keywords: set[str]) -> tuple[float, float]:
        """T6: compute fit_score and novelty_score for a signal."""
        sig_words = set(signal.claim.lower().split())
        fit_score = jaccard_similarity(sig_words, context_keywords)
        # Novelty: inverse of similarity to all existing signals
        if not self._signals:
            novelty_score = 1.0
        else:
            max_sim = max(jaccard_similarity(sig_words, set(s.claim.lower().split())) for s in self._signals[:20])
            novelty_score = 1.0 - max_sim
        return fit_score, novelty_score


class NuwaOrgan(OrganBase):
    """女娲: Generate designs. T3 quantity>quality, T4 Gagarin channel."""

    def __init__(self, event_bus: EventBus | None = None, exploratory_ratio: float = 0.2) -> None:
        super().__init__("nuwa", event_bus)
        self._exploratory_ratio = exploratory_ratio
        self._designs: list[RawDesign] = []

    def process(self, data: list[CausalSignal] | None = None) -> list[RawDesign]:
        if not self._alive:
            return []
        if self._degraded:
            return self._designs[:2]
        signals = data or []
        if not signals:
            return []
        n_total = 10  # T3: quantity
        n_exploratory = max(2, int(n_total * self._exploratory_ratio))  # T4
        designs = []
        # Regular designs: recombination from signals
        for i in range(n_total - n_exploratory):
            combo = random.sample(signals, min(3, len(signals)))
            design = RawDesign(
                design_type=random.choice(["skill", "config", "code"]),
                inputs=[s.entity for s in combo],
                outputs=["improved_" + combo[0].entity],
                innovation_dimensions=["recombination"],
                resource_impact=random.uniform(0.1, 0.5),
                parallelization_potential=random.uniform(0, 1),
            )
            designs.append(design)
        # T4 Gagarin channel: exploratory designs
        for i in range(n_exploratory):
            design = RawDesign(
                design_type=random.choice(["skill", "config", "code"]),
                inputs=["hypothetical"],
                outputs=["unknown"],
                innovation_dimensions=["exploratory", "gagarin_channel"],
                resource_impact=0.0,
                parallelization_potential=random.uniform(0.5, 1.0),
            )
            designs.append(design)
        self._designs.extend(designs)
        if self._event_bus:
            self._event_bus.publish("nuwa_done", {"designs": len(designs), "exploratory": n_exploratory}, source="nuwa")
        return designs


class DarwinOrgan(OrganBase):
    """达尔文: Mutate all designs, push ALL to pool (no pre-filtering per T3)."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        super().__init__("darwin", event_bus)

    def process(self, data: list[RawDesign] | None = None) -> list[RawDesign]:
        if not self._alive:
            return []
        designs = data or []
        if self._degraded:
            return designs  # T5: pass-through without mutation
        # Mutate all designs (no filtering — T3)
        mutated = []
        for design in designs:
            m = RawDesign(
                design_type=design.design_type,
                inputs=design.inputs + ["mutated"],
                outputs=design.outputs,
                innovation_dimensions=(design.innovation_dimensions or []) + ["mutated"],
                resource_impact=design.resource_impact,
                parallelization_potential=design.parallelization_potential,
            )
            mutated.append(m)
        if self._event_bus:
            self._event_bus.publish("darwin_done", {"experiments": len(mutated)}, source="darwin")
        return mutated


class PoolOrgan(OrganBase):
    """进化池: Fitness evaluation. Canary vs evolved comparison."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        super().__init__("pool", event_bus)
        self._canary_baseline: float = 0.0
        self._reports: list[FitnessReport] = []

    def process(self, data: list[RawDesign] | None = None) -> list[FitnessReport]:
        if not self._alive:
            return []
        if self._degraded:
            return []  # T5: pending queue
        designs = data or []
        reports = []
        for i, design in enumerate(designs):
            evolved_score = random.uniform(0.0, 1.0)  # In production: real benchmark
            verdict = "survived" if evolved_score > self._canary_baseline else "died"
            report = FitnessReport(
                experiment_id=f"exp_{i}",
                canary_score=self._canary_baseline,
                evolved_score=evolved_score,
                verdict=verdict,
                metrics_delta={"delta": evolved_score - self._canary_baseline},
            )
            reports.append(report)
            if verdict == "survived":
                self._canary_baseline = evolved_score
        self._reports.extend(reports)
        if self._event_bus:
            survived = sum(1 for r in reports if r.verdict == "survived")
            self._event_bus.publish("pool_done", {"total": len(reports), "survived": survived}, source="pool")
        return reports


class GuardOrgan(OrganBase):
    """守卫: Promotion gate. T8: code changes need human approval."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        super().__init__("guard", event_bus)

    def process(self, data: list[FitnessReport] | None = None) -> list[dict]:
        if not self._alive:
            return []
        if self._degraded:
            # T5: fail-closed — reject all
            return [{"action": "rejected", "reason": "guard_degraded"}]
        reports = data or []
        promotions = []
        for report in reports:
            if report.verdict != "survived":
                continue
            # T8: code changes require approval
            design_type = report.metrics_delta.get("design_type", "config")
            if design_type == "code":
                promotions.append({"action": "pending_approval", "reason": "T8_code_requires_approval", "report": report})
            else:
                promotions.append({"action": "auto_apply", "reason": "non_code_improvement", "report": report})
        if self._event_bus:
            self._event_bus.publish("guard_done", {"promotions": len(promotions)}, source="guard")
        return promotions


class OrganPipeline:
    """Wire 5 organs via EventBus. T5 degradation support."""

    def __init__(self, event_bus: EventBus | None = None, config=None) -> None:
        self._event_bus = event_bus or EventBus()
        self.taotie = TaotieOrgan(self._event_bus)
        self.nuwa = NuwaOrgan(self._event_bus)
        self.darwin = DarwinOrgan(self._event_bus)
        self.pool = PoolOrgan(self._event_bus)
        self.guard = GuardOrgan(self._event_bus)

    def run(self, signals: list[CausalSignal]) -> list[dict]:
        """Run full pipeline: signals → designs → mutations → fitness → promotions."""
        # Step 1: Taotie ingests signals
        processed_signals = self.taotie.process(signals)
        # Step 2: Nuwa generates designs
        designs = self.nuwa.process(processed_signals)
        # Step 3: Darwin mutates
        mutated = self.darwin.process(designs)
        # Step 4: Pool evaluates fitness
        reports = self.pool.process(mutated)
        # Step 5: Guard promotes
        promotions = self.guard.process(reports)
        return promotions

    def degrade_organ(self, organ_name: str) -> None:
        """T5: Degrade specific organ."""
        organs = {"taotie": self.taotie, "nuwa": self.nuwa, "darwin": self.darwin, "pool": self.pool, "guard": self.guard}
        if organ_name in organs:
            organs[organ_name].degrade()


# ── Organ-Evolution Bridge (V8 organs/bridge.py consolidated) ─────

class OrganEvolutionBridge:
    """Bidirectional bridge between organ pipeline and evolution engine.

    V8 had organs/bridge.py (192 lines) as separate module.
    V9PRO consolidates: pipeline↔genome conversion + feedback loop.

    Without this bridge, organs and evolution are disconnected:
    - Organs don't know evolution results
    - Evolution doesn't receive organ feedback
    This bridge closes the loop: organ output → genome updates →
    evolution → organ parameter adjustments.
    """

    def __init__(self, pipeline: OrganPipeline | None = None) -> None:
        self._pipeline = pipeline
        self._feedback_count = 0
        self._genome_updates = 0

    def pipeline_to_genome(self, reports: list[dict]) -> dict[str, Any]:
        """Convert pipeline fitness reports into genome update suggestions.

        Maps organ-level outcomes to genome parameters:
        - Taotie signal quality → L0 meta-params
        - Nuwa design novelty → L1 direction weights
        - Darwin mutation success → L2 mutation config
        - Pool fitness scores → L3 benchmark thresholds
        - Guard promotion rate → L5 meta-evolution
        """
        self._feedback_count += 1
        suggestions: dict[str, Any] = {
            "mutation_rate_adjustment": 0.0,
            "exploration_bonus": 0.0,
            "fitness_threshold_adjustment": 0.0,
            "direction_weights": {"forward": 1.0, "lateral": 1.0, "reverse": 1.0},
        }

        if not reports:
            return suggestions

        # Analyze reports
        avg_fitness = sum(r.get("fitness_delta", 0.0) for r in reports) / max(1, len(reports))
        success_rate = sum(1 for r in reports if r.get("fitness_delta", 0.0) > 0) / max(1, len(reports))

        # Fitness feedback → mutation rate
        if avg_fitness < 0:
            suggestions["mutation_rate_adjustment"] = 0.1  # increase exploration
        elif avg_fitness > 0.1:
            suggestions["mutation_rate_adjustment"] = -0.05  # exploit current direction

        # Success rate → exploration bonus
        if success_rate < 0.3:
            suggestions["exploration_bonus"] = 0.2  # boost Gagarin channel
        elif success_rate > 0.7:
            suggestions["exploration_bonus"] = -0.1

        # Fitness distribution → direction weights
        if avg_fitness > 0:
            suggestions["direction_weights"]["forward"] = 1.5  # continue forward
        else:
            suggestions["direction_weights"]["lateral"] = 1.5  # try lateral

        return suggestions

    def genome_to_pipeline(self, genome_config: dict[str, Any]) -> dict[str, Any]:
        """Convert genome configuration into pipeline parameter adjustments.

        Maps genome state to organ parameters:
        - mutation_rate → Darwin's mutation intensity
        - direction → Nuwa's design generation strategy
        - fitness_threshold → Pool's evaluation strictness
        """
        self._genome_updates += 1
        adjustments: dict[str, Any] = {}

        mutation_rate = genome_config.get("mutation_rate", 0.1)
        adjustments["darwin_intensity"] = mutation_rate
        adjustments["nuwa_novelty_threshold"] = max(0.1, 1.0 - mutation_rate * 5)
        adjustments["pool_threshold"] = genome_config.get("fitness_threshold", 0.5)

        return adjustments

    def apply_feedback(self, genome, reports: list[dict]) -> None:
        """Apply pipeline feedback to genome (closes the loop)."""
        suggestions = self.pipeline_to_genome(reports)
        if hasattr(genome, "config") and isinstance(genome.config, dict):
            current_mr = genome.config.get("mutation_rate", 0.1)
            genome.config["mutation_rate"] = max(0.01, min(0.5,
                current_mr + suggestions["mutation_rate_adjustment"]))
            self._genome_updates += 1

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "feedback_count": self._feedback_count,
            "genome_updates": self._genome_updates,
        }
