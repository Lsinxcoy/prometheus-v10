"""Prometheus V9 Harness — Architecture-level meta-evolution (evolution of the evolution system itself).

The Harness monitors the evolution engine's own architecture and can restructure it.
This is L5 MetaEvolution's big brother: L5 handles parameter stagnation,
Harness handles architectural stagnation.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from prometheus_v10.schema import Genome

logger = logging.getLogger(__name__)


class HarnessAction(str, Enum):
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    REORDER_LAYERS = "reorder_layers"
    MODIFY_LAYER_WEIGHT = "modify_layer_weight"
    ADD_ORGAN = "add_organ"
    RECONFIGURE_ORGAN = "reconfigure_organ"
    ADAPT_STRATEGY = "adapt_strategy"
    EXPAND_SEARCH_SPACE = "expand_search_space"
    CONTRACT_SEARCH_SPACE = "contract_search_space"


@dataclass
class HarnessProposal:
    """A proposed architectural change."""
    action: HarnessAction = HarnessAction.ADAPT_STRATEGY
    target: str = ""
    parameters: dict = field(default_factory=dict)
    expected_impact: float = 0.0
    risk: float = 0.5
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class HarnessResult:
    """Result of applying a harness action."""
    proposal: HarnessProposal | None = None
    applied: bool = False
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    rollback_possible: bool = True


class EvolutionHarness:
    """Meta-evolution harness: monitors and restructures the evolution engine.

    Detection signals:
    - Layer utilization: which layers produce the most fitness delta?
    - Stagnation pattern: is the same layer always stagnant?
    - Organ bottleneck: which organ is the slowest?
    - Search space collapse: are we only exploring a small region?

    Actions:
    - Increase weight of high-impact layers
    - Decrease weight of low-impact layers
    - Expand search space when stagnant
    - Contract search space when chaotic
    """

    def __init__(self, engine=None) -> None:
        self._engine = engine
        self._history: list[HarnessResult] = []
        self._layer_impact: dict[str, float] = {}
        self._proposal_count = 0

    def analyze(self, evolution_results: list[dict]) -> HarnessProposal | None:
        """Analyze evolution results and propose architectural change."""
        if not evolution_results:
            return None

        # Compute layer impact scores
        for result in evolution_results:
            layer_name = result.get("layer_name", "")
            delta = result.get("fitness_delta", 0.0)
            self._layer_impact[layer_name] = self._layer_impact.get(layer_name, 0.0) + delta

        # Find lowest-impact and highest-impact layers
        if not self._layer_impact:
            return None
        sorted_layers = sorted(self._layer_impact.items(), key=lambda x: x[1])
        lowest = sorted_layers[0]
        highest = sorted_layers[-1]

        # Propose action based on analysis
        if lowest[1] < 0 and highest[1] > 0.1:
            # Low-impact layer is negative → reduce its weight
            return HarnessProposal(
                action=HarnessAction.MODIFY_LAYER_WEIGHT,
                target=lowest[0],
                parameters={"weight_factor": 0.5},
                expected_impact=abs(lowest[1]) * 0.5,
                risk=0.2,
                rationale=f"Layer {lowest[0]} has negative impact ({lowest[1]:.3f}), reducing weight",
            )
        elif highest[1] < 0.01:
            # All layers low impact → expand search space
            return HarnessProposal(
                action=HarnessAction.EXPAND_SEARCH_SPACE,
                target="global",
                parameters={"expansion_factor": 1.5},
                expected_impact=0.1,
                risk=0.4,
                rationale="All layers have low impact, expanding search space",
            )
        return None

    def apply_proposal(self, proposal: HarnessProposal, genome: Genome) -> HarnessResult:
        """Apply a harness proposal to the genome."""
        self._proposal_count += 1
        fitness_before = genome.fitness

        result = HarnessResult(proposal=proposal, fitness_before=fitness_before)

        try:
            if proposal.action == HarnessAction.MODIFY_LAYER_WEIGHT:
                factor = proposal.parameters.get("weight_factor", 1.0)
                target = proposal.target
                if target in genome.config:
                    genome.config[target] = genome.config[target] * factor
                result.applied = True
            elif proposal.action == HarnessAction.EXPAND_SEARCH_SPACE:
                expansion = proposal.parameters.get("expansion_factor", 1.5)
                genome.config["mutation_rate"] = min(1.0, genome.config.get("mutation_rate", 0.3) * expansion)
                result.applied = True
            elif proposal.action == HarnessAction.CONTRACT_SEARCH_SPACE:
                contraction = proposal.parameters.get("contraction_factor", 0.7)
                genome.config["mutation_rate"] = max(0.01, genome.config.get("mutation_rate", 0.3) * contraction)
                result.applied = True
            elif proposal.action == HarnessAction.ADAPT_STRATEGY:
                result.applied = True  # Strategy adaptation handled by L1

            result.fitness_after = genome.fitness
        except Exception as e:
            logger.warning(f"Harness action failed: {e}")
            result.applied = False
            result.rollback_possible = True

        self._history.append(result)
        return result

    def rollback_last(self, genome: Genome) -> bool:
        """Rollback the last applied proposal."""
        if not self._history or not self._history[-1].rollback_possible:
            return False
        last = self._history.pop()
        if last.proposal:
            # Reverse the action
            if last.proposal.action == HarnessAction.MODIFY_LAYER_WEIGHT:
                factor = last.proposal.parameters.get("weight_factor", 1.0)
                target = last.proposal.target
                if target in genome.config and factor > 0:
                    genome.config[target] = genome.config[target] / factor
            elif last.proposal.action == HarnessAction.EXPAND_SEARCH_SPACE:
                expansion = last.proposal.parameters.get("expansion_factor", 1.5)
                genome.config["mutation_rate"] = max(0.01, genome.config.get("mutation_rate", 0.3) / expansion)
        return True

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "proposals_made": self._proposal_count,
            "proposals_applied": sum(1 for h in self._history if h.applied),
            "layer_impact": dict(self._layer_impact),
        }
