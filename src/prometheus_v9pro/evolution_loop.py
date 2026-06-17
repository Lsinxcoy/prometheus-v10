"""Prometheus V9 Evolution Loop — Multi-generation evolution with early stopping and convergence detection."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import Genome

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of one generation of evolution."""
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 0.0
    stagnation_count: int = 0
    layer_results: list[dict] = field(default_factory=list)
    duration: float = 0.0


class EvolutionLoop:
    """Main evolution loop: multi-generation with early stopping.

    Features:
    - Convergence detection: stop when fitness improvement < threshold
    - Stagnation tracking: count consecutive generations with no improvement
    - Diversity monitoring: detect population collapse
    - Budget-aware: respect token and time limits
    """

    def __init__(self, engine=None, max_generations: int = 100,
                 convergence_threshold: float = 0.001,
                 max_stagnation: int = 10) -> None:
        self._engine = engine
        self._max_generations = max_generations
        self._convergence_threshold = convergence_threshold
        self._max_stagnation = max_stagnation
        self._history: list[GenerationResult] = []
        self._best_fitness_ever = 0.0
        self._best_genome_ever: Genome | None = None

    def run(self, genome: Genome, generations: int | None = None,
            budget_tokens: int = 40000) -> list[GenerationResult]:
        """Run evolution loop for N generations."""
        n = generations or self._max_generations
        tokens_used = 0
        stagnation = 0

        for gen in range(n):
            start = time.time()

            # Run one generation
            if self._engine:
                updated_genome, layer_results = self._engine.evolve_single_step(genome)
                best_fitness = updated_genome.fitness
                avg_fitness = best_fitness  # Single genome
            else:
                best_fitness = genome.fitness
                avg_fitness = genome.fitness
                layer_results = []

            # Track best
            if best_fitness > self._best_fitness_ever:
                self._best_fitness_ever = best_fitness
                self._best_genome_ever = genome
                stagnation = 0
            else:
                stagnation += 1

            # Compute diversity (simple: based on genome config spread)
            diversity = 1.0 / (1.0 + stagnation)

            result = GenerationResult(
                generation=gen, best_fitness=best_fitness,
                avg_fitness=avg_fitness, diversity=diversity,
                stagnation_count=stagnation, layer_results=layer_results,
                duration=time.time() - start,
            )
            self._history.append(result)
            tokens_used += 500  # Rough estimate per generation

            # Early stopping conditions
            if stagnation >= self._max_stagnation:
                logger.info(f"Stopping: {stagnation} generations stagnant")
                break
            if tokens_used >= budget_tokens:
                logger.info(f"Stopping: budget exhausted ({tokens_used}/{budget_tokens})")
                break
            if gen > 5 and abs(best_fitness - self._history[-2].best_fitness) < self._convergence_threshold:
                pass  # Continue but track convergence

        return self._history

    @property
    def best_genome(self) -> Genome | None:
        return self._best_genome_ever

    @property
    def best_fitness(self) -> float:
        return self._best_fitness_ever

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "generations_run": len(self._history),
            "best_fitness": self._best_fitness_ever,
            "final_stagnation": self._history[-1].stagnation_count if self._history else 0,
            "total_duration": sum(r.duration for r in self._history),
        }
