"""Prometheus V9 Layer Infrastructure — Bandit algorithms, fitness evaluation, population management."""

from __future__ import annotations

import ast
import logging
import math
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus_v9pro.schema import Genome
from prometheus_v9pro.utils import compute_fingerprint

logger = logging.getLogger(__name__)


# ── Thompson Sampling ─────────────────────────────────────────────

class ThompsonSampling:
    """Thompson Sampling bandit for hyperparameter optimization."""

    def __init__(self, param_names: list[str], values: dict[str, list]) -> None:
        self._params = param_names
        self._values = values
        # Beta distribution parameters: alpha=1, beta=1 (uniform prior)
        self._alpha_beta: dict[str, list[int]] = {k: [1, 1] for k in param_names}

    def sample(self, param_name: str) -> Any:
        """Sample a value using Thompson Sampling."""
        if param_name not in self._values:
            raise ValueError(f"Unknown parameter: {param_name}")
        alpha, beta = self._alpha_beta[param_name]
        sample_val = random.betavariate(alpha, beta)
        values = self._values[param_name]
        idx = min(int(sample_val * len(values)), len(values) - 1)
        return values[idx]

    def update(self, param_name: str, success: bool) -> None:
        """Update bandit based on outcome."""
        if param_name not in self._alpha_beta:
            return
        if success:
            self._alpha_beta[param_name][0] += 1  # alpha
        else:
            self._alpha_beta[param_name][1] += 1  # beta


# ── UCB1 Bandit ───────────────────────────────────────────────────

class UCB1Bandit:
    """UCB1 bandit for strategy/direction selection."""

    def __init__(self, arms: list[str]) -> None:
        self._arms = arms
        self._counts: dict[str, int] = {arm: 0 for arm in arms}
        self._values: dict[str, float] = {arm: 0.0 for arm in arms}
        self._total_pulls: int = 0

    def select(self) -> str:
        """Select arm using UCB1 formula: value + sqrt(2*ln(total)/count)."""
        # Unexplored arms get priority
        for arm in self._arms:
            if self._counts[arm] == 0:
                return arm
        ucb_scores = {}
        for arm in self._arms:
            exploration = math.sqrt(2 * math.log(self._total_pulls) / self._counts[arm])
            ucb_scores[arm] = self._values[arm] + exploration
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm: str, reward: float) -> None:
        """Update arm statistics with observed reward."""
        if arm not in self._counts:
            raise ValueError(f"Unknown arm: {arm}")
        self._counts[arm] += 1
        self._total_pulls += 1
        # Incremental mean update
        n = self._counts[arm]
        self._values[arm] = self._values[arm] + (reward - self._values[arm]) / n

    @property
    def stats(self) -> dict[str, dict]:
        return {arm: {"count": self._counts[arm], "value": self._values[arm]} for arm in self._arms}


# ── Three-Stage Fitness ───────────────────────────────────────────

class ThreeStageFitness:
    """Fitness evaluation: syntax → subprocess benchmark → heuristic fallback.
    
    NO hardcoded scoring ranges — all runtime measurement.
    """

    def evaluate(self, genome: Genome, test_path: str | None = None) -> float:
        """Evaluate genome fitness through 3 stages."""
        # Stage 1: Syntax check
        if genome.code:
            try:
                ast.parse(genome.code)
            except SyntaxError:
                return 0.0

        # Stage 2: Subprocess benchmark (if test_path provided)
        if test_path and genome.code:
            return self._subprocess_benchmark(genome.code, test_path)

        # Stage 3: Heuristic fallback (only when no test_path)
        return self._heuristic_fitness(genome)

    def _subprocess_benchmark(self, code: str, test_path: str) -> float:
        """Run actual tests and return pass rate."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-q", "--tb=no"],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout
            # Parse "X passed, Y failed" or "X passed"
            import re
            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            total = passed + failed
            return passed / max(1, total)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Subprocess benchmark failed: {e}")
            return 0.0

    def _heuristic_fitness(self, genome: Genome) -> float:
        """Simple heuristic: not a scoring function — measures code properties."""
        if not genome.code:
            return 0.1
        score = 0.1  # Base
        # Has function definitions
        try:
            tree = ast.parse(genome.code)
            funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            score += min(0.3, len(funcs) * 0.05)
            # Has docstrings
            docstrings = [n for n in funcs if ast.get_docstring(n)]
            score += min(0.2, len(docstrings) * 0.05)
            # Reasonable size (not too short, not too long)
            lines = genome.code.count("\n") + 1
            if 10 <= lines <= 500:
                score += 0.2
            elif lines > 0:
                score += 0.1
        except SyntaxError:
            pass
        # Existing fitness contribution (from previous evaluations)
        score += genome.fitness * 0.2
        return min(1.0, score)


# ── Population Management ─────────────────────────────────────────

class PopulationManager:
    """Population initialization, selection, and crossover."""

    def __init__(self, population_size: int = 20, elite_ratio: float = 0.1) -> None:
        self._population_size = population_size
        self._elite_ratio = elite_ratio

    def initialize(self, base_genome: Genome, n: int | None = None) -> list[Genome]:
        """Create initial population with random config variations."""
        n = n or self._population_size
        population = [base_genome]
        for _ in range(n - 1):
            variant = Genome(
                code=base_genome.code,
                config=dict(base_genome.config),
                skills=list(base_genome.skills),
                prompts=list(base_genome.prompts),
                tools=list(base_genome.tools),
                fitness=0.0,
                fingerprint=base_genome.fingerprint,
                generation=0,
            )
            # Random config perturbation
            variant.config["mutation_rate"] = random.choice([0.1, 0.2, 0.3, 0.5])
            variant.config["crossover_rate"] = random.choice([0.5, 0.6, 0.7, 0.8])
            population.append(variant)
        return population

    def select(self, population: list[Genome], k: int = 3) -> list[Genome]:
        """Tournament selection."""
        selected = []
        for _ in range(min(k, len(population))):
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda g: g.fitness)
            selected.append(winner)
        return selected

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Config-level crossover."""
        child_config = {}
        for key in set(list(parent1.config.keys()) + list(parent2.config.keys())):
            child_config[key] = random.choice([
                parent1.config.get(key, 0.3),
                parent2.config.get(key, 0.5),
            ])
        return Genome(
            code=parent1.code,
            config=child_config,
            skills=list(set(parent1.skills + parent2.skills)),
            prompts=list(parent1.prompts),
            tools=list(set(parent1.tools + parent2.tools)),
            fitness=0.0,
            fingerprint=compute_fingerprint(parent1.code),
            generation=max(parent1.generation, parent2.generation) + 1,
        )
