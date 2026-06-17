"""Prometheus V9 Crossover — Multi-genome crossover recombination."""
from __future__ import annotations

import copy
import logging
import random
from typing import Any

from prometheus_v9pro.schema import Genome

logger = logging.getLogger(__name__)


class CrossoverOperator:
    """Crossover recombination for genetic evolution.

    Methods:
    1. Uniform: Randomly select config from either parent
    2. Single-point: Split config at one point
    3. Blend: Weighted average of numeric config values
    """

    def uniform_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Uniform crossover: each config key from random parent."""
        child_config = {}
        all_keys = set(parent1.config) | set(parent2.config)
        for key in all_keys:
            child_config[key] = random.choice([
                parent1.config.get(key), parent2.config.get(key)
            ])
        return Genome(
            code=random.choice([parent1.code, parent2.code]),
            config=child_config, skills=[], prompts=[], tools=[],
        )

    def single_point_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Single-point crossover on config keys."""
        keys = sorted(set(parent1.config) | set(parent2.config))
        if not keys:
            return copy.deepcopy(parent1)
        point = random.randint(1, max(1, len(keys) - 1))
        child_config = {}
        for i, key in enumerate(keys):
            if i < point:
                child_config[key] = parent1.config.get(key)
            else:
                child_config[key] = parent2.config.get(key)
        return Genome(code=parent1.code, config=child_config, skills=[], prompts=[], tools=[])

    def blend_crossover(self, parent1: Genome, parent2: Genome, alpha: float = 0.5) -> Genome:
        """Blend crossover: weighted average of numeric values."""
        child_config = {}
        all_keys = set(parent1.config) | set(parent2.config)
        for key in all_keys:
            v1, v2 = parent1.config.get(key), parent2.config.get(key)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                child_config[key] = v1 * alpha + v2 * (1 - alpha)
            else:
                child_config[key] = random.choice([v1, v2])
        return Genome(code=parent1.code, config=child_config, skills=[], prompts=[], tools=[])
