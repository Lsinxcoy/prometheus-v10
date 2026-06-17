"""Prometheus V9 Checkpoint — Evolution state save/restore for crash recovery."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus_v10.schema import Genome

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """A checkpoint of evolution state."""
    id: str = ""
    generation: int = 0
    genome: Genome | None = None
    fitness: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class CheckpointManager:
    """Save and restore evolution state for crash recovery.

    Implements T5 (anti-fragile): crash → restore from checkpoint, not start over.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: list[CheckpointData] = []

    def save(self, genome: Genome, generation: int = 0,
             metadata: dict | None = None) -> str:
        """Save a checkpoint with atomic write (V8 persistence.py enhancement).

        Atomic write: write to temp file first, then rename. Prevents
        corruption if process crashes mid-write.
        """
        cp_id = f"cp_{generation}_{int(time.time())}"
        cp = CheckpointData(
            id=cp_id, generation=generation,
            genome=genome, fitness=genome.fitness,
            metadata=metadata or {},
        )
        self._checkpoints.append(cp)

        # Persist to disk with atomic write
        data = {
            "id": cp_id, "generation": generation,
            "fitness": genome.fitness, "timestamp": cp.timestamp,
            "genome_config": genome.config, "genome_code": genome.code,
            "metadata": metadata or {},
        }
        path = self._dir / f"{cp_id}.json"
        temp_path = self._dir / f"{cp_id}.json.tmp"
        try:
            temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.rename(path)
            logger.info(f"Checkpoint saved: {cp_id} (gen={generation}, fitness={genome.fitness:.3f})")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        return cp_id

    def restore(self, cp_id: str | None = None) -> CheckpointData | None:
        """Restore from checkpoint. If cp_id is None, restore latest."""
        if cp_id:
            path = self._dir / f"{cp_id}.json"
        else:
            # Find latest
            paths = sorted(self._dir.glob("cp_*.json"))
            if not paths:
                return None
            path = paths[-1]

        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        genome = Genome(
            code=data.get("genome_code", ""),
            config=data.get("genome_config", {}),
            skills=[], prompts=[], tools=[],
        )
        genome.fitness = data.get("fitness", 0.0)

        return CheckpointData(
            id=data["id"], generation=data["generation"],
            genome=genome, fitness=data["fitness"],
            timestamp=data.get("timestamp", 0),
            metadata=data.get("metadata", {}),
        )

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        cps = []
        for path in sorted(self._dir.glob("cp_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cps.append({"id": data["id"], "generation": data["generation"], "fitness": data["fitness"]})
            except Exception:
                pass
        return cps

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_checkpoints": len(self._checkpoints),
            "disk_checkpoints": len(list(self._dir.glob("cp_*.json"))),
            "latest_fitness": self._checkpoints[-1].fitness if self._checkpoints else 0,
        }
