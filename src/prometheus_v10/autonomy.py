"""Prometheus V9 Autonomy — 5-level autonomy with real execution gating + T8 creator sovereignty."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AutonomyLevel(enum.Enum):
    L0_OBSERVE = 0      # Only observe, no action
    L1_SUGGEST = 1      # Suggest actions, human must approve all
    L2_ASSIST = 2       # Execute safe actions, ask for risky ones
    L3_AUTONOMOUS = 3   # Execute most actions, report outcomes
    L4_FULL_AUTO = 4    # Full autonomy (never for code changes — T8)


@dataclass
class AutonomyDecision:
    level: AutonomyLevel
    action: str
    auto_execute: bool
    reason: str
    requires_approval: bool = False
    budget_consumed: float = 0.0


class AutonomyManager:
    """5-level autonomy with real execution gating. T8: code changes always need approval."""

    # Risk classification per action type
    ACTION_RISK = {
        "read": 0, "search": 0, "observe": 0,
        "suggest": 1, "analyze": 1,
        "write_config": 2, "modify_config": 2, "add_memory": 2,
        "write_code": 3, "modify_code": 3, "delete_code": 3,
        "system_command": 4, "network_access": 4,
    }

    def __init__(self, default_level: AutonomyLevel = AutonomyLevel.L2_ASSIST, budget_tokens: int = 4000, budget_time: int = 240) -> None:
        self._default_level = default_level
        self._current_level = default_level
        self._budget_tokens = budget_tokens
        self._budget_time = budget_time
        self._tokens_consumed = 0
        self._time_consumed = 0.0
        self._action_log: list[AutonomyDecision] = []

    def decide(self, action: str, context: dict[str, Any] | None = None) -> AutonomyDecision:
        """Decide whether action can auto-execute based on autonomy level + risk + budget."""
        risk = self.ACTION_RISK.get(action, 2)
        # T8: code changes ALWAYS require approval regardless of level
        requires_approval = action in ("write_code", "modify_code", "delete_code")
        # Budget check
        budget_ok = self._tokens_consumed < self._budget_tokens and self._time_consumed < self._budget_time
        # Level-based auto-execute decision
        auto_execute = False
        if requires_approval:
            auto_execute = False  # T8 override
        elif not budget_ok:
            auto_execute = False
        elif risk <= self._current_level.value:
            auto_execute = True
        decision = AutonomyDecision(
            level=self._current_level,
            action=action,
            auto_execute=auto_execute,
            reason=f"risk={risk}, level={self._current_level.value}, budget_ok={budget_ok}, T8={requires_approval}",
            requires_approval=requires_approval,
            budget_consumed=self._tokens_consumed / max(1, self._budget_tokens),
        )
        self._action_log.append(decision)
        return decision

    def set_level(self, level: AutonomyLevel) -> None:
        self._current_level = level

    def consume_budget(self, tokens: int = 0, time_seconds: float = 0.0) -> None:
        self._tokens_consumed += tokens
        self._time_consumed += time_seconds

    @property
    def current_level(self) -> AutonomyLevel:
        return self._current_level

    @property
    def budget_remaining(self) -> dict[str, float]:
        return {
            "tokens": max(0, self._budget_tokens - self._tokens_consumed),
            "time": max(0, self._budget_time - self._time_consumed),
        }
