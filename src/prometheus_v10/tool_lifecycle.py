"""Prometheus V10 Tool Lifecycle — Typed lifecycle operations for safe tool evolution.

Inspired by SkillSmith (arXiv:2606.01314) §3.1:
- 5 typed lifecycle operations: WRAP, EDIT, COMPOSE, SPLIT, RETIRE
- Atomic proposal bundles: joint skill+tool updates guaranteed to be consistent
- FreeTool (unconstrained edits) → 14.7% error rate; typed ops → 3.2%

Key difference from V9PRO:
- V9PRO organs.py: 5 fixed organs, no tool mutation capability
- V10 tool_lifecycle.py: safe, typed tool evolution with atomic guarantees
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LifecycleOp(Enum):
    """Typed tool lifecycle operations from SkillSmith §3.1."""
    WRAP = "wrap"       # Wrap existing tool with pre/post processing
    EDIT = "edit"       # Patch tool implementation (type-safe)
    COMPOSE = "compose" # Combine multiple tools into one
    SPLIT = "split"     # Split one tool into focused sub-tools
    RETIRE = "retire"   # Remove tool, transfer knowledge to anti-pattern


@dataclass
class ToolSpec:
    """Tool specification."""
    name: str = ""
    interface: str = ""         # input/output format contract
    version: str = "1.0.0"
    dependencies: list[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class LifecycleEdit:
    """One typed edit to a tool."""
    op: LifecycleOp = LifecycleOp.EDIT
    target: str = ""            # tool name
    patch: str = ""             # description of the change
    pre_hook: str | None = None # for WRAP: pre-processing logic
    post_hook: str | None = None # for WRAP: post-processing logic
    source_tools: list[str] = field(default_factory=list)  # for COMPOSE/SPLIT
    retire_reason: str = ""     # for RETIRE


@dataclass
class ProposalBundle:
    """Atomic proposal bundle: joint skill+tool updates.

    Following SkillSmith §3.1: Atomicity guarantees interdependent changes
    take effect simultaneously. Creating a new Tool without updating the
    Skill that invokes it is invalid.
    """
    skill_edits: list[dict[str, Any]] = field(default_factory=list)
    tool_edits: list[LifecycleEdit] = field(default_factory=list)
    bundle_id: str = ""
    rationale: str = ""
    vertical_attribution: str = ""   # "skill_defect" or "tool_deficiency"
    horizontal_attribution: str = "" # "single_component" or "multi_skill_conflict"


class ToolLifecycleManager:
    """Manage safe tool evolution through typed lifecycle operations."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._edit_history: list[LifecycleEdit] = []
        self._error_count: dict[str, int] = {}  # tool_name → consecutive errors
        self._max_consecutive_errors = 5

    def register_tool(self, spec: ToolSpec) -> None:
        """Register a tool in the lifecycle manager."""
        self._tools[spec.name] = spec

    def apply_edit(self, edit: LifecycleEdit) -> tuple[bool, str]:
        """Apply a typed lifecycle edit. Returns (success, reason)."""
        # Validate edit
        valid, reason = self._validate_edit(edit)
        if not valid:
            return False, reason

        # Apply the edit
        if edit.op == LifecycleOp.WRAP:
            return self._apply_wrap(edit)
        elif edit.op == LifecycleOp.EDIT:
            return self._apply_edit(edit)
        elif edit.op == LifecycleOp.COMPOSE:
            return self._apply_compose(edit)
        elif edit.op == LifecycleOp.SPLIT:
            return self._apply_split(edit)
        elif edit.op == LifecycleOp.RETIRE:
            return self._apply_retire(edit)
        else:
            return False, f"Unknown lifecycle op: {edit.op}"

    def apply_bundle(self, bundle: ProposalBundle) -> tuple[bool, str]:
        """Apply an atomic proposal bundle.

        Either all edits succeed, or none are applied.
        """
        # Phase 1: Validate all edits
        for edit in bundle.tool_edits:
            valid, reason = self._validate_edit(edit)
            if not valid:
                return False, f"Bundle validation failed for {edit.target}: {reason}"

        # Phase 2: Apply all edits (if any fail, roll back)
        applied: list[tuple[LifecycleEdit, ToolSpec | None]] = []
        for edit in bundle.tool_edits:
            # Save snapshot before edit
            snapshot = self._tools.get(edit.target)
            applied.append((edit, snapshot))

            success, reason = self.apply_edit(edit)
            if not success:
                # Roll back
                for prev_edit, prev_snapshot in reversed(applied[:-1]):
                    if prev_snapshot:
                        self._tools[prev_edit.target] = prev_snapshot
                return False, f"Bundle application failed at {edit.target}: {reason}"

        self._edit_history.extend(bundle.tool_edits)
        logger.info(f"Applied bundle '{bundle.bundle_id}': {len(bundle.tool_edits)} tool edits")
        return True, f"Bundle applied: {len(bundle.tool_edits)} edits"

    def record_tool_error(self, tool_name: str) -> None:
        """Record a tool execution error."""
        self._error_count[tool_name] = self._error_count.get(tool_name, 0) + 1

    def record_tool_success(self, tool_name: str) -> None:
        """Record a tool execution success (reset error count)."""
        self._error_count[tool_name] = 0

    def should_auto_retire(self, tool_name: str) -> bool:
        """Check if a tool should be auto-retired due to errors."""
        return self._error_count.get(tool_name, 0) >= self._max_consecutive_errors

    def get_active_tools(self) -> list[ToolSpec]:
        """Get all active tools."""
        return [t for t in self._tools.values() if t.is_active]

    def get_tool(self, name: str) -> ToolSpec | None:
        """Get a tool by name."""
        return self._tools.get(name)

    # ── Private Implementation ──────────────────────────────────

    def _validate_edit(self, edit: LifecycleEdit) -> tuple[bool, str]:
        """Validate a lifecycle edit before application."""
        if edit.op == LifecycleOp.EDIT:
            if edit.target not in self._tools:
                return False, f"Tool '{edit.target}' not found"
            if not self._tools[edit.target].is_active:
                return False, f"Tool '{edit.target}' is retired"

        elif edit.op == LifecycleOp.WRAP:
            if edit.target not in self._tools:
                return False, f"Tool '{edit.target}' not found for wrapping"

        elif edit.op == LifecycleOp.COMPOSE:
            if len(edit.source_tools) < 2:
                return False, "COMPOSE requires at least 2 source tools"
            for src in edit.source_tools:
                if src not in self._tools:
                    return False, f"Source tool '{src}' not found"

        elif edit.op == LifecycleOp.SPLIT:
            if edit.target not in self._tools:
                return False, f"Tool '{edit.target}' not found for splitting"

        elif edit.op == LifecycleOp.RETIRE:
            if edit.target not in self._tools:
                return False, f"Tool '{edit.target}' not found for retirement"

        return True, ""

    def _apply_wrap(self, edit: LifecycleEdit) -> tuple[bool, str]:
        tool = self._tools[edit.target]
        tool.version = self._increment_version(tool.version)
        logger.info(f"Wrapped tool '{edit.target}' → v{tool.version}")
        return True, f"Wrapped '{edit.target}'"

    def _apply_edit(self, edit: LifecycleEdit) -> tuple[bool, str]:
        tool = self._tools[edit.target]
        tool.version = self._increment_version(tool.version)
        logger.info(f"Edited tool '{edit.target}' → v{tool.version}: {edit.patch[:50]}")
        return True, f"Edited '{edit.target}'"

    def _apply_compose(self, edit: LifecycleEdit) -> tuple[bool, str]:
        new_name = f"composed_{'_'.join(edit.source_tools)}"
        deps = edit.source_tools
        self._tools[new_name] = ToolSpec(
            name=new_name, interface="composite", dependencies=deps
        )
        logger.info(f"Composed tools {edit.source_tools} → '{new_name}'")
        return True, f"Composed into '{new_name}'"

    def _apply_split(self, edit: LifecycleEdit) -> tuple[bool, str]:
        original = self._tools[edit.target]
        sub_a = ToolSpec(name=f"{edit.target}_a", interface=f"sub_a_of_{edit.target}")
        sub_b = ToolSpec(name=f"{edit.target}_b", interface=f"sub_b_of_{edit.target}")
        self._tools[sub_a.name] = sub_a
        self._tools[sub_b.name] = sub_b
        original.is_active = False
        logger.info(f"Split tool '{edit.target}' → '{sub_a.name}', '{sub_b.name}'")
        return True, f"Split '{edit.target}' into '{sub_a.name}', '{sub_b.name}'"

    def _apply_retire(self, edit: LifecycleEdit) -> tuple[bool, str]:
        tool = self._tools[edit.target]
        tool.is_active = False
        logger.info(f"Retired tool '{edit.target}': {edit.retire_reason}")
        return True, f"Retired '{edit.target}'"

    @staticmethod
    def _increment_version(version: str) -> str:
        try:
            parts = version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except (ValueError, IndexError):
            return version

    def stats(self) -> dict[str, Any]:
        active = [t for t in self._tools.values() if t.is_active]
        retired = [t for t in self._tools.values() if not t.is_active]
        return {
            "total_tools": len(self._tools),
            "active": len(active),
            "retired": len(retired),
            "total_edits": len(self._edit_history),
            "error_tools": {k: v for k, v in self._error_count.items() if v > 0},
        }
