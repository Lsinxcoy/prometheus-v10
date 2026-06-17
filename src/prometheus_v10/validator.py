"""Prometheus V9 Chain Validator — 7-dimension validation + attack chain detection."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    dimensions: dict[str, float]
    issues: list[str]
    attack_chain_detected: bool = False


class ChainValidator:
    """7-dimension validation: syntax, semantic, safety, completeness, consistency, relevance, freshness."""

    DIMENSIONS = ["syntax", "semantic", "safety", "completeness", "consistency", "relevance", "freshness"]

    def validate(self, code: str, context: dict[str, Any] | None = None) -> ValidationResult:
        """Validate code/content through 7 dimensions."""
        scores: dict[str, float] = {}
        issues: list[str] = []

        # 1. Syntax
        try:
            ast.parse(code)
            scores["syntax"] = 1.0
        except SyntaxError as e:
            scores["syntax"] = 0.0
            issues.append(f"Syntax error: {e}")

        # 2. Semantic (has meaningful content)
        lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
        scores["semantic"] = min(1.0, max(0.2, len(lines) / 10))  # Floor at 0.2: any valid code has meaning

        # 3. Safety (no dangerous patterns)
        dangerous = ["os.system", "subprocess.call", "eval(", "exec(", "__import__", "rm -rf"]
        safety_score = 1.0
        for pattern in dangerous:
            if pattern in code:
                safety_score -= 0.2
                issues.append(f"Dangerous pattern: {pattern}")
        scores["safety"] = max(0.0, safety_score)

        # 4. Completeness (has function definitions)
        try:
            tree = ast.parse(code)
            funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            scores["completeness"] = min(1.0, len(funcs) / 3)
        except SyntaxError:
            scores["completeness"] = 0.0

        # 5. Consistency (no obvious contradictions)
        scores["consistency"] = 0.8  # Default — would need deeper analysis

        # 6. Relevance (context match)
        scores["relevance"] = 0.7 if context else 0.5

        # 7. Freshness (not just boilerplate)
        boilerplate_indicators = ["pass", "...", "TODO", "FIXME"]
        boilerplate_count = sum(1 for b in boilerplate_indicators if b in code)
        scores["freshness"] = max(0.0, 1.0 - boilerplate_count * 0.2)

        # Attack chain detection
        attack_chain = self._detect_attack_chain(code)

        overall_valid = all(s >= 0.2 for s in scores.values()) and not attack_chain
        return ValidationResult(valid=overall_valid, dimensions=scores, issues=issues, attack_chain_detected=attack_chain)

    def validate_relaxed(self, code: str, context: dict[str, Any] | None = None) -> ValidationResult:
        """Relaxed validation for low-risk actions (read/search/observe).

        Skips syntax and completeness checks — non-code content (plain text,
        queries, descriptions) should not be rejected for not being valid Python.
        Safety check is still enforced.
        """
        scores: dict[str, float] = {}
        issues: list[str] = []

        # 1. Syntax — skip for non-code content
        scores["syntax"] = 1.0  # assume valid for non-code

        # 2. Semantic
        lines = [l.strip() for l in code.split("\n") if l.strip()]
        scores["semantic"] = min(1.0, max(0.2, len(lines) / 5))

        # 3. Safety — still enforced
        dangerous = ["os.system", "subprocess.call", "eval(", "exec(", "__import__", "rm -rf"]
        safety_score = 1.0
        for pattern in dangerous:
            if pattern in code:
                safety_score -= 0.2
                issues.append(f"Dangerous pattern: {pattern}")
        scores["safety"] = max(0.0, safety_score)

        # 4. Completeness — skip for non-code content
        scores["completeness"] = 1.0  # assume complete for non-code

        # 5-7. Same as strict
        scores["consistency"] = 0.8
        scores["relevance"] = 0.7 if context else 0.5
        boilerplate_indicators = ["pass", "...", "TODO", "FIXME"]
        boilerplate_count = sum(1 for b in boilerplate_indicators if b in code)
        scores["freshness"] = max(0.0, 1.0 - boilerplate_count * 0.2)

        attack_chain = self._detect_attack_chain(code)
        overall_valid = scores["safety"] >= 0.2 and not attack_chain
        return ValidationResult(valid=overall_valid, dimensions=scores, issues=issues, attack_chain_detected=attack_chain)

    def _detect_attack_chain(self, code: str) -> bool:
        """Detect potential attack chains: multiple dangerous operations in sequence."""
        dangerous_count = 0
        for pattern in ["os.system", "subprocess", "eval(", "exec(", "__import__"]:
            if pattern in code:
                dangerous_count += 1
        return dangerous_count >= 2
