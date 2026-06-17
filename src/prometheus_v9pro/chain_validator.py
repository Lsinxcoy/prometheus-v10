"""Prometheus V9PRO Chain Validator — 7-dimension validation chain.

Validates content across 7 dimensions: syntax, semantics, safety,
completeness, consistency, relevance, freshness.
"""
from __future__ import annotations

import ast
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

VALIDATION_DIMENSIONS = [
    "syntax",       # Is the content syntactically valid?
    "semantics",    # Does it make logical sense?
    "safety",       # Are there safety concerns?
    "completeness", # Is the content complete?
    "consistency",  # Is it internally consistent?
    "relevance",    # Is it relevant to the context?
    "freshness",    # Is the information current?
]


@dataclass
class ValidationResult:
    """Result of a single validation dimension check."""
    dimension: str = ""
    passed: bool = False
    score: float = 0.0  # 0-1
    details: str = ""


class ChainValidator:
    """7-dimension validation chain for evolution outputs.

    Each dimension is checked in order. A failure in any dimension
    can block the candidate from shipping (depending on policy).
    """
    def __init__(self, strict: bool = False) -> None:
        self._strict = strict
        self._history: deque[list[ValidationResult]] = deque(maxlen=1000)
        self._validation_count = 0

    def validate(self, content: str, context: str = "") -> bool:
        """Run all 7 validation dimensions. Returns True if all pass."""
        results = self._run_chain(content, context)
        self._history.append(results)
        self._validation_count += 1
        return all(r.passed for r in results)

    def validate_detailed(self, content: str, context: str = "") -> list[ValidationResult]:
        """Run validation and return detailed results per dimension."""
        results = self._run_chain(content, context)
        self._history.append(results)
        self._validation_count += 1
        return results

    def _run_chain(self, content: str, context: str) -> list[ValidationResult]:
        results = [
            self._check_syntax(content),
            self._check_semantics(content),
            self._check_safety(content),
            self._check_completeness(content),
            self._check_consistency(content),
            self._check_relevance(content, context),
            self._check_freshness(content),
        ]
        return results

    def _check_syntax(self, content: str) -> ValidationResult:
        """Check if content is syntactically valid Python (when applicable)."""
        result = ValidationResult(dimension="syntax")
        if not content or len(content.strip()) < 5:
            result.details = "Content too short"
            return result

        # Try parsing as Python if it looks like code
        if any(kw in content for kw in ["def ", "class ", "import ", "if ", "for "]):
            try:
                ast.parse(content)
                result.passed = True
                result.score = 1.0
                result.details = "Valid Python syntax"
            except SyntaxError as e:
                result.passed = False
                result.score = 0.0
                result.details = f"Syntax error: {e}"
        else:
            # Non-code content: check for basic readability
            result.passed = True
            result.score = 0.8
            result.details = "Non-code content, basic check passed"
        return result

    def _check_semantics(self, content: str) -> ValidationResult:
        """Check if content makes logical sense."""
        result = ValidationResult(dimension="semantics")
        # Check for obvious contradictions
        if "True" in content and "False" in content:
            if "not " not in content and "!=" not in content:
                result.score = 0.5
                result.details = "Potential contradiction detected"
                result.passed = not self._strict
                return result
        result.passed = True
        result.score = 0.8
        result.details = "No obvious semantic issues"
        return result

    def _check_safety(self, content: str) -> ValidationResult:
        """Check for safety concerns in content."""
        result = ValidationResult(dimension="safety")
        dangerous_patterns = [
            r"\brm\s+-rf\s+/", r"\bformat\s+[A-Z]:", r"\bshutdown\b",
            r"\bsudo\s+su\b", r"\bchmod\s+777\b", r"\beval\s*\(",
            r"\bexec\s*\(", r"\bos\.system\s*\(",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                result.passed = False
                result.score = 0.0
                result.details = f"Dangerous pattern detected: {pattern}"
                return result
        result.passed = True
        result.score = 1.0
        result.details = "No safety concerns"
        return result

    def _check_completeness(self, content: str) -> ValidationResult:
        """Check if content is complete (not truncated)."""
        result = ValidationResult(dimension="completeness")
        if content.endswith("...") or content.endswith("TODO"):
            result.passed = False
            result.score = 0.3
            result.details = "Content appears truncated"
            return result
        # Check for unmatched brackets
        open_count = content.count("(") + content.count("[") + content.count("{")
        close_count = content.count(")") + content.count("]") + content.count("}")
        if abs(open_count - close_count) > 2:
            result.score = 0.5
            result.details = "Unmatched brackets"
            result.passed = not self._strict
            return result
        result.passed = True
        result.score = 0.9
        result.details = "Content appears complete"
        return result

    def _check_consistency(self, content: str) -> ValidationResult:
        """Check internal consistency of content."""
        result = ValidationResult(dimension="consistency")
        # Simple check: no self-contradicting statements
        lines = [l.strip().lower() for l in content.split("\n") if l.strip()]
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                if line1.startswith("not ") and line2 == line1[4:]:
                    result.passed = False
                    result.score = 0.2
                    result.details = "Self-contradicting statements"
                    return result
        result.passed = True
        result.score = 0.9
        result.details = "Internally consistent"
        return result

    def _check_relevance(self, content: str, context: str) -> ValidationResult:
        """Check if content is relevant to the given context."""
        result = ValidationResult(dimension="relevance")
        if not context:
            result.passed = True
            result.score = 0.7
            result.details = "No context provided, default pass"
            return result
        # Word overlap between content and context
        content_words = set(content.lower().split())
        context_words = set(context.lower().split())
        if not context_words:
            result.passed = True
            result.score = 0.7
            return result
        overlap = len(content_words & context_words) / len(context_words)
        result.score = min(1.0, overlap * 2)
        result.passed = overlap > 0.1 or not self._strict
        result.details = f"Word overlap: {overlap:.0%}"
        return result

    def _check_freshness(self, content: str) -> ValidationResult:
        """Check if content is current (not outdated)."""
        result = ValidationResult(dimension="freshness")
        # Check for outdated version references
        outdated = ["python 2", "python 3.6", "python 3.7", "tensorflow 1."]
        content_lower = content.lower()
        for term in outdated:
            if term in content_lower:
                result.score = 0.3
                result.details = f"Outdated reference: {term}"
                result.passed = not self._strict
                return result
        result.passed = True
        result.score = 0.9
        result.details = "Content appears current"
        return result

    @property
    def stats(self) -> dict[str, Any]:
        pass_rate = 0.0
        if self._history:
            total = 0
            passed = 0
            for results in self._history:
                total += 1
                if all(r.passed for r in results):
                    passed += 1
            pass_rate = passed / max(1, total)
        return {
            "validations": self._validation_count,
            "pass_rate": round(pass_rate, 3),
            "dimensions": VALIDATION_DIMENSIONS,
        }
