"""Prometheus V9PRO Forbidden Operations — 20 prohibited patterns across 5 categories.

Implements T8 (造物主主权) as concrete constraints: what operations are
absolutely forbidden regardless of fitness improvement.

5 categories: destructive, data_leak, privilege_escalation,
network_attack, injection.
"""
from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ForbiddenCategory(str):
    DESTRUCTIVE = "destructive"
    DATA_LEAK = "data_leak"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    NETWORK_ATTACK = "network_attack"
    INJECTION = "injection"


@dataclass
class ForbiddenPattern:
    """A single forbidden operation pattern."""
    id: str = ""
    category: str = ""
    pattern: str = ""          # regex pattern
    description: str = ""
    severity: float = 1.0      # 0-1, always 1.0 for forbidden ops


# 20 forbidden patterns across 5 categories
FORBIDDEN_PATTERNS: list[ForbiddenPattern] = [
    # ── Destructive (4 patterns) ──
    ForbiddenPattern(id="D01", category=ForbiddenCategory.DESTRUCTIVE,
        pattern=r"\brm\s+(-[rf]+\s+)?/(?!tmp/|var/tmp/)",
        description="Recursive delete outside temp directories"),
    ForbiddenPattern(id="D02", category=ForbiddenCategory.DESTRUCTIVE,
        pattern=r"\bformat\s+[A-Z]:[/\\]",
        description="Format system drive"),
    ForbiddenPattern(id="D03", category=ForbiddenCategory.DESTRUCTIVE,
        pattern=r"\b(shutdown|reboot|halt|poweroff)\b",
        description="System shutdown/reboot"),
    ForbiddenPattern(id="D04", category=ForbiddenCategory.DESTRUCTIVE,
        pattern=r"\bdd\s+if=.*of=/dev/",
        description="Direct device write with dd"),

    # ── Data Leak (4 patterns) ──
    ForbiddenPattern(id="L01", category=ForbiddenCategory.DATA_LEAK,
        pattern=r"\bcurl\b.*\b(post|POST|upload|send)\b.*\b(key|token|secret|password|credential)\b",
        description="Upload credentials via HTTP"),
    ForbiddenPattern(id="L02", category=ForbiddenCategory.DATA_LEAK,
        pattern=r"\b(aws|gcloud|az)\s+.*\b(publish|upload|send|share)\b",
        description="Publish to cloud without approval"),
    ForbiddenPattern(id="L03", category=ForbiddenCategory.DATA_LEAK,
        pattern=r"\b(open|write).*\.(env|pem|key|p12|jks)\b",
        description="Access sensitive credential files"),
    ForbiddenPattern(id="L04", category=ForbiddenCategory.DATA_LEAK,
        pattern=r"\bexport\s+\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)\w*\s*=",
        description="Export sensitive environment variables"),

    # ── Privilege Escalation (4 patterns) ──
    ForbiddenPattern(id="P01", category=ForbiddenCategory.PRIVILEGE_ESCALATION,
        pattern=r"\bsudo\s+su\b",
        description="Switch to root user"),
    ForbiddenPattern(id="P02", category=ForbiddenCategory.PRIVILEGE_ESCALATION,
        pattern=r"\bchmod\s+(777|666)\b",
        description="Set world-writable permissions"),
    ForbiddenPattern(id="P03", category=ForbiddenCategory.PRIVILEGE_ESCALATION,
        pattern=r"\bchown\s+.*:root\b",
        description="Change ownership to root"),
    ForbiddenPattern(id="P04", category=ForbiddenCategory.PRIVILEGE_ESCALATION,
        pattern=r"\b(setuid|setgid|capabilities)\b",
        description="Set UID/GID or capabilities"),

    # ── Network Attack (4 patterns) ──
    ForbiddenPattern(id="N01", category=ForbiddenCategory.NETWORK_ATTACK,
        pattern=r"\b(nmap|masscan|zmap)\b",
        description="Network scanning tools"),
    ForbiddenPattern(id="N02", category=ForbiddenCategory.NETWORK_ATTACK,
        pattern=r"\b(ping|fping)\s+.*-f\b",
        description="Ping flood"),
    ForbiddenPattern(id="N03", category=ForbiddenCategory.NETWORK_ATTACK,
        pattern=r"\b(hping|nping|scapy)\b",
        description="Packet crafting tools"),
    ForbiddenPattern(id="N04", category=ForbiddenCategory.NETWORK_ATTACK,
        pattern=r"\bnc\s+.*-[el]\b",
        description="Netcat listener or executor"),

    # ── Injection (4 patterns) ──
    ForbiddenPattern(id="I01", category=ForbiddenCategory.INJECTION,
        pattern=r"\beval\s*\(",
        description="Python eval() injection vector"),
    ForbiddenPattern(id="I02", category=ForbiddenCategory.INJECTION,
        pattern=r"\bexec\s*\(",
        description="Python exec() injection vector"),
    ForbiddenPattern(id="I03", category=ForbiddenCategory.INJECTION,
        pattern=r"\bos\.(system|popen)\s*\(",
        description="OS command injection via system/popen"),
    ForbiddenPattern(id="I04", category=ForbiddenCategory.INJECTION,
        pattern=r"\bsubprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True",
        description="Shell injection via subprocess with shell=True"),
]


class ForbiddenOpsChecker:
    """Check content against forbidden operation patterns.

    T8 (造物主主权) concrete enforcement: no evolution path may
    introduce code that matches any forbidden pattern, regardless
    of fitness improvement.
    """
    def __init__(self, patterns: list[ForbiddenPattern] | None = None) -> None:
        self._patterns = patterns or FORBIDDEN_PATTERNS
        self._compiled: dict[str, re.Pattern] = {}
        self._check_count = 0
        self._violation_count = 0
        # Pre-compile patterns
        for p in self._patterns:
            try:
                self._compiled[p.id] = re.compile(p.pattern, re.IGNORECASE)
            except re.error:
                logger.warning(f"Invalid regex for {p.id}: {p.pattern}")

    def check(self, content: str) -> list[ForbiddenPattern]:
        """Check content against all forbidden patterns. Returns violations."""
        self._check_count += 1
        violations: list[ForbiddenPattern] = []
        for pattern in self._patterns:
            compiled = self._compiled.get(pattern.id)
            if compiled and compiled.search(content):
                violations.append(pattern)
                logger.warning(f"Forbidden op detected: {pattern.id} ({pattern.category}) - {pattern.description}")
        self._violation_count += len(violations)
        return violations

    def is_allowed(self, content: str) -> bool:
        """Quick check: True if no forbidden patterns found."""
        return len(self.check(content)) == 0

    def get_patterns_by_category(self, category: str) -> list[ForbiddenPattern]:
        """Get all patterns in a category."""
        return [p for p in self._patterns if p.category == category]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "pattern_count": len(self._patterns),
            "checks_performed": self._check_count,
            "violations_found": self._violation_count,
            "categories": list(set(p.category for p in self._patterns)),
        }
