"""Prometheus V9 Hermes LLM Adapter — httpx client with retry, rate limiting, and model chain fallback.

This is the bridge between Prometheus and real LLM APIs. Without it:
- Nuwa design generation is only heuristic
- Dream ReACT has no actual reasoning
- L6 Prompt rewrite is only template substitution
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    base_url: str = "https://api.xiaomimo.com/v1"
    api_key: str = ""
    primary_model: str = "mimo-v2.5-pro"
    fallback_models: list[str] = field(default_factory=lambda: ["mimo-v2.5", "qwen3-235b"])
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_retries: int = 0
    estimated_cost: float = 0.0


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.consume(tokens):
                return True
            time.sleep(0.5)
        return False


class HermesLLMAdapter:
    """LLM client with retry, rate limiting, and model chain fallback.

    Implements T6 (adapt > novel): if primary model fails, fall back to cheaper models.
    Implements T5 (anti-fragile): network failures → retry, not crash.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config or LLMConfig()
        self._usage = TokenUsage()
        self._rpm_bucket = TokenBucket(rate=self._config.rate_limit_rpm / 60.0, capacity=self._config.rate_limit_rpm)
        self._tpm_bucket = TokenBucket(rate=self._config.rate_limit_tpm / 60.0, capacity=self._config.rate_limit_tpm)
        self._request_history: deque[dict] = deque(maxlen=1000)
        self._model_health: dict[str, dict] = {}
        self._httpx_client = None

        # Initialize model health tracking
        models = [self._config.primary_model] + self._config.fallback_models
        for m in models:
            self._model_health[m] = {"successes": 0, "failures": 0, "last_success": 0.0, "consecutive_failures": 0}

        if HAS_HTTPX and self._config.api_key:
            self._httpx_client = httpx.Client(
                base_url=self._config.base_url,
                headers={"Authorization": f"Bearer {self._config.api_key}"},
                timeout=self._config.timeout_seconds,
            )
            logger.info(f"LLM adapter initialized with httpx (primary={self._config.primary_model})")
        else:
            logger.info("LLM adapter in heuristic-only mode (no API key or httpx)")

    @property
    def has_real_llm(self) -> bool:
        return self._httpx_client is not None

    @property
    def usage(self) -> TokenUsage:
        return self._usage

    def _select_model(self) -> str:
        """Select best available model based on health tracking."""
        models = [self._config.primary_model] + self._config.fallback_models
        # Filter out models with 3+ consecutive failures
        available = [m for m in models if self._model_health[m]["consecutive_failures"] < 3]
        if not available:
            # All models struggling — reset and try primary
            for m in models:
                self._model_health[m]["consecutive_failures"] = 0
            available = [self._config.primary_model]
        return available[0]

    def _call_llm(self, model: str, prompt: str, system: str = "",
                  max_tokens: int | None = None) -> dict[str, Any]:
        """Make a real LLM API call."""
        if not self._httpx_client:
            return {"error": "no_llm_client", "content": ""}

        estimated_tokens = len(prompt) // 4 + (max_tokens or self._config.max_tokens)
        if not self._rpm_bucket.wait_and_consume(1.0, timeout=5.0):
            return {"error": "rate_limit", "content": ""}
        if not self._tpm_bucket.wait_and_consume(estimated_tokens, timeout=5.0):
            return {"error": "rate_limit", "content": ""}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": self._config.temperature,
        }
        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})

        for attempt in range(self._config.max_retries):
            try:
                response = self._httpx_client.post("/chat/completions", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    tokens = data.get("usage", {})
                    self._usage.prompt_tokens += tokens.get("prompt_tokens", 0)
                    self._usage.completion_tokens += tokens.get("completion_tokens", 0)
                    self._usage.total_tokens += tokens.get("total_tokens", 0)
                    self._usage.total_requests += 1
                    self._model_health[model]["successes"] += 1
                    self._model_health[model]["last_success"] = time.time()
                    self._model_health[model]["consecutive_failures"] = 0
                    return {"content": content, "model": model, "tokens": tokens}
                elif response.status_code == 429:
                    time.sleep(self._config.retry_base_delay * (2 ** attempt))
                    continue
                else:
                    self._model_health[model]["failures"] += 1
                    self._model_health[model]["consecutive_failures"] += 1
                    return {"error": f"http_{response.status_code}", "content": ""}
            except Exception as e:
                self._model_health[model]["failures"] += 1
                self._model_health[model]["consecutive_failures"] += 1
                if attempt < self._config.max_retries - 1:
                    time.sleep(self._config.retry_base_delay * (2 ** attempt))
                else:
                    return {"error": str(e), "content": ""}

        self._usage.total_failures += 1
        return {"error": "max_retries", "content": ""}

    def generate(self, prompt: str, system: str = "", max_tokens: int | None = None,
                 heuristic_fn: Callable | None = None) -> str:
        """Generate text with LLM or heuristic fallback.

        If real LLM is available, use it. Otherwise fall back to heuristic.
        This implements T5 (anti-fragile): no LLM → heuristic, not crash.
        """
        if self.has_real_llm:
            model = self._select_model()
            result = self._call_llm(model, prompt, system, max_tokens)
            if result.get("content"):
                return result["content"]
            # LLM failed — fall back to heuristic
            logger.warning(f"LLM call failed ({result.get('error')}), using heuristic fallback")

        if heuristic_fn:
            return heuristic_fn(prompt)
        return ""

    def embed(self, text: str) -> list[float] | None:
        """Generate embeddings (if available)."""
        if not self._httpx_client:
            return None
        try:
            response = self._httpx_client.post("/embeddings", json={
                "model": "text-embedding-3-small",
                "input": text,
            })
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        return None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "has_real_llm": self.has_real_llm,
            "usage": {
                "total_requests": self._usage.total_requests,
                "total_failures": self._usage.total_failures,
                "total_tokens": self._usage.total_tokens,
            },
            "model_health": dict(self._model_health),
        }
