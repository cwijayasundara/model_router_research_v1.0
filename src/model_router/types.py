"""Core data types: model specs, classifications, routing decisions, responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    """Coarse capability/cost band. Concrete models are attached in the catalog."""

    FAST = "fast"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

    @classmethod
    def order(cls) -> tuple["Tier", ...]:
        return (cls.FAST, cls.BALANCED, cls.PERFORMANCE)

    @classmethod
    def next(cls, tier: "Tier") -> "Tier":
        tiers = cls.order()
        return tiers[min(tiers.index(tier) + 1, len(tiers) - 1)]

    @classmethod
    def strongest(cls, a: "Tier", b: "Tier") -> "Tier":
        return a if cls.order().index(a) >= cls.order().index(b) else b


@dataclass(slots=True)
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(slots=True)
class ModelSpec:
    """One routable model entry. Everything is data so the catalog is user-editable."""

    id: str
    provider: str
    api: str = "openai-chat"  # "openai-chat" | "anthropic" | "jev-evaluate"
    kind: str = "chat"  # "chat" | "classifier" | "image" | ...
    tier: Tier = Tier.BALANCED
    base_url: str | None = None
    api_key_env: str | None = None
    input_price: float = 0.0  # USD per 1M input tokens
    output_price: float = 0.0  # USD per 1M output tokens
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    capabilities: frozenset[str] = frozenset()  # {"tools","vision","json","reasoning",...}
    intelligence: float = 0.0  # relative quality score, 0-10 (Artificial-Analysis-style)
    aliases: tuple[str, ...] = ()
    notes: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.provider, self.id)

    def label(self) -> str:
        return f"{self.provider}/{self.id}"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1e6) * self.input_price + (output_tokens / 1e6) * self.output_price

    def expected_cost(self, est_input_tokens: int = 1_000, est_output_tokens: int = 256) -> float:
        return self.cost(est_input_tokens, est_output_tokens)


@dataclass(slots=True)
class TaskClassification:
    """Typed result of classifying a task (mirrors the Jev demo's three questions)."""

    tier: Tier = Tier.BALANCED
    tier_confidence: float = 0.5  # probability of the tier choice
    complexity: float = 1.0  # 0=mechanical .. 3=high consequence
    needs_planning: float = 0.0  # probability the task needs up-front planning
    task_type: str = "other"  # code_change | question | summarize | extract | ...
    required_capabilities: frozenset[str] = frozenset()
    est_input_tokens: int = 0
    est_output_tokens: int = 512
    classifier: str = ""
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    """Which physical model was dispatched, why, and what was considered."""

    model: ModelSpec
    tier: Tier
    reason: str = "classified"  # classified | sticky | retry | escalated | implementation | ...
    events: list[str] = field(default_factory=list)  # human-readable audit trail
    classification: TaskClassification | None = None
    state: Any = None  # policy/session state (e.g. PhasedState)
    sticky: bool = False
    alternatives: list[ModelSpec] = field(default_factory=list)


@dataclass(slots=True)
class Response:
    """Result of a full classify -> route -> generate round trip."""

    text: str
    provider: str
    model_id: str
    decision: RouteDecision
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    stop_reason: str = "stop"
    raw: Any = None


class ConfigError(Exception):
    """Missing key, unknown model, or otherwise invalid configuration."""


class ProviderError(Exception):
    """Upstream provider failed; carries status for retry decisions."""

    def __init__(self, message: str = "", *, status: int = 0, body: str = ""):
        super().__init__(message or f"provider error (status={status})")
        self.status = status
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status in {408, 409, 425, 429} or self.status >= 500

    @property
    def context_overflow(self) -> bool:
        b = self.body.lower()
        return self.status in (400, 413) and (
            "context length" in b or "maximum context" in b or "too many tokens" in b
        )


class ClassifierUnavailable(Exception):
    """The classifier backend could not be reached; routers should fall back."""
