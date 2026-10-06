"""Provider interface and response type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import Message, ModelSpec


@dataclass(slots=True)
class ProviderResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "stop"
    raw: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Provider(Protocol):
    def complete(self, spec: ModelSpec, messages: list[Message], **params: Any) -> ProviderResponse:
        ...
