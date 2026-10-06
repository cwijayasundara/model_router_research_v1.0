"""Fake provider for offline tests, demos, and dry runs."""

from __future__ import annotations

from typing import Any, Callable

from ..types import Message, ModelSpec, ProviderError
from .base import ProviderResponse


class FakeProvider:
    """Recorded, deterministic provider.

    responses: model_id -> fixed text or callable(messages) -> text
    fail_first_n: raise a retryable ProviderError on the first N calls (tests retry/escalation)
    """

    api = "any"

    def __init__(
        self,
        *,
        responses: dict[str, str | Callable[[list[Message]], str]] | None = None,
        default: str = "ok: done",
        fail_first_n: int = 0,
        usage: tuple[int, int] = (120, 60),
        error_status: int = 429,
    ) -> None:
        self.responses = responses or {}
        self.default = default
        self.fail_first_n = fail_first_n
        self.usage = usage
        self.error_status = error_status
        self.calls: list[tuple[str, str, list[Message]]] = []
        self._call_count = 0

    def complete(self, spec: ModelSpec, messages: list[Message], **params: Any) -> ProviderResponse:
        self.calls.append((spec.provider, spec.id, list(messages)))
        self._call_count += 1
        if self._call_count <= self.fail_first_n:
            raise ProviderError("fake rate limited", status=self.error_status, body="rate limited")
        resp = self.responses.get(spec.id, self.default)
        text = resp(messages) if callable(resp) else resp
        return ProviderResponse(
            text=text,
            input_tokens=self.usage[0],
            output_tokens=self.usage[1],
            raw={"fake": True},
        )
