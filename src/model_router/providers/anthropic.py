"""Anthropic Messages API provider (for Claude models)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..types import ConfigError, Message, ModelSpec, ProviderError
from .base import ProviderResponse


class AnthropicProvider:
    api = "anthropic"

    def __init__(
        self,
        *,
        base_url: str = "https://api.anthropic.com",
        api_key: str = "",
        version: str = "2023-06-01",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": version,
            },
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_spec(cls, spec: ModelSpec, **kwargs: Any) -> "AnthropicProvider":
        key = os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY", "")
        if not key:
            raise ConfigError("missing API key: set ANTHROPIC_API_KEY")
        return cls(api_key=key, **kwargs)

    def complete(
        self,
        spec: ModelSpec,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        extra: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> ProviderResponse:
        system = "\n".join(m.content for m in messages if m.role == "system") or None
        payload: dict[str, Any] = {
            "model": spec.id,
            "max_tokens": max_tokens or spec.max_output_tokens,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role in ("user", "assistant")
            ],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if extra:
            payload.update(extra)

        resp = self._client.post("/v1/messages", json=payload)
        if resp.status_code >= 400:
            raise ProviderError(
                f"anthropic http {resp.status_code}", status=resp.status_code, body=resp.text[:500]
            )
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            stop_reason=data.get("stop_reason", "stop"),
            raw=data,
        )
