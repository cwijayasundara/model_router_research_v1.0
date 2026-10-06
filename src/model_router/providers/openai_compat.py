"""OpenAI-compatible chat provider.

One adapter covers OpenAI, Fireworks.ai, Baseten, OpenRouter, DeepSeek official,
Moonshot, Google Gemini's OpenAI-compatible endpoint, vLLM, SGLang, llama.cpp,
Ollama - anywhere that speaks /chat/completions.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..types import ConfigError, Message, ModelSpec, ProviderError
from .base import ProviderResponse

DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "baseten": "https://inference.baseten.co/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}


class OpenAICompatProvider:
    api = "openai-chat"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        name: str = "openai-compat",
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.name = name
        headers = {"content-type": "application/json"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def from_spec(cls, spec: ModelSpec, **kwargs: Any) -> "OpenAICompatProvider":
        base_url = spec.base_url or DEFAULT_BASE_URLS.get(spec.provider)
        if not base_url:
            raise ConfigError(
                f"no base_url for provider {spec.provider!r}; set it in the catalog entry"
            )
        key = os.environ.get(spec.api_key_env or "", "")
        if spec.api_key_env and not key:
            raise ConfigError(f"missing API key: set {spec.api_key_env}")
        return cls(base_url=base_url, api_key=key, name=spec.provider, **kwargs)

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
        payload: dict[str, Any] = {
            "model": spec.id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if extra:
            payload.update(extra)

        resp = self._client.post("/chat/completions", json=payload)
        if resp.status_code >= 400:
            raise ProviderError(
                f"{spec.provider} http {resp.status_code}", status=resp.status_code, body=resp.text[:500]
            )
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=message.get("content") or "",
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            stop_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )
