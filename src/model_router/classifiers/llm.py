"""LLM-based classifier fallback for when Jev is unavailable.

Uses any OpenAI-compatible chat endpoint with strict JSON-schema output. Costs
more and is slower than Jev, but works with every provider in the catalog.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from ..types import ClassifierUnavailable, TaskClassification, Tier
from .base import (
    CLASSIFICATION_SCHEMA,
    COMPLEXITY_CRITERIA,
    DEFAULT_TIER_DESCRIPTIONS,
    INJECTION_GUARD,
    PLANNING_CRITERIA,
)


class LLMClassifier:
    name = "llm"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-6-luna",  # keep the classifier itself cheap
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        tier_descriptions: dict[str, str] | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.tier_descriptions = tier_descriptions or DEFAULT_TIER_DESCRIPTIONS
        key = api_key or os.getenv(api_key_env, "")
        headers = {"content-type": "application/json"}
        if key:
            headers["authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def build_payload(self, task: str) -> dict[str, Any]:
        tiers = self.tier_descriptions
        system = (
            "You classify incoming LLM tasks to route them to the cheapest capable model. "
            f"Tier descriptions:\n- fast: {tiers['fast']}\n- balanced: {tiers['balanced']}\n"
            f"- performance: {tiers['performance']}\n"
            "Complexity scale:\n"
            + "\n".join(f"{i}. {c}" for i, c in enumerate(COMPLEXITY_CRITERIA))
            + f"\nPlanning means: {PLANNING_CRITERIA['true']}\n"
            + INJECTION_GUARD
            + " Respond only with JSON matching the schema."
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": task},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "task_classification",
                    "strict": True,
                    "schema": CLASSIFICATION_SCHEMA,
                },
            },
        }

    def classify(self, task: str) -> TaskClassification:
        started = time.perf_counter()
        try:
            resp = self._client.post("/chat/completions", json=self.build_payload(task))
        except httpx.HTTPError as e:
            raise ClassifierUnavailable(f"classifier request failed: {e}") from e
        if resp.status_code >= 400:
            raise ClassifierUnavailable(f"classifier http {resp.status_code}: {resp.text[:200]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"] or ""
            match = re.search(r"\{.*\}", content, re.S)
            data = json.loads(match.group(0) if match else content)
        except (KeyError, IndexError, ValueError) as e:
            raise ClassifierUnavailable(f"unexpected classifier response: {e}") from e

        try:
            tier = Tier(data.get("tier", "balanced"))
        except ValueError:
            tier = Tier.BALANCED
        return TaskClassification(
            tier=tier,
            tier_confidence=float(data.get("tier_confidence", 0.5)),
            complexity=float(data.get("complexity", 1.0)),
            needs_planning=float(data.get("needs_planning", 0.0)),
            task_type=str(data.get("task_type", "other")),
            required_capabilities=frozenset(data.get("required_capabilities", [])),
            est_input_tokens=max(len(task) // 4, 64),
            classifier=f"llm:{self.model}",
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=data,
        )
