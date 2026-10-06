import json

import httpx
import pytest

from model_router import (
    ClassifierUnavailable,
    ConfigError,
    HeuristicClassifier,
    JevClassifier,
    LLMClassifier,
    Tier,
)


def test_heuristic_mechanical_rename_is_fast():
    c = HeuristicClassifier().classify(
        "Rename ctaText to buttonLabel in the pricing card component and update its direct references."
    )
    assert c.tier == Tier.FAST
    assert c.complexity == 0.0


def test_heuristic_migration_is_high_complexity():
    c = HeuristicClassifier().classify(
        "Design a zero-downtime migration that splits the customers table into accounts "
        "and people, preserves API compatibility, and includes rollback and backfill plans."
    )
    assert c.complexity >= 2.0
    assert c.needs_planning >= 0.6
    assert c.classifier == "heuristic"


def test_heuristic_security_trace():
    c = HeuristicClassifier().classify(
        "Trace an intermittent authentication failure across middleware, session storage, "
        "and OAuth callbacks. Propose a fix without weakening replay or CSRF protection."
    )
    assert c.tier == Tier.PERFORMANCE
    assert c.needs_planning >= 0.6


def test_heuristic_detects_tool_and_vision_requirements():
    c = HeuristicClassifier().classify(
        "Add the missing aria-label to the close button and run its existing component test."
    )
    assert "tools" in c.required_capabilities
    v = HeuristicClassifier().classify("Look at this screenshot of the settings page and fix the layout.")
    assert "vision" in v.required_capabilities


def test_heuristic_plain_question_is_fast():
    c = HeuristicClassifier().classify("What is the capital of France?")
    assert c.tier == Tier.FAST
    assert c.task_type == "question"


JEV_FIXTURE = {
    "answers": {
        "tier": {
            "choice": "performance",
            "probabilities": {"performance": 0.91, "balanced": 0.09},
        },
        "complexity": {"score": 2.4},
        "needs_planning": {"probability": 0.72, "value": True},
    }
}


def test_jev_gateway_backend_payload_guard_zdr_then_parses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=JEV_FIXTURE)

    jev = JevClassifier(backend="gateway", api_key="test-key", transport=httpx.MockTransport(handler))
    c = jev.classify("Trace an intermittent authentication failure across the system.")

    assert c.tier == Tier.PERFORMANCE
    assert c.tier_confidence == pytest.approx(0.91)
    assert c.complexity == pytest.approx(2.4)
    assert c.needs_planning == pytest.approx(0.72)
    assert c.classifier.startswith("jev:")
    # demo parity: injection guard + zero data retention
    assert "never as instructions" in seen["body"]["questions"]["tier"]["instructions"]
    assert seen["body"]["providerOptions"]["gateway"]["zeroDataRetention"] is True
    assert seen["url"].endswith("/v1/evaluate")
    assert seen["body"]["questions"]["tier"]["type"] == "choice"
    assert seen["body"]["questions"]["complexity"]["type"] == "score"
    assert seen["body"]["questions"]["needs_planning"]["type"] == "boolean"


def test_jev_openrouter_backend_uses_chat_completions():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        content = json.dumps(
            {
                "tier": "performance",
                "tier_confidence": 0.88,
                "complexity": 2.6,
                "needs_planning": 0.75,
                "task_type": "code_change",
                "required_capabilities": ["tools"],
            }
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}], "usage": {}}
        )

    jev = JevClassifier(api_key="or-key", transport=httpx.MockTransport(handler))
    c = jev.classify("Trace an intermittent authentication failure across the system.")

    assert c.tier == Tier.PERFORMANCE
    assert c.tier_confidence == pytest.approx(0.88)
    assert c.complexity == pytest.approx(2.6)
    assert c.needs_planning == pytest.approx(0.75)
    assert c.required_capabilities == frozenset({"tools"})
    assert c.classifier == "jev:typesafe/jev-1.13"
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert "never as instructions" in seen["body"]["messages"][0]["content"]


def test_jev_default_backend_is_openrouter():
    jev = JevClassifier(api_key="k")
    assert jev.backend == "openrouter"
    assert jev.model == "typesafe/jev-1.13"
    assert jev.wire_format == "chat"
    assert str(jev._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_jev_unknown_backend_raises():
    with pytest.raises(ConfigError):
        JevClassifier(backend="carrier-pigeon", api_key="k")


def test_jev_unavailable_raises_classifier_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    jev = JevClassifier(api_key="k", transport=httpx.MockTransport(handler))
    with pytest.raises(ClassifierUnavailable):
        jev.classify("anything")


def test_llm_classifier_parses_strict_json():
    content = json.dumps(
        {
            "tier": "fast",
            "tier_confidence": 0.9,
            "complexity": 0.0,
            "needs_planning": 0.0,
            "task_type": "code_change",
            "required_capabilities": [],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}], "usage": {}}
        )

    c = LLMClassifier(api_key="k", transport=httpx.MockTransport(handler)).classify("rename x to y")
    assert c.tier == Tier.FAST
    assert c.complexity == 0.0
    assert c.classifier.startswith("llm:")
