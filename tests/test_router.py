import pytest

from model_router import (
    CostLedger,
    FakeProvider,
    HeuristicClassifier,
    Router,
    Tier,
    TierPolicy,
)

ALL_PROVIDERS = ["openai", "google", "fireworks", "baseten", "openrouter", "deepseek"]


def make_router(fake: FakeProvider | None = None, **kwargs) -> tuple[Router, FakeProvider]:
    fake = fake or FakeProvider()
    router = Router(
        classifier=HeuristicClassifier(),
        policy=TierPolicy(),
        providers={p: fake for p in ALL_PROVIDERS},
        **kwargs,
    )
    return router, fake


def test_ask_end_to_end_with_fake_provider():
    router, fake = make_router()
    resp = router.ask(
        "Rename ctaText to buttonLabel in the pricing card component and update its references."
    )
    assert resp.text == "ok: done"
    assert resp.decision.model.tier == Tier.FAST
    assert resp.cost > 0
    assert resp.input_tokens == 120 and resp.output_tokens == 60
    assert len(router.ledger) == 1
    assert fake.calls[0][1] == resp.model_id


def test_sticky_continuation_keeps_model():
    router, fake = make_router()
    r1 = router.ask("Rename ctaText to buttonLabel.", session="s1")
    r2 = router.ask("Now update the snapshot tests too.", session="s1", reason="continuation")
    assert r1.model_id == r2.model_id
    assert r2.decision.reason == "sticky"
    assert len(fake.calls) == 2
    # conversation history accumulated for the session
    history = router._sessions["s1"].history
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]


def test_retry_escalates_to_next_tier_on_rate_limit():
    router, _ = make_router(fake=FakeProvider(fail_first_n=1))
    resp = router.ask(
        "Rename ctaText to buttonLabel in the pricing card component and update its references."
    )
    assert resp.text == "ok: done"
    assert resp.decision.model.tier == Tier.BALANCED  # escalated after 429
    assert any("retry" in e for e in resp.decision.events)
    assert len(router.ledger) == 1  # only the successful dispatch is billed


def test_non_retryable_error_raises():
    from model_router import ProviderError

    class Forbidden(FakeProvider):
        def complete(self, spec, messages, **params):
            raise ProviderError("forbidden", status=403, body="denied")

    router, _ = make_router(fake=Forbidden())
    with pytest.raises(ProviderError):
        router.ask("Rename ctaText to buttonLabel.")


def test_ledger_totals_and_export(tmp_path):
    router, _ = make_router()
    router.ask("What is the capital of France?")
    router.ask("Rename ctaText to buttonLabel.")
    totals = router.ledger.totals()
    assert totals["grand"]["calls"] == 2
    assert totals["grand"]["cost"] > 0
    assert set(totals["by_model"]) == {f"fireworks/glm-5.3-flash"}
    out = tmp_path / "ledger.json"
    router.ledger.to_json(str(out))
    assert out.exists()


def test_route_only_does_not_dispatch():
    router, fake = make_router()
    decision = router.route(
        "Design a zero-downtime migration with rollback and backfill plans."
    )
    assert decision.model.tier == Tier.PERFORMANCE
    assert fake.calls == []


def test_classifier_fallback_on_unavailable():
    from model_router import ClassifierUnavailable

    class Flaky(HeuristicClassifier):
        name = "jev-flaky"

        def classify(self, task):
            raise ClassifierUnavailable("jev endpoint down")

    router, fake = make_router()
    router.classifier = Flaky()
    resp = router.ask(
        "Rename ctaText to buttonLabel in the pricing card component and update its references."
    )
    assert resp.text == "ok: done"  # request still served
    cls = resp.decision.classification
    assert cls.classifier == "heuristic"  # fell back to offline heuristic
    assert cls.raw.get("classifier_fallback") == "jev-flaky"


def test_injected_provider_registry_bypasses_env_keys():
    from model_router import ModelCatalog

    catalog = ModelCatalog()  # empty catalog; register one model
    from model_router import ModelSpec

    catalog.register(
        ModelSpec(
            id="m1",
            provider="acme",
            tier=Tier.BALANCED,
            intelligence=9.0,
            api_key_env="ACME_KEY",  # deliberately unset in env
            input_price=1.0,
            output_price=2.0,
        )
    )
    router = Router(
        catalog=catalog,
        classifier=HeuristicClassifier(),
        providers={"acme": FakeProvider(default="from-acme")},
    )
    resp = router.ask("hello there")
    assert resp.text == "from-acme"
    assert resp.provider == "acme"
