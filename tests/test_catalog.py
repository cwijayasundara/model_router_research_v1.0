import json

import pytest

from model_router import ConfigError, ModelCatalog, ModelSpec, Tier


def test_load_default_catalog():
    c = ModelCatalog.load()
    assert c.get("gpt-6-luna") is not None
    assert c.get("gpt-6.1-sol").tier == Tier.BALANCED
    fast = c.by_tier(Tier.FAST)
    assert fast, "default catalog must contain fast-tier chat models"
    costs = [s.expected_cost(1_000, 256) for s in fast]
    assert costs == sorted(costs), "by_tier must be sorted cheapest-first"


def test_multi_provider_same_id():
    c = ModelCatalog.load()
    hosts = [s for s in c.all() if s.id == "deepseek-4"]
    assert len(hosts) >= 2, "deepseek-4 should exist under multiple providers"
    assert c.get("deepseek-4", provider="baseten").provider == "baseten"
    assert c.get("deepseek-4").provider == hosts[0].provider  # first registration wins


def test_capable_filters_and_sorts():
    c = ModelCatalog.load()
    capable = c.capable(required_capabilities={"tools"}, min_intelligence=8.0)
    assert capable
    assert all(s.intelligence >= 8.0 and "tools" in s.capabilities for s in capable)
    costs = [s.expected_cost(1_000, 256) for s in capable]
    assert costs == sorted(costs)


def test_classifier_models_are_separate_kind():
    c = ModelCatalog.load()
    chat = c.all(kind="chat")
    cls = c.all(kind="classifier")
    assert chat and cls
    assert all(s.kind == "classifier" for s in cls)
    assert all("jev" in s.id.lower() for s in cls)


def test_register_save_roundtrip(tmp_path):
    c = ModelCatalog.load()
    c.register(ModelSpec(id="test-model", provider="test", tier=Tier.FAST, intelligence=9.9))
    assert c.get("test-model").intelligence == 9.9
    path = tmp_path / "catalog.json"
    c.save(path)
    c2 = ModelCatalog()
    c2.add_dict(json.loads(path.read_text()))
    assert c2.get("test-model") is not None


def test_invalid_entries_raise():
    c = ModelCatalog()
    with pytest.raises(ConfigError):
        c.add_dict({"models": [{"id": "no-provider"}]})
    with pytest.raises(ConfigError):
        c.add_dict({"models": [{"id": "x", "provider": "p", "tier": "ultra"}]})


def test_strongest_fallback():
    c = ModelCatalog.load()
    strongest = c.strongest()
    assert strongest.intelligence >= max(s.intelligence for s in c.all(kind="chat")) - 1e-9
