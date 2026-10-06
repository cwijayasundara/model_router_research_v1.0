import pytest

from model_router import (
    CheapestCapablePolicy,
    ModelCatalog,
    PhasedPolicy,
    TaskClassification,
    Tier,
    TierPolicy,
)


def make_cls(
    tier: Tier = Tier.FAST,
    conf: float = 0.9,
    complexity: float = 0.0,
    planning: float = 0.0,
    caps: frozenset[str] = frozenset(),
    est_in: int = 1_000,
) -> TaskClassification:
    return TaskClassification(
        tier=tier,
        tier_confidence=conf,
        complexity=complexity,
        needs_planning=planning,
        required_capabilities=caps,
        est_input_tokens=est_in,
    )


@pytest.fixture
def catalog() -> ModelCatalog:
    return ModelCatalog.load()


def test_clean_classification_stays_in_tier(catalog):
    d = TierPolicy().decide(make_cls(), catalog)
    assert d.model.tier == Tier.FAST
    assert d.reason == "classified"
    assert d.events == []


def test_uncertainty_escalates_to_performance(catalog):
    d = TierPolicy().decide(make_cls(conf=0.5), catalog)
    assert d.model.tier == Tier.PERFORMANCE
    assert any("uncertain" in e for e in d.events)


def test_complexity_escalates(catalog):
    d = TierPolicy().decide(make_cls(tier=Tier.BALANCED, conf=0.9, complexity=2.0), catalog)
    assert d.model.tier == Tier.PERFORMANCE
    assert any("complexity" in e for e in d.events)


def test_planning_probability_escalates(catalog):
    d = TierPolicy().decide(make_cls(complexity=1.0, planning=0.8), catalog)
    assert d.model.tier == Tier.PERFORMANCE


def test_retry_escalates_one_tier(catalog):
    d = TierPolicy().decide(make_cls(), catalog, reason="retry")
    assert d.model.tier == Tier.BALANCED
    assert any("retry" in e for e in d.events)


def test_capability_shortfall_escalates_tier(catalog):
    # no fast-tier model has "reasoning" -> must escalate to balanced
    d = TierPolicy().decide(make_cls(caps=frozenset({"reasoning"})), catalog)
    assert d.model.tier == Tier.BALANCED
    assert any("no eligible" in e for e in d.events)


def test_huge_context_escalates_tier(catalog):
    d = TierPolicy().decide(make_cls(caps=frozenset({"tools"}), est_in=900_000), catalog)
    assert d.model.context_window >= 904_000


def test_cheapest_capable_picks_cheapest_eligible(catalog):
    d = CheapestCapablePolicy().decide(make_cls(), catalog)
    assert d.model.label() == "fireworks/glm-5.3-flash"
    assert d.alternatives  # shows what it rejected


def test_cheapest_capable_respects_intelligence_floor(catalog):
    d = CheapestCapablePolicy().decide(
        make_cls(tier=Tier.PERFORMANCE, complexity=3.0), catalog
    )
    assert d.model.intelligence >= 8.0
    assert d.model.label() == "deepseek/deepseek-4"  # cheapest host with int >= 8


def test_phased_policy_plans_then_implements(catalog):
    policy = PhasedPolicy()
    planning = policy.decide(make_cls(), catalog)
    assert planning.state.phase == "planning"
    assert planning.state.impl_model == "gpt-6-luna"

    assert policy.should_start_implementation("edit", ok=True)
    assert not policy.should_start_implementation("read", ok=True)

    # after the first successful edit, the agent loop transitions the phase
    impl_state = policy.implementation_state(planning.state)
    impl = policy.decide(make_cls(), catalog, state=impl_state)
    assert impl.model.id == "gpt-6-luna"
    assert impl.reason == "implementation"

    direct = policy.decide(make_cls(), catalog, reason="direct")
    assert direct.model.id == "gpt-6-luna"
