"""Phased routing (Pi's jev-router pattern): plan on a strong model, implement on a cheap one.

Sketch for agent-loop authors: your loop keeps calling `router.route(...)` with the
session and advances the phase state after the first successful edit. The single
model switch costs one prompt-cache miss.

    python examples/phased_coding.py
"""

from model_router import (
    HeuristicClassifier,
    ModelCatalog,
    PhasedPolicy,
    Router,
    TaskClassification,
    Tier,
)


def main() -> None:
    catalog = ModelCatalog.load()
    classification = TaskClassification(
        tier=Tier.BALANCED,
        tier_confidence=0.9,
        complexity=2.0,
        needs_planning=0.7,
        est_input_tokens=2_000,
    )
    router = Router(
        catalog=catalog,
        classifier=HeuristicClassifier(),
        policy=PhasedPolicy(impl_model_id="gpt-6-luna", impl_provider="openai"),
    )

    decision = router.route("Refactor the billing module to support usage-based pricing.")
    state = decision.state
    print(f"phase=planning      -> {decision.model.label()}")

    # ... your agent loop runs the planning model; suppose it makes its first edit ...
    if state and PhasedPolicy.should_start_implementation("edit", ok=True):
        state = PhasedPolicy.implementation_state(state)
        decision = router.policy.decide(classification, catalog, state=state)
        print(f"phase=implementation -> {decision.model.label()}  (single prompt-cache miss)")


if __name__ == "__main__":
    main()
