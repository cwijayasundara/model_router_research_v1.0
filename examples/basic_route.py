"""Basic routing: Jev classifier if a key is present, heuristics otherwise.

    python examples/basic_route.py "your task here"
"""

import os
import sys

from model_router import HeuristicClassifier, JevClassifier, Router, TierPolicy


def main() -> None:
    task = " ".join(sys.argv[1:]) or (
        "The checkout became slow after last week's release. Find the cause and fix it "
        "without changing user-visible behavior."
    )

    if os.getenv("OPENROUTER_API_KEY"):
        classifier = JevClassifier(backend="openrouter")
        print(f"classifier: jev via OpenRouter ({classifier.model})")
    elif os.getenv("AI_GATEWAY_API_KEY"):
        classifier = JevClassifier(backend="gateway")
        print(f"classifier: jev via Vercel AI Gateway ({classifier.model})")
    elif os.getenv("TYPESAFE_API_KEY"):
        classifier = JevClassifier(backend="typesafe")
        print(f"classifier: jev via TypeSafe ({classifier.model})")
    else:
        classifier = HeuristicClassifier()
        print("classifier: heuristic (set OPENROUTER_API_KEY for Jev)")

    router = Router(classifier=classifier, policy=TierPolicy())
    decision = router.route(task)

    print(f"\ntask:          {task[:90]}{'...' if len(task) > 90 else ''}")
    print(f"classified:    tier={decision.classification.tier.value} "
          f"confidence={decision.classification.tier_confidence:.2f} "
          f"complexity={decision.classification.complexity:.1f} "
          f"planning={decision.classification.needs_planning:.2f}")
    print(f"routed to:     {decision.model.label()} ({decision.model.tier.value} tier)")
    print(f"est. cost:     ${decision.model.expected_cost(decision.classification.est_input_tokens):.6f}")
    if decision.events:
        print(f"policy events: {'; '.join(decision.events)}")

    # To actually dispatch (requires the relevant provider key):
    # resp = router.ask(task, session="demo")
    # print(resp.text, resp.cost, router.ledger.totals())


if __name__ == "__main__":
    main()
