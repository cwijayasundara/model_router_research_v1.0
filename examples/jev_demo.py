"""Port of jev-model-router-demo: route six sample tasks, print the decision table.

Runs fully offline with the heuristic classifier; uses Jev when OPENROUTER_API_KEY
(default backend) or AI_GATEWAY_API_KEY / TYPESAFE_API_KEY (alternates) is set.

    python examples/jev_demo.py
"""

import json
import os
import time

from model_router import HeuristicClassifier, JevClassifier, Router, TierPolicy

SAMPLE_TASKS = [
    ("README edit", "Change the README install command from npm to pnpm and verify the Markdown still renders correctly."),
    ("Accessibility fix", "Add the missing aria-label to the icon-only close button in the settings modal and run its existing component test."),
    ("Localized refactor", "Rename ctaText to buttonLabel in the pricing card component and update its direct references."),
    ("Ambiguous perf bug", "The checkout became slow after last week's release. Find the cause and fix it without changing user-visible behavior."),
    ("Security diagnosis", "Trace an intermittent authentication failure across middleware, session storage, and OAuth callbacks. Propose a fix without weakening replay or CSRF protection."),
    ("Data migration", "Design a zero-downtime migration that splits the customers table into accounts and people, preserves API compatibility, and includes rollback and backfill plans."),
]


def main() -> None:
    if os.getenv("OPENROUTER_API_KEY"):
        classifier, name = JevClassifier(backend="openrouter"), f"jev via OpenRouter ({JevClassifier().model})"
    elif os.getenv("AI_GATEWAY_API_KEY"):
        classifier, name = JevClassifier(backend="gateway"), "jev via Vercel AI Gateway"
    elif os.getenv("TYPESAFE_API_KEY"):
        classifier, name = JevClassifier(backend="typesafe"), "jev via TypeSafe"
    else:
        classifier, name = HeuristicClassifier(), "heuristic (offline)"

    router = Router(classifier=classifier, policy=TierPolicy())
    print(f"\nmodel_router demo — classifier: {name}")
    print("Complex or uncertain -> performance tier | contained and mechanical -> fast tier\n")

    rows, total_ms = [], 0.0
    for label, task in SAMPLE_TASKS:
        started = time.perf_counter()
        decision = router.route(task)
        ms = (time.perf_counter() - started) * 1000
        total_ms += ms
        c = decision.classification
        rows.append(
            {
                "task": label,
                "tier": c.tier.value,
                "model": decision.model.label(),
                "confidence": round(c.tier_confidence, 2),
                "complexity": c.complexity,
                "planning": c.needs_planning,
                "latency_ms": round(ms),
                "events": "; ".join(decision.events) or "-",
            }
        )

    print(json.dumps(rows, indent=2))
    print(f"\ntotal classification latency: {total_ms:.0f}ms")


if __name__ == "__main__":
    main()
