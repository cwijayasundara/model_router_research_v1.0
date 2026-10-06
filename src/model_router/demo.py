"""Offline demo: route six sample tasks without any API keys.

    python -m model_router.demo
"""

from __future__ import annotations

from .classifiers import HeuristicClassifier
from .policies import TierPolicy
from .router import Router
from .types import Tier

SAMPLE_TASKS = [
    ("README edit", "Change the README install command from npm to pnpm and verify the Markdown still renders correctly."),
    ("Accessibility fix", "Add the missing aria-label to the icon-only close button in the settings modal and run its existing component test."),
    ("Localized refactor", "Rename ctaText to buttonLabel in the pricing card component and update its direct references."),
    ("Ambiguous perf bug", "The checkout became slow after last week's release. Find the cause and fix it without changing user-visible behavior."),
    ("Security diagnosis", "Trace an intermittent authentication failure across middleware, session storage, and OAuth callbacks. Propose a fix without weakening replay or CSRF protection."),
    ("Data migration", "Design a zero-downtime migration that splits the customers table into accounts and people, preserves API compatibility, and includes rollback and backfill plans."),
]


def main() -> None:
    router = Router(classifier=HeuristicClassifier(), policy=TierPolicy())
    print("model_router offline demo (heuristic classifier, no API keys)\n")
    print(f"{'task':<20} {'tier':<12} {'model':<28} {'cx':>4} {'plan':>5}  events")
    print("-" * 110)
    for label, task in SAMPLE_TASKS:
        decision = router.route(task)
        cls = decision.classification
        print(
            f"{label:<20} {cls.tier.value:<12} {decision.model.label():<28} "
            f"{cls.complexity:>4.1f} {cls.needs_planning:>5.2f}  {', '.join(decision.events) or '-'}"
        )
    print(
        "\nConservative policy: uncertainty < 0.68, complexity >= 1.75, or planning >= 0.60\n"
        "escalates to the performance tier (ported from jev-model-router-demo)."
    )
    print(f"tiers available: {', '.join(t.value for t in Tier)}")


if __name__ == "__main__":
    main()
