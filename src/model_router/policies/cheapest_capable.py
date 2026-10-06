"""Cheapest-capable policy: argmin expected cost subject to constraints.

Instead of fixed tiers, it filters the whole catalog by required capabilities,
estimated context, and a minimum intelligence score that scales with task
complexity, then picks the cheapest eligible (provider, price) pair. This is the
"Pareto frontier" approach: an open-weight model on a cheap host often wins.
"""

from __future__ import annotations

from typing import Any

from ..catalog import ModelCatalog
from ..types import RouteDecision, TaskClassification

DEFAULT_MIN_INTELLIGENCE = {0: 0.0, 1: 5.0, 2: 6.5, 3: 8.0}


class CheapestCapablePolicy:
    def __init__(
        self,
        min_intelligence_by_complexity: dict[int, float] | None = None,
        context_margin_tokens: int = 4_000,
    ) -> None:
        self.min_intelligence = dict(min_intelligence_by_complexity or DEFAULT_MIN_INTELLIGENCE)
        self.context_margin_tokens = context_margin_tokens

    def decide(
        self,
        classification: TaskClassification,
        catalog: ModelCatalog,
        *,
        reason: str = "user",
        state: Any = None,
    ) -> RouteDecision:
        events: list[str] = []
        bucket = min(int(classification.complexity), max(self.min_intelligence))
        min_int = self.min_intelligence.get(bucket, 0.0)
        if reason == "retry":
            min_int += 0.5
            events.append(f"retry: raised intelligence floor to {min_int}")

        needed_ctx = (
            classification.est_input_tokens
            + classification.est_output_tokens
            + self.context_margin_tokens
        )
        candidates = catalog.capable(
            required_capabilities=classification.required_capabilities,
            min_ctx=needed_ctx,
            min_intelligence=min_int,
        )
        if not candidates:
            strongest = catalog.strongest()
            events.append(
                f"no model meets constraints (caps={sorted(classification.required_capabilities)}, "
                f"min_intelligence={min_int}); fallback to {strongest.label()}"
            )
            candidates = [strongest]

        spec = candidates[0]
        return RouteDecision(
            model=spec,
            tier=spec.tier,
            reason="cheapest-capable" if not events else "escalated",
            events=events,
            classification=classification,
            state=state,
            alternatives=candidates[1:4],
        )
