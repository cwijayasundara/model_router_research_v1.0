"""Tiered conservative policy - the ported `selectHarness` from jev-model-router-demo.

Picks the tier the classifier chose, then escalates to `override_target` when any
doubt exists: low route confidence, high complexity, or likely planning. Also
escalates when no model in the tier satisfies required capabilities or the
estimated context, and on retries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..catalog import ModelCatalog
from ..types import RouteDecision, TaskClassification, Tier


@dataclass(slots=True)
class TierFloors:
    route_confidence_floor: float = 0.68  # below -> escalate (demo default)
    complexity_floor: float = 1.75  # at/above -> escalate (demo default)
    planning_floor: float = 0.6  # at/above -> escalate (demo default)


class TierPolicy:
    def __init__(
        self,
        floors: TierFloors | None = None,
        override_target: Tier = Tier.PERFORMANCE,
        context_margin_tokens: int = 4_000,
    ) -> None:
        self.floors = floors or TierFloors()
        self.override_target = override_target
        self.context_margin_tokens = context_margin_tokens

    def __repr__(self) -> str:  # pragma: no cover
        return f"TierPolicy(floors={self.floors!r}, override_target={self.override_target!r})"

    # -- helpers -----------------------------------------------------------

    def _pick_in_tier(
        self, tier: Tier, classification: TaskClassification, catalog: ModelCatalog
    ):
        needed_ctx = (
            classification.est_input_tokens
            + classification.est_output_tokens
            + self.context_margin_tokens
        )
        for spec in catalog.by_tier(tier):
            if (
                classification.required_capabilities <= spec.capabilities
                and spec.context_window >= needed_ctx
            ):
                return spec
        return None

    # -- Policy --------------------------------------------------------------

    def decide(
        self,
        classification: TaskClassification,
        catalog: ModelCatalog,
        *,
        reason: str = "user",
        state: Any = None,
    ) -> RouteDecision:
        events: list[str] = []
        f = self.floors
        tier = classification.tier

        if reason == "retry":
            tier = Tier.next(tier)
            events.append(f"retry: escalated tier to {tier.value}")

        # Conservative overrides (demo policy): any doubt -> override_target.
        if classification.tier_confidence < f.route_confidence_floor:
            tier = Tier.strongest(tier, self.override_target)
            events.append(
                f"uncertain route ({classification.tier_confidence:.2f} < "
                f"{f.route_confidence_floor:.2f}) -> {self.override_target.value}"
            )
        if classification.complexity >= f.complexity_floor:
            tier = Tier.strongest(tier, self.override_target)
            events.append(
                f"complexity {classification.complexity:.2f} >= {f.complexity_floor} "
                f"-> {self.override_target.value}"
            )
        if classification.needs_planning >= f.planning_floor:
            tier = Tier.strongest(tier, self.override_target)
            events.append(
                f"planning probability {classification.needs_planning:.2f} >= "
                f"{f.planning_floor} -> {self.override_target.value}"
            )

        spec = self._pick_in_tier(tier, classification, catalog)
        if spec is None:
            events.append(f"no eligible {tier.value} model (caps/context); escalating tier")
            for t in Tier.order()[Tier.order().index(tier) + 1 :]:
                spec = self._pick_in_tier(t, classification, catalog)
                if spec:
                    tier = t
                    break
        if spec is None:
            spec = catalog.strongest()
            events.append(f"fallback: strongest model {spec.label()}")

        route_reason = "escalated" if events else "classified"
        return RouteDecision(
            model=spec,
            tier=spec.tier,
            reason=route_reason,
            events=events,
            classification=classification,
            state=state,
        )
