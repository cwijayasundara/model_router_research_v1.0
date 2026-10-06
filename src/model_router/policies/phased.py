"""Phased policy - ported from Pi's jev-router.ts example.

Plans on a strong model chosen by a nested policy (or fixed tier), then hands
the rest of the turn to a fixed cheap implementation model after the first
successful edit/write tool call, accepting a single prompt-cache miss. The phase
lives in `decision.state`, which the router keeps per session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..catalog import ModelCatalog
from ..types import RouteDecision, TaskClassification

IMPLEMENT_TOOLS = frozenset({"edit", "write", "apply_patch", "multiedit"})


@dataclass(slots=True)
class PhasedState:
    phase: str  # "planning" | "implementation"
    planning_model: str
    impl_model: str


class PhasedPolicy:
    def __init__(
        self,
        planning_policy: Any | None = None,
        impl_model_id: str = "gpt-6-luna",
        impl_provider: str | None = "openai",
    ) -> None:
        from .tiered import TierPolicy  # local import avoids cycle

        self.planning_policy = planning_policy or TierPolicy()
        self.impl_model_id = impl_model_id
        self.impl_provider = impl_provider

    @staticmethod
    def should_start_implementation(tool_name: str, ok: bool = True) -> bool:
        """Agent loops call this after each tool result; True means switch phase."""
        return ok and tool_name in IMPLEMENT_TOOLS

    @staticmethod
    def implementation_state(state: PhasedState) -> PhasedState:
        """Transition helper: after the first successful edit, hand off to the impl model."""
        return PhasedState(
            phase="implementation", planning_model=state.planning_model, impl_model=state.impl_model
        )

    def decide(
        self,
        classification: TaskClassification,
        catalog: ModelCatalog,
        *,
        reason: str = "user",
        state: Any = None,
    ) -> RouteDecision:
        impl = catalog.require(self.impl_model_id, self.impl_provider) if self.impl_provider else catalog.require(self.impl_model_id)

        if reason == "direct" or (isinstance(state, PhasedState) and state.phase == "implementation"):
            return RouteDecision(
                model=impl,
                tier=impl.tier,
                reason="implementation",
                events=["phased: implementation on cheap model"],
                classification=classification,
                state=state,
            )

        planning = self.planning_policy.decide(classification, catalog, reason=reason, state=None)
        new_state = PhasedState(phase="planning", planning_model=planning.model.id, impl_model=impl.id)
        planning.state = new_state
        planning.events.append(f"phased: planning on {planning.model.label()}")
        return planning
