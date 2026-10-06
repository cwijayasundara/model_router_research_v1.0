"""Router: classify -> decide -> dispatch, with sticky sessions and retry escalation.

Design mirrors Pi's virtual-model semantics:
- reasons: "user" (fresh classification), "continuation" (sticky - stays on the
  session's model to preserve prompt cache), "retry" (escalate after failure),
  "direct" (cheap housekeeping calls).
- Every dispatch is recorded in the CostLedger with its audit trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .catalog import ModelCatalog
from .classifiers import Classifier, HeuristicClassifier
from .cost import CostLedger
from .policies import Policy, TierPolicy
from .providers.anthropic import AnthropicProvider
from .providers.base import Provider, ProviderResponse
from .providers.openai_compat import OpenAICompatProvider
from .types import (
    ClassifierUnavailable,
    ConfigError,
    Message,
    ModelSpec,
    ProviderError,
    Response,
    RouteDecision,
    TaskClassification,
    Tier,
)


@dataclass(slots=True)
class _Session:
    decision: RouteDecision
    history: list[Message] = field(default_factory=list)


def _last_user_text(messages: list[Message] | None) -> str:
    if not messages:
        return ""
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


class Router:
    def __init__(
        self,
        *,
        catalog: ModelCatalog | None = None,
        classifier: Classifier | None = None,
        policy: Policy | None = None,
        ledger: CostLedger | None = None,
        providers: dict[str, Provider] | None = None,
    ) -> None:
        self.catalog = catalog or ModelCatalog.load()
        self.classifier = classifier or HeuristicClassifier()
        self.policy = policy or TierPolicy()
        self.ledger = ledger or CostLedger()
        self._injected: dict[str, Provider] = dict(providers or {})
        self._providers: dict[tuple[str, str], Provider] = {}
        self._sessions: dict[str, _Session] = {}
        self._fallback_classifier: HeuristicClassifier | None = None

    # -- routing -------------------------------------------------------------

    def route(
        self,
        task: str | None = None,
        *,
        messages: list[Message] | None = None,
        session: str | None = None,
        reason: str = "user",
        classification: TaskClassification | None = None,
    ) -> RouteDecision:
        """Decision only (no provider call). Useful for agents, demos, and evals."""
        sess = self._sessions.get(session) if session else None

        # Sticky: tool follow-ups / continuations stay on the model that handled the turn.
        if sess and reason == "continuation":
            d = sess.decision
            return RouteDecision(
                model=d.model,
                tier=d.model.tier,
                reason="sticky",
                events=["sticky: continuation stays on previous model"],
                classification=d.classification,
                state=d.state,
                sticky=True,
            )

        text = task or _last_user_text(messages)
        if not text:
            raise ConfigError("route() needs a task or messages containing a user message")
        if classification is None:
            classification = self._classify_with_fallback(text)
        state = sess.decision.state if sess else None
        return self.policy.decide(classification, self.catalog, reason=reason, state=state)

    def _classify_with_fallback(self, text: str) -> TaskClassification:
        """Classify; degrade to the offline heuristic if the classifier is unreachable
        (mirrors Pi's jev-router, which falls back to the mid tier when Jev is down)."""
        try:
            return self.classifier.classify(text)
        except ClassifierUnavailable:
            if self._fallback_classifier is None:
                self._fallback_classifier = HeuristicClassifier()
            classification = self._fallback_classifier.classify(text)
            classification.raw = {
                **classification.raw,
                "classifier_fallback": getattr(self.classifier, "name", type(self.classifier).__name__),
            }
            return classification

    # -- dispatch ------------------------------------------------------------

    def ask(
        self,
        task: str | None = None,
        *,
        messages: list[Message] | None = None,
        session: str | None = None,
        system: str | None = None,
        reason: str = "user",
        classification: TaskClassification | None = None,
        max_retries: int = 2,
        **params: Any,
    ) -> Response:
        incoming = list(messages or [])
        if task is not None:
            incoming.append(Message(role="user", content=task))
        if not incoming:
            raise ConfigError("ask() needs a task or messages")
        if system and not any(m.role == "system" for m in incoming):
            incoming.insert(0, Message(role="system", content=system))

        sess = self._sessions.get(session) if session else None
        history = list(sess.history) if sess else []
        conversation = history + incoming

        decision = self.route(
            task, messages=messages, session=session, reason=reason, classification=classification
        )
        spec = decision.model
        events = list(decision.events)
        classification = decision.classification

        attempt = 0
        while True:
            provider = self._provider_for(spec)
            started = time.perf_counter()
            try:
                out = provider.complete(spec, conversation, **params)
                break
            except ProviderError as e:
                attempt += 1
                events.append(f"attempt {attempt}: {spec.label()} failed ({e})")
                if attempt > max_retries or not e.retryable:
                    raise
                decision = self._escalate(spec, decision)
                spec = decision.model
                events.append(f"retry: switching to {spec.label()}")

        latency_ms = int((time.perf_counter() - started) * 1000)
        cost = spec.cost(out.input_tokens, out.output_tokens)
        decision.events = events
        self.ledger.record(
            provider=spec.provider,
            model=spec.id,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            session=session,
            reason=reason,
            events=events,
        )

        if session:
            new_history = conversation + [Message(role="assistant", content=out.text)]
            self._sessions[session] = _Session(decision=decision, history=new_history)

        return Response(
            text=out.text,
            provider=spec.provider,
            model_id=spec.id,
            decision=decision,
            input_tokens=out.input_tokens,
            output_tokens=out.output_tokens,
            cost=cost,
            latency_ms=latency_ms,
            stop_reason=out.stop_reason,
            raw=out.raw,
        )

    # -- session / provider management ---------------------------------------

    def end_session(self, session: str) -> None:
        self._sessions.pop(session, None)

    def _provider_for(self, spec: ModelSpec) -> Provider:
        injected = self._injected.get(spec.provider)
        if injected is not None:
            return injected
        key = (spec.provider, spec.api)
        if key not in self._providers:
            if spec.api == "openai-chat":
                self._providers[key] = OpenAICompatProvider.from_spec(spec)
            elif spec.api == "anthropic":
                self._providers[key] = AnthropicProvider.from_spec(spec)
            else:
                raise ConfigError(f"no provider adapter for api={spec.api!r} ({spec.label()})")
        return self._providers[key]

    def _escalate(self, failed: ModelSpec, decision: RouteDecision) -> RouteDecision:
        """Pick a different model after a failure: policy retry, else next tier."""
        if decision.classification is not None:
            d = self.policy.decide(
                decision.classification, self.catalog, reason="retry", state=decision.state
            )
            if d.model.key != failed.key:
                return d
        for tier in Tier.order()[Tier.order().index(failed.tier) + 1:]:
            for candidate in self.catalog.by_tier(tier):
                if candidate.key != failed.key and failed.capabilities <= candidate.capabilities:
                    return RouteDecision(
                        model=candidate,
                        tier=tier,
                        reason="retry",
                        events=[f"escalated to {candidate.label()} after failure"],
                        classification=decision.classification,
                        state=decision.state,
                    )
        strongest = self.catalog.strongest()
        return RouteDecision(
            model=strongest,
            tier=strongest.tier,
            reason="retry",
            events=[f"escalated to strongest model {strongest.label()}"],
            classification=decision.classification,
            state=decision.state,
        )
