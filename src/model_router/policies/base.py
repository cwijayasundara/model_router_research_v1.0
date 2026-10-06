"""Policy interface: map a classification to a concrete model decision."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..catalog import ModelCatalog
from ..types import RouteDecision, TaskClassification


@runtime_checkable
class Policy(Protocol):
    def decide(
        self,
        classification: TaskClassification,
        catalog: ModelCatalog,
        *,
        reason: str = "user",
        state: Any = None,
    ) -> RouteDecision:
        """reason mirrors Pi's virtual-model semantics: user | continuation | retry | direct."""
        ...
