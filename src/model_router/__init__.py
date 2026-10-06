"""model_router: route each task to the cheapest capable LLM.

Harness-style model routing (LangChain) with Jev decision-model classification
(jev-model-router-demo), sticky/phase routing and selection-vs-dispatch
separation (Pi virtual models).
"""

from .catalog import ModelCatalog
from .classifiers import Classifier, HeuristicClassifier, JevClassifier, LLMClassifier
from .cost import CostLedger, UsageEntry
from .policies import (
    CheapestCapablePolicy,
    PhasedPolicy,
    PhasedState,
    Policy,
    TierFloors,
    TierPolicy,
)
from .providers import (
    AnthropicProvider,
    FakeProvider,
    OpenAICompatProvider,
    Provider,
    ProviderResponse,
)
from .router import Router
from .types import (
    ConfigError,
    ClassifierUnavailable,
    Message,
    ModelSpec,
    ProviderError,
    Response,
    RouteDecision,
    TaskClassification,
    Tier,
)

__version__ = "0.1.0"

__all__ = [
    "Router",
    "ModelCatalog",
    "ModelSpec",
    "Tier",
    "Message",
    "TaskClassification",
    "RouteDecision",
    "Response",
    "CostLedger",
    "UsageEntry",
    "Classifier",
    "JevClassifier",
    "LLMClassifier",
    "HeuristicClassifier",
    "ClassifierUnavailable",
    "Policy",
    "TierPolicy",
    "TierFloors",
    "CheapestCapablePolicy",
    "PhasedPolicy",
    "PhasedState",
    "Provider",
    "ProviderResponse",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "FakeProvider",
    "ConfigError",
    "ProviderError",
    "__version__",
]
