"""Zero-cost, offline heuristic classifier.

Not as good as Jev, but always available: use it as the default fallback, for
tests, and for the offline demo. Conservative by design - moderate confidence so
the tier policy escalates when unsure.
"""

from __future__ import annotations

import re
import time

from ..types import TaskClassification, Tier
from .base import DEFAULT_TIER_DESCRIPTIONS  # noqa: F401  (re-export convenience)

_COMPLEX = (
    r"migrat", r"architect", r"security", r"race condition", r"zero-downtime",
    r"diagnos", r"intermittent", r"distributed", r"root cause", r"trade-?off",
    r"scal(e|ing)", r"throughput", r"concurrency", r"auth", r"encrypt",
    r"across (the )?(system|service|codebase|stack)", r"refactor", r"redesign",
    r"performance (bug|regression|issue)", r"memory leak", r"deadlock",
)
_MECHANICAL = (
    r"\btypo\b", r"\brename\b", r"readme", r"aria-label", r"comment", r"formatting",
    r"log message", r"bump version", r"import sort", r"lint", r"spelling",
    r"variable name", r"docstring", r"\bcss\b (color|padding|margin)",
)
_PLANNING = (
    r"\bdesign\b", r"\bpropose\b", r"\bplan\b", r"should we", r"\bwhether\b",
    r"\bstrategy\b", r"\bapproach\b", r"ambigu", r"without (breaking|weakening)",
    r"rollback", r"zero-downtime",
)
_QUESTION = re.compile(r"^\s*(what|who|when|where|why|how|which|is|are|can|could|does|do|should)\b", re.I)
_TOOLS = re.compile(r"\b(run|execute|deploy|install|benchmark|migrat\w*|test suite|its (existing )?test)\b", re.I)
_VISION = re.compile(r"\b(image|screenshot|diagram|photo|figure|chart|mockup)\b", re.I)
_CODE = re.compile(r"\b(code|function|class|bug|fix|refactor|test|api|endpoint|component)\b", re.I)


class HeuristicClassifier:
    name = "heuristic"

    def classify(self, task: str) -> TaskClassification:
        started = time.perf_counter()
        t = task.strip()
        low = t.lower()
        complex_hits = sum(1 for p in _COMPLEX if re.search(p, low))
        mechanical_hits = sum(1 for p in _MECHANICAL if re.search(p, low))
        planning_hits = sum(1 for p in _PLANNING if re.search(p, low))

        is_question = bool(_QUESTION.match(t))
        if complex_hits >= 2:
            complexity, tier, confidence = 3.0, Tier.PERFORMANCE, 0.75
        elif complex_hits == 1:
            complexity, tier, confidence = 2.0, Tier.BALANCED, 0.7
        elif (is_question or (mechanical_hits and len(t) < 600)) and not planning_hits:
            complexity, tier, confidence = 0.0, Tier.FAST, 0.8
        elif len(t) > 1200:
            complexity, tier, confidence = 1.0, Tier.BALANCED, 0.55
        else:
            complexity, tier, confidence = 1.0, Tier.BALANCED, 0.55

        needs_planning = 0.8 if planning_hits else (0.4 if "?" in t else 0.1)
        if is_question and complexity == 0:
            task_type = "question"
        elif _CODE.search(low):
            task_type = "code_change"
        else:
            task_type = "other"

        required: set[str] = set()
        if _TOOLS.search(low):
            required.add("tools")
        if _VISION.search(low):
            required.add("vision")

        return TaskClassification(
            tier=tier,
            tier_confidence=confidence,
            complexity=complexity,
            needs_planning=needs_planning,
            task_type=task_type,
            required_capabilities=frozenset(required),
            est_input_tokens=max(len(t) // 4, 64),
            classifier=self.name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw={"complex_hits": complex_hits, "mechanical_hits": mechanical_hits},
        )
