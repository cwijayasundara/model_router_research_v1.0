# model-router

Route each task to the **cheapest capable LLM** across commercial and open-weight providers.

Synthesizes three approaches to model routing:

- **LangChain's harness router** — route in the harness, three tiers on the Pareto frontier, track outcomes (64% median cost cut)
- **jev-model-router-demo** — Jev decision-model classification with a conservative escalation policy
- **Pi's virtual models** — selection vs dispatch, sticky routing, phased plan→implement routing

```python
from model_router import Router, JevClassifier, TierPolicy

router = Router(classifier=JevClassifier(), policy=TierPolicy())
resp = router.ask("Add an aria-label to the close button and run its test.", session="t1")
print(resp.model_id, f"${resp.cost:.6f}", resp.decision.events)
```

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ".[dev]"      # from this directory
pytest                    # 36 offline tests
python -m model_router.demo   # offline routing demo, no API keys needed
```

- Tests always run against the **live source tree**: `conftest.py` puts `src/` first on `sys.path`, so edit-and-test needs no reinstall.
- Scripts/examples use the installed copy — after editing `src/`, refresh it with `pip install -U .` (or run with `PYTHONPATH=src`).
- ⚠️ Python 3.14 gotcha: avoid `pip install -e .` in environments that stamp files with the macOS `hidden` flag — 3.14's hardened `site.py` silently skips hidden `.pth` files, breaking editable installs. If you must use `-e`, clear it after installing: `chflags nohidden .venv/lib/python*/site-packages/*.pth`.

## How it works

```
task → Classifier → TaskClassification → Policy → RouteDecision → Provider → Response + CostLedger
```

| Component | Options | Notes |
|---|---|---|
| **Classifier** | `JevClassifier` — default backend **OpenRouter** (`typesafe/jev-1.13`); also `backend="gateway"` (Vercel AI Gateway) and `backend="typesafe"` | 1 request, 3 typed questions: tier choice + complexity score + needs-planning. ~50× faster than LLM classification. Injection guard + ZDR. Router degrades to the offline heuristic if Jev is unreachable. |
| | `LLMClassifier` | Any OpenAI-compatible chat model with strict JSON output. |
| | `HeuristicClassifier` | Zero-cost, offline. Default when no classifier is given. |
| **Policy** | `TierPolicy` | Ported `selectHarness`: escalate to performance tier when confidence < 0.68, complexity ≥ 1.75, or planning ≥ 0.6. |
| | `CheapestCapablePolicy` | argmin expected cost subject to capabilities, context, and a complexity-scaled intelligence floor. |
| | `PhasedPolicy` | Pi's plan→implement: strong model plans, cheap model implements after first edit. |
| **Provider** | `OpenAICompatProvider` | Covers OpenAI, Fireworks, Baseten, OpenRouter, DeepSeek, Gemini (OpenAI endpoint), vLLM/Ollama/llama.cpp. |
| | `AnthropicProvider` | Claude models. |
| | `FakeProvider` | Deterministic offline dispatch for tests and dry runs. |

**Router semantics** (mirroring Pi): `reason="user"` re-classifies; `reason="continuation"` is **sticky** (same model, preserves prompt cache); provider failures retry on the next tier or cheapest alternative host; every dispatch is recorded with its audit trail.

## Default catalog

Ships with: `gpt-6-luna`, `gpt-6.1-sol`, `gpt-6-astra` (OpenAI) · `gemini-3.8-flash` (Google) · `glm-5.3-flash`, `kimi-k3` (Fireworks + Baseten) · `deepseek-4` (DeepSeek, Fireworks, Baseten, OpenRouter) · Jev classifier entries.

> ⚠️ **Prices, context windows, and intelligence scores are placeholders.** Verify against provider pricing pages and the Artificial Analysis index; edit `src/model_router/data/default_catalog.json` or overlay your own file:

```python
catalog = ModelCatalog.load(path="my_catalog.json")  # merged over defaults
router = Router(catalog=catalog, ...)
```

The same open-weight model can be registered under several providers — the router picks the cheapest eligible host.

## API keys

| Env var | Provider |
|---|---|
| `OPENROUTER_API_KEY` | **Jev (default backend)** + deepseek-4 via OpenRouter |
| `AI_GATEWAY_API_KEY` | Jev alternate backend (Vercel AI Gateway) |
| `TYPESAFE_API_KEY` | Jev alternate backend (TypeSafe direct) |
| `OPENAI_API_KEY` | gpt-6 models |
| `GEMINI_API_KEY` | gemini-3.8-flash |
| `FIREWORKS_API_KEY` / `BASETEN_API_KEY` | open-weight hosts |
| `DEEPSEEK_API_KEY` | DeepSeek official |
| `ANTHROPIC_API_KEY` | Claude models |

## Usage patterns

**Decision only** (no dispatch — for demos, evals, agent loops):

```python
decision = router.route("Design a zero-downtime migration with rollback plans.")
print(decision.model.label(), decision.events)
```

**Session with sticky routing:**

```python
router.ask(msg, session="user-123")                          # classify + route
router.ask(followup, session="user-123", reason="continuation")  # same model, cache-friendly
```

**Cost report:**

```python
print(router.ledger.totals())
router.ledger.to_json("ledger.json")
```

**Custom policy or classifier** — implement `decide(classification, catalog, *, reason, state)` / `classify(task)` and pass the instance to `Router`.

See `PROPOSAL.md` for the full design rationale and roadmap.
