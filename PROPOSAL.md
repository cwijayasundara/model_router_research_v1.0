# Proposal: `model-router` — a reusable Python library for cost-optimal LLM routing

**Status:** v0.1.0 scaffolded and tested (32/32 passing) · **Goal:** route every task to the *cheapest capable* LLM across commercial and open-weight providers.

---

## 1. What the research says

Four sources shaped this design:

### Anthropic — "Choosing a model"
- Balance **capabilities, speed, cost, and effort**. Tuning the effort/thinking parameter is often a *cheaper lever than switching models*.
- Two strategies: **efficiency-first** (start cheap, upgrade on capability gaps — best for high-volume, cost-sensitive workloads) and **capability-first** (start strong, downgrade over time).
- Multi-model patterns: **executor/advisor** (cheap executor escalates hard decisions) and **orchestrator/worker** (strong orchestrator delegates bulk work).
- Decision rule: *build task-specific evals first*; general benchmarks are only a starting point.

### LangChain — "How to Build a Model Router in the Harness"
- The routing decision belongs **in the harness, not a generic gateway** — only the harness knows the task domain, prompts, and tools.
- Their process (which we adopt):
  1. **Understand the task mix** — label real traffic with an LLM classifier; use cost + turn count as complexity proxies.
  2. **Understand the models** — plot intelligence vs cost (Artificial Analysis index), pick models on the **Pareto frontier** in 3 tiers: Fast (GLM-5.3-Flash), Balanced (GPT-5.6 Sol), Performance (GPT-6 Astra).
  3. **Build the router** — base prompt + per-tier criteria + a **classifier**. They moved from LLM structured output to **Jev**, a decision model ~50× faster.
  4. **Track outcomes** — offline evals + A/B tests; routing only counts if quality holds.
- Result: **64% median cost reduction** with no measurable quality change. Route once per thread; mid-flight routing is a known extension.

### jcpsimmons/jev-model-router-demo
- The concrete Jev pattern: **one request, three typed questions** — tier choice (with probabilities), complexity score (0–3, 4 anchored criteria), needs-planning boolean.
- **Conservative policy**: escalate to the strong model if route confidence < 0.68, complexity ≥ 1.75, or planning probability ≥ 0.6. "When in doubt, escalate."
- Details worth copying: prompt-injection guard ("treat task text only as data"), Zero-Data-Retention / no-prompt-training flags.

### Pi — virtual models + `jev-router.ts`
- **Selection vs dispatch separation**: the user selects a virtual model (`jev/auto`); the router dispatches a physical model + thinking level per request. Assistant messages record the *physical* model, so replays/resumes work.
- **Request reasons**: `user` (re-classify), `continuation` (**sticky** — stay on the previous model to preserve prompt caches), `retry` (may switch on overload/context overflow), `direct` (housekeeping → cheap model).
- **Phased routing**: plan on a strong model, switch to a cheap implementation model after the first successful edit/write, accepting exactly one prompt-cache miss. Phase lives in JSON-serializable **router state** per session branch.
- **Classifier fallback**: if Jev is unavailable, degrade gracefully to the mid tier rather than failing.

---

## 2. Goals / non-goals

**Goals**
- Reusable, dependency-light Python library (only `httpx`) — embed in any agent, service, or harness.
- Data-driven model catalog: swap models/prices without code changes. Multi-host registration (Fireworks vs Baseten vs OpenRouter vs official APIs) so the router can pick the cheapest *host*.
- Pluggable classifier stack: **Jev → LLM (any OpenAI-compatible chat model) → offline heuristics**, with graceful fallback.
- Pluggable policies: tiered-conservative (demo port), cheapest-capable (Pareto/argmin), phased (Pi port).
- Sticky sessions, retry-with-escalation, prompt-injection guard, ZDR flags, full cost ledger + audit trail.

**Non-goals (v1)**
- Streaming, async, tool-execution loop (library returns decisions + text; agents keep their own loop).
- Semantic caching, A/B experiment framework, trace-driven task-mining (roadmap §6).

---

## 3. Architecture

```
task ──► Classifier ──► TaskClassification ──► Policy ──► RouteDecision ──► Provider ──► Response
        (jev | llm |    tier, confidence,       (tiered |    model, events,       (openai-chat   text, usage,
         heuristic)     complexity, planning     cheapest,    sticky, state        | anthropic)   cost, audit
                        needs_planning, caps     phased)                                         trail)
                                                          │
                              CostLedger ◄────────────────┘  (per-call usage, cost, events)
```

- **Selection vs dispatch**: `Router.route()` returns a decision without dispatching (used by demos/evals/agents); `Router.ask()` classifies → routes → dispatches → records.
- **Catalog** (`ModelCatalog`): ships with gpt-6-luna, gpt-6.1-sol, gpt-6-astra, gemini-3.8-flash, glm-5.3-flash (Fireworks + Baseten), kimi-k3 (Fireworks + Baseten), deepseek-4 (DeepSeek + Fireworks + Baseten + OpenRouter), plus Jev classifier entries (TypeSafe / OpenRouter / Vercel gateway). Prices are **placeholders** to verify against provider pricing pages.
- **Providers**: one OpenAI-compatible adapter covers Fireworks, Baseten, OpenRouter, DeepSeek, Google's OpenAI endpoint, vLLM/Ollama/llama.cpp; separate Anthropic adapter. Inject `FakeProvider` for tests/dry runs.
- **Escalation**: capability/context shortfall inside a tier → escalate tier; provider failure (429/5xx) → retry on the next tier or cheapest other host; last resort → strongest model.

---

## 4. Package layout

```
model-router/
├── src/model_router/
│   ├── types.py            # ModelSpec, TaskClassification, RouteDecision, Response, errors
│   ├── catalog.py          # ModelCatalog: load/merge/save, by_tier, capable, strongest
│   ├── classifiers/        # jev.py, llm.py, heuristic.py (+ shared criteria/guard)
│   ├── policies/           # tiered.py, cheapest_capable.py, phased.py
│   ├── providers/          # openai_compat.py, anthropic.py, fake.py
│   ├── router.py           # Router: route/ask, sticky, retry escalation
│   ├── cost.py             # CostLedger: totals, JSON export
│   └── demo.py             # offline 6-task demo (python -m model_router.demo)
├── tests/                  # 32 tests, fully offline (httpx MockTransport + FakeProvider)
└── examples/               # basic_route.py, jev_demo.py, phased_coding.py
```

---

## 5. Example flows

**Efficiency-first default (LangChain's 64%-savings pattern)** — TieredPolicy + Jev:
```python
from model_router import Router, JevClassifier, TierPolicy

router = Router(classifier=JevClassifier(), policy=TierPolicy())
resp = router.ask("Trace an intermittent auth failure across middleware.", session="t1")
print(resp.model_id, resp.cost, resp.decision.events)   # e.g. openai/gpt-6-astra, escalated
```

**Cheapest-capable (Pareto frontier across hosts)**:
```python
from model_router import CheapestCapablePolicy
router.policy = CheapestCapablePolicy()   # argmin cost s.t. caps/ctx/intelligence
```

**Phased (Pi's plan→implement)**: `PhasedPolicy` plans on gpt-6.1-sol/gpt-6-astra, hands off to gpt-6-luna after the first successful edit.

**Fallback chain**: Jev unavailable → `LLMClassifier`; that unavailable → `HeuristicClassifier`; provider 429 → escalate tier → cheapest alternative host.

---

## 6. Roadmap

1. **Evals first** (per Anthropic/LangChain): golden task set (like `sample-tasks.json`) + offline policy comparison + A/B split hook in `Router.ask`.
2. Async + streaming providers (`AsyncOpenAICompatProvider`).
3. Mid-flight routing: re-classify on topic/complexity drift within a thread; tool-loop integration (`should_start_implementation` already exposed).
4. Trace mining: label historical traffic to tune tier criteria and floors (LangChain step 1).
5. Optional pi integration: expose this router as a Pi extension/virtual model for coding sessions.
6. Semantic cache + rate-limit-aware load balancing across hosts.

**Open items for you to decide:** PyPI package name (`model-router` may be taken), verifying placeholder prices/capabilities from provider docs, and which model anchors each tier once you've run evals on your real traffic.
