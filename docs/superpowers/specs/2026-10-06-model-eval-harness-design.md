# Model Eval Harness — Design Spec

**Date:** 2026-10-06 · **Status:** draft for review · **Project:** model_router_v1.0
**Method basis:** LangChain "How to Build a Model Router in the Harness" (steps 1 & 4), judge design proven in `~/Documents/learning_101/jev_as_judge_v1.0`

## Goal

Empirically rank open-weight coding models on the user's **real task traffic** (mined from pi coding sessions) and report **quality-per-dollar**. Models under test: `glm-5.3-flash`, `kimi-k3`, `deepseek-4`, `gemma-4` (new, Google), with `gpt-6.1-sol` as closed-model reference.

## Non-goals (v1)

- No sandboxed execution of generated code (v2 — needs real infra)
- No live A/B split (v2)
- No prompt optimization

## Background (decisions from brainstorming)

1. **Task set**: real traffic now — mined from `~/.pi/agent/sessions/` (user's own coding prompts), not hand-authored
2. **Judge**: Jev via OpenRouter System One API, the user's proven pattern (`mean` of 3 yes/no quality probabilities, `does_pass` at 0.5 cutoff, `outcome` choice; 500/500 oracle agreement, ~$0.00035/call in their experiment)
3. **Grading depth**: output-text quality + embedded deterministic checks; no sandbox execution

## Architecture

New sub-package `src/model_router/evals/`:

| Module | Responsibility |
|---|---|
| `mine.py` | Extract real tasks from pi sessions. Walk `<sessions-dir>/*/*.jsonl` (default `~/.pi/agent/sessions`), take the **first user message per session** (the primary request), extract text from `message.content` text blocks, filter (length 20–4000 chars; drop trivial greetings/thanks via regex), dedupe by sha256 of normalized text, cap at `--limit` (default 20) using Jev's `task_type` classification for stratification (best-effort: without a key, round-robin over a shuffled list). Output: `evals/data/real_traffic.json` (**gitignored** — may contain private prompts). |
| `cases.py` | Case loader + schema. `EvalCase = {id, task, task_type, complexity?, oracle: {passes: bool \| null, notes}, source}`. Loads `real_traffic.json` (mined) or the hand-authored fallback set. Mined cases ship with `oracle.passes = null`. |
| `runner.py` | Cartesian case × model dispatch. Uses a new `FixedModelPolicy` (model never changes on retry — comparisons stay clean) so the existing `Router.ask()` handles dispatch, retries, cost accounting. No tier escalation during evals. |
| `judge.py` | `JevJudge`: one System One request per (case, model, output). Questions: `is_correct`, `is_complete`, `is_well_crafted` (yes/no) → `quality = mean(probs)`; `does_pass` (yes/no) → pass at 0.5; `outcome` (choice: `answered`/`partially_answered`/`failed`). Computes **judge-oracle agreement** where `oracle.passes` is non-null. Optional `LLMJudge` fallback flag. |
| `report.py` | Per model: n, pass_rate, mean_quality, total_cost, cost_per_task, cost_per_pass, p50_latency_ms, **quality_per_dollar**, judge_oracle_agreement. Outputs JSON (`evals/data/results-<timestamp>.json`) + a Markdown table sorted by quality-per-dollar. |
| `__main__.py` | CLI: `python -m model_router.evals --models glm-5.3-flash,kimi-k3,deepseek-4,gemma-4,gpt-6.1-sol --judge jev --budget-usd 5.0 --limit 20`. Hard-stops dispatch when cumulative cost exceeds `--budget-usd`. |

Supporting changes:

- `src/model_router/classifiers/systemone.py` — shared **System One client** (plain `httpx`, no `typesafe_sdk` dependency). Wire contract verified against `typesafe_sdk` source:
  - `POST {base_url}/v1/systemone` — base `https://openrouter.ai/api`, model `~typesafe/jev-latest`
  - Request body: `{"state": {...}, "model": "~typesafe/jev-latest", "questions": {...}}`
  - Question JSON: noul → `{"type": "noul", "instructions": str, "criteria": {"true": str, "false": str}}`; choice → `{"type": "choice", "instructions": str, "criteria": {label: description}}`
  - Response: `{"answers": {name: {...}}}` where noul answers carry `probability`, choice answers carry `choice` + `probabilities`; plus `usage`
  - Used by both `JevClassifier` and `JevJudge`.
- **Fix `JevClassifier`** OpenRouter backend: current implementation wrongly uses chat/completions with JSON schema; real wire format is System One (above): base `https://openrouter.ai/api`, path `/v1/systemone`, model `~typesafe/jev-latest`. Evaluate-style backends (gateway/typesafe) unchanged.
- **Catalog**: add `gemma-4` — provider `google`, `openai-chat` via `https://generativelanguage.googleapis.com/v1beta/openai`, `GEMINI_API_KEY`, tier `fast`, intelligence ~7.5 (placeholder — verify), prices placeholder — flagged for verification like all catalog entries.
- `src/model_router/policies/fixed.py` — `FixedModelPolicy(model_id, provider?)`: always returns that model regardless of `reason`; retry stays on-model.

## Testing strategy

- **Offline (default, no keys)**: mining parser vs fixture JSONL in `tests/fixtures/sessions/`; case-loader schema validation; runner across 3 fake models via `FakeProvider`; judge parsing via `httpx.MockTransport` fixture; report math (quality-per-dollar, agreement, p50); CLI smoke; budget-guard stop.
- **Live smoke** (`@pytest.mark.skipif` no `OPENROUTER_API_KEY`): classify 1 task; judge 1 canned answer.
- Existing 36 tests keep passing (catalog gains an entry; JevClassifier payload change updates its tests).

## Privacy & safety

- Mined prompts stay local; `real_traffic.json` and results files are gitignored (may contain private prompts).
- Jev calls carry Zero-Data-Retention + no-prompt-training flags (as in `jev_as_judge_v1.0`).
- Candidate-model calls send the user's own prompts to providers they configured.

## Success criteria

Report produced from **≥ 15 mined real tasks × 5 models**; cost, pass rate, quality, and quality-per-dollar per model; judge-oracle agreement reported for any labeled cases; total spend < $5; full offline test suite green.

## Risks

| Risk | Mitigation |
|---|---|
| No oracle labels on mined traffic → judge is the only truth | Mix in 2–3 hand-labeled cases; report judge-oracle agreement; oracle fields ready for later labeling |
| Gemma-4 endpoint/pricing unverified | Catalog entry flagged as placeholder, like the rest; runner fails soft per-model |
| Mined traffic contains non-coding prompts | Length/triviality filters + Jev `task_type` stratification |
| Judge same-family bias | Judge is Jev (never a candidate model); candidates span 4 families |
