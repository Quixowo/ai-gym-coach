# HeyCoach — an agentic AI weightlifting coach

A weightlifting tracker (sets / reps / RIR) with an **agentic coach** layered on top.

The coach can discuss your own logged training data, adjust your saved programs
(within safety limits), and answer grounded training / injury-prevention / nutrition questions. 

## Architecture at a glance

- **Hand-rolled agent loop** on the raw Anthropic SDK. Claude Sonnet drives the loop;
  Claude Haiku handles the cheap classification / synthesis calls.
  A hard cap on tool-call iterations per turn is used as the loop-safety mechanism.
- **Seven tools**, all server-executed. The verified user id is injected server-side
  from the JWT before any tool runs and is *never* an LLM-supplied argument.
- **Deterministic logic lives in code, not the model**: progression math (Epley 1RM,
  RIR trend, plateau detection), fuzzy exercise-name matching, and the 10% program
  load-jump cap are all enforced in tool implementations.
- **RAG is a tool** (`search_knowledge_base`), not always-on context — the agent
  decides per message whether to retrieve. Retrieval → Haiku synthesis grounded
  strictly in the retrieved chunks → a second Haiku groundedness check; a failed
  check keeps the sources and returns a conservative answer rather than fabricating.
- **Injury red-flag classifier**: a cheap Haiku call runs before the agent loop and
  short-circuits acute-injury messages to a fixed "see a professional" response.
- **Application-level access control** (`WHERE user_id = …`), not Postgres RLS.
- **Live trace view** piggybacked on the chat SSE stream (no persisted trace storage).

**Stack:** FastAPI (async) · React + Vite · Postgres + pgvector · Upstash Redis ·
Anthropic API · Voyage embeddings (`voyage-4-lite`). Local dev on Docker Compose;
deploy on Render (backend) + Vercel (frontend).

## The eval suite — what it does and doesn't verify

Per the project's CI rule, **model-behavior tests run against recorded fixtures, never
a live model.** Each of the three model-facing suites replays Claude / Voyage
responses that were captured once against the live API
(`backend/tests/fixtures/record_fixtures.py`) and committed under
`backend/tests/fixtures/claude_responses/`. CI needs no API keys and makes no live
calls.

**These tests verify that the *code* handles a given model response correctly** — the
orchestrator calls the right tool with the right arguments, the RAG pipeline stays
grounded and refuses out-of-corpus questions, the classifier parses verdicts and
enforces its fail-open rule. **They do not re-verify that a live model still makes the
same decisions today.** Re-checking that is a manual, periodic activity: if a Claude
model version changes materially, re-run the recorder and re-review the fixtures. The
pure-code suites (progression math, fuzzy search) need no fixtures and test real
behavior on every run.

The three suites:

| Suite | File | What it asserts |
|---|---|---|
| Tool-call correctness | `test_tool_correctness.py` | Recorded agent turns invoke the expected tool(s) with correct args; the 10% load-jump cap *rejects* an over-cap program edit; a "log 3 sets" turn writes the correct `set_entries` rows. |
| RAG groundedness | `test_groundedness.py` | The real `search_knowledge_base` reproduces the recorded grounded answer / sources / verdict; answerable questions stay grounded; out-of-corpus traps are refused, not fabricated. |
| Injury red-flag recall | `test_red_flag_recall.py` | The real classifier maps each recorded Haiku verdict to the right boolean; recall and false-positive rate hold. |

## Eval results (recorded pass)

Recorded once against `claude-sonnet-4-6`, `claude-haiku-4-5`, and `voyage-4-lite`
(raw per-item outcomes in `backend/tests/fixtures/recording_metrics.json`):

| Metric | Result |
|---|---|
| **RAG groundedness pass rate** | **15 / 15** (training 12/12, injury-prevention 3/3) |
| **Out-of-corpus traps refused** | **4 / 4** (0 fabricated) |
| **Citation density** | mean **2.27 / 5** retrieved chunks actually used per answer |
| **Injury red-flag recall** | **12 / 12** acute cases caught (100%) |
| **Injury red-flag false-positive rate** | **0 / 13** routine cases flagged (0%) |
| **Tool-call correctness** | **6 / 6** turns invoked the expected tools; the 1 load-jump-cap case fired its rejection |
| **Red-flag classifier latency** (Haiku) | n=25 · mean **808 ms** · median **646 ms** · p95 **1381 ms** |

These are frozen in the committed fixtures, so the suite reproduces them
deterministically in CI. Re-recording (a manual step) is the only way they change.

## Running it locally

```bash
docker compose up -d                                   # Postgres+pgvector, Redis
cd backend && .venv/Scripts/python -m alembic upgrade head
cd backend && .venv/Scripts/python -m seed.seed_exercises          # exercise catalog
cd backend && .venv/Scripts/python -m seed.ingest_knowledge_base   # RAG corpus (needs VOYAGE_API_KEY)
cd backend && uvicorn app.main:app --reload            # backend
cd frontend && npm run dev                             # frontend
```

Tests and lint (no API keys required — the eval suites run on recorded fixtures):

```bash
cd backend && pytest
cd backend && ruff check . && ruff format --check .
```

> **Note:** `pytest` wipes and rewrites `knowledge_chunks` in the local dev DB (the RAG
> suites insert their own fixtures). Re-run `seed.ingest_knowledge_base` before using
> the live coach after a test run.

Re-recording the eval fixtures (live API, manual, rarely needed):

```bash
cd backend && python -m tests.fixtures.record_fixtures            # all three suites
cd backend && python -m tests.fixtures.record_fixtures --mock     # plumbing check, no spend
```

## CI

`.github/workflows/ci.yml` runs on every push and PR: ruff lint + format check, then
the full `pytest` suite against a `pgvector/pgvector:pg16` service and a Redis service,
on Python 3.13. `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` are set explicitly empty so CI
can never become a live-API run.

## Secrets

All secrets live in a gitignored root `.env` (never committed). Required:
`ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `DATABASE_URL` (+ Supabase service key),
`UPSTASH_REDIS_URL`, `JWT_SECRET`.
