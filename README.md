# Stripe Coding Assistant

### Author: [Ali Eddeb](https://www.linkedin.com/in/ali-eddeb/)

### Date completed: August 17, 2026

This project is my capstone submission for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp), a free course on building LLM applications by DataTalksClub. I built a retrieval-augmented (RAG) assistant that answers plain-English questions about integrating Stripe payments. Every answer is grounded in the official Stripe documentation and cites its sources. When the answer is an executable payment flow, the assistant can also prove it by running the flow in Stripe's test sandbox and showing the real API responses and webhook events.

**Live demo:** https://stripe-coding-assistant.streamlit.app (runs on free tiers, so the first load after idling can take ~30 seconds)

## Table of Contents

- [Introduction](#introduction)
- [Approach](#approach)
- [How a question is answered](#how-a-question-is-answered)
- [Dataset](#dataset)
- [Retrieval evaluation](#retrieval-evaluation)
- [LLM evaluation](#llm-evaluation)
- [Stripe Sandbox](#stripe-sandbox)
- [Monitoring](#monitoring)
- [How to run it](#how-to-run-it)
- [Cloud deployment](#cloud-deployment)
- [Next steps](#next-steps)

## Introduction

A developer integrating payments today has three options, and each one has a problem:

1. **Read the docs.** Stripe's documentation is excellent but enormous (~490 pages in its LLM-ready channel alone, plus the API reference), and it is organized by product rather than by use case. The answer to "subscription with a free trial that prorates on upgrade" is stitched together from four or five separate pages.
2. **Ask a generic AI.** LLMs hallucinate plausible-but-wrong parameter names and pin stale API versions. Stripe's own `llms.txt` opens with a warning to never trust version numbers from training data.
3. **Ask Stack Overflow.** Tens of thousands of `stripe`-tagged questions show the demand is real, but answers are slow to arrive and go stale as the API evolves.

The stakes are higher than ordinary documentation friction. Payments code that is mostly right still loses money: double fulfillment from a mishandled webhook, missed events, broken refund flows. Most real payments bugs live in the asynchronous parts (payments finish via webhooks, not API responses), which is exactly where copy-pasted answers fail silently.

The goal of this project was to build an assistant whose answers you can actually verify.

## Approach

The assistant makes three commitments:

- **Grounded** — answers are generated only from retrieved documentation excerpts and cite their sources, so you can check the primary text in one click. If retrieval finds nothing relevant, the assistant refuses instead of answering from the model's memory.
- **Proven** — for executable flows, the assistant runs the recommended integration against Stripe's test-mode sandbox and shows the trace: each API call with its exact payload, the responses, and the webhook events that fired, in order.
- **Measured** — retrieval and answer quality are evaluated against real developer questions curated from Stack Overflow, with the numbers reported below.

Stripe does have its own docs AI, but it is a black box: you cannot see how it retrieves or measure how well it does. This project's claim is a transparent, reproducible, measured retrieval system on a real corpus. The same pattern (docs RAG plus an execution sandbox) would generalize to any API vendor.

## How a question is answered

Each question passes through four stages:

1. **Router** — one small, cheap LLM call decides whether the question is about Stripe at all and whether it is several questions in one (max 3). It also rewrites each part into a search query phrased the way the docs phrase things, since retrieval works better on "authorize payment manual capture" than on "charge him but hold the money". Off-topic questions are refused here and the main model never runs.
2. **Retrieval** — the rewritten queries search ~7k embedded chunks of the official docs using the winning config from the evaluation below. Multi-part questions retrieve per part; results are merged, deduplicated, and capped.
3. **Grounded generation** — the model answers from the retrieved excerpts only, citing each claim with `[n]` markers. Partly-covered questions get a partial answer that names what the docs do not address, instead of a refusal.
4. **Logging** — question, answer, router verdict, retrieved chunks, and latency land in Postgres for the monitoring dashboard.

Since this is a public demo, there are guardrails: retrieved text and user text are treated as data, never instructions, at both LLM layers; pasted API keys are redacted before anything is stored or sent; and three stacked limits (per-session, per-minute, and a database-backed daily cap) bound the worst-case LLM spend.

## Dataset

The corpus is Stripe's official documentation, fetched from the [`llms.txt`](https://docs.stripe.com/llms.txt) index that Stripe publishes for LLM use. The ingestion pipeline (built with dlt) fetches the index and every page it lists (~440 pages, politely rate-limited), stores versioned snapshots in Postgres, splits each page at its markdown headings into ~7k chunks, and embeds them with a local ONNX model into pgvector. Re-running ingestion only writes new versions for pages whose content actually changed.

Please note that the raw fetched data is not saved in this repo (the `data/` folder is gitignored), but the whole corpus can be rebuilt with the commands in [How to run it](#how-to-run-it).

## Retrieval evaluation

I wanted the question set to be real developer questions, not questions generated from the corpus itself. I collected 119 high-vote `stripe`-tagged Stack Overflow questions and kept the 43 whose canonical answer page exists in the corpus, hand-mapping each one to that page. Questions answered by pages outside the docs corpus (dashboard-only workflows, deprecated APIs) were dropped rather than force-mapped. I then added 12 synthetic questions, written from corpus pages, to cover product areas the Stack Overflow set does not reach (disputes, payouts, Radar, Terminal, Identity). That gives **55 questions**, each labeled with the exact doc page that answers it (`evals/ground_truth.jsonl`).

Scoring is strict: a retrieved chunk counts only on an exact URL match with the labeled answer page. A chunk from a neighbouring page scores zero.

I swept four retrieval approaches (plus keyword-strictness variants) over k ∈ {5, 10, 20}, plus re-ranked variants of the two strongest modes — 21 configurations in total, all logged to `rag.experiments`:

| config (k=10) | SO hit rate | SO MRR | synth hit rate | synth MRR |
|---|---|---|---|---|
| **vector (cosine)** | **0.698** | 0.464 | 0.833 | 0.597 |
| vector + re-rank | 0.674 | **0.470** | 0.833 | 0.611 |
| hybrid + RRF | 0.698 | 0.417 | **0.833** | 0.639 |
| hybrid + RRF + re-rank | 0.674 | 0.446 | 0.833 | **0.708** |
| hybrid + RRF (kw-any) | 0.674 | 0.421 | 0.833 | 0.639 |
| keyword FTS (any-term) | 0.395 | 0.174 | 0.750 | 0.581 |
| keyword FTS (all-terms) | 0.209 | 0.085 | 0.417 | 0.347 |

Findings:

- **Winner: vector search, k=10** — best MRR overall (0.49 weighted) and tied-best hit rate (0.73). This is the serving config in the app.
- Hybrid + RRF is a close second. The keyword arm rescues lexical queries (error strings, event names) that embeddings miss, but the fusion dilutes rank quality on questions vector alone handles well.
- Keyword-only search collapses on real questions (0.09–0.19 MRR). Developers do not phrase questions in the docs' vocabulary, which is the gap that makes RAG worth building here.
- Raising k from 5 to 10 improves hit rate everywhere; 10 to 20 buys little for vector search, so serving uses k=10 to keep the LLM context small.

I also implemented a second-stage re-ranker (overfetch 3×k candidates, re-score with normalized vector + keyword signals, keep the top k). On the real-question set it moved MRR by +0.006 and hit rate by −0.024 at k=10, inside the noise of a 43-question set, so serving keeps plain vector search and skips the overfetch latency. Evaluating and declining a technique is still a result.

Query rewriting happens before retrieval, in the router (`services/router.py`).

Reproduce with `make eval`; `make experiments` lists logged runs.

## LLM evaluation

With retrieval fixed at the serving config, I compared three answer-prompt variants over all 55 ground-truth questions — identical retrieved context, different instructions to the model:

- **v1-grounded-cite** — grounding, citation, and scope rules; structure left to the model.
- **v2-answer-first** — same rules, plus a mandated answer shape: direct answer → numbered steps → caveats.
- **v3-explained-summary** — v2's shape, but the opening section must be a beginner-facing summary: name the Stripe pieces involved, define terms on first use, and explain why this is the right approach.

An LLM judge scored every answer on faithfulness (are the claims supported by the excerpts?) and relevance (does it answer the question?), each 0–2. Citation discipline is checked programmatically, not by the judge.

| variant | faithfulness | relevance | both perfect | cited | avg length |
|---|---|---|---|---|---|
| v1-grounded-cite | 1.84 / 2 | 1.98 / 2 | 84% | 98% | 906 chars |
| v2-answer-first | 1.91 / 2 | 1.98 / 2 | 91% | 91% | 745 chars |
| **v3-explained-summary** | **1.93 / 2** | **2.00 / 2** | **93%** | 87% | 1225 chars |

**Winner: v3-explained-summary**, which is the app's serving prompt. The richer explanation did not loosen grounding; the trade-offs are longer answers (~65% more than v2) and a small dip in citation discipline (87% vs 91% of answers carry `[n]` markers). One caveat: the judge and the answerer come from the same provider chain, so treat these as a relative comparison between variants, not an absolute quality score.

Reproduce with `make llm-eval`. Every LLM call is disk-cached (`data/llm_cache/`), so re-runs cost zero API calls.

## Stripe Sandbox

This is the differentiating feature: for executable flows, the assistant does not just cite the docs, it runs the recommendation against Stripe's test-mode sandbox and shows the receipt.

Ask *"charge $50 but hold the money until I ship"* and the answer arrives with a **Stripe Sandbox** section that shows the exact API payloads it will send. One click executes the flow with real API calls (no real money):

1. `PaymentIntent.create` (manual capture, confirmed with Stripe's test Visa) → status `requires_capture` — the hold is placed
2. `PaymentIntent.capture` → status `succeeded` — the funds move

It then shows each call's full API response and the events Stripe recorded for the flow, in lifecycle order (`payment_intent.created` → ... → `payment_intent.succeeded`) — exactly what a webhook endpoint would have received. This makes the asynchronous part of payments, where most integration bugs hide, visible.

Safety is structural, not behavioral:

- **Test mode only** — execution refuses any key that is not `sk_test_...`.
- **Whitelisted flows** — a question can only trigger a vetted, hardcoded call sequence (`agent/sandbox.py`). Model output never constructs API calls.

Try it standalone: `uv run --env-file .env python -m agent.sandbox`

## Monitoring

Every exchange is logged to Postgres, and every answer takes 👍/👎 feedback in the UI. A provisioned Grafana instance (started by `make up`, at `localhost:3000`, read-only DB role) ships with a six-panel dashboard: questions over time, route distribution (answered / refused / errored), feedback counts, answer latency, top retrieval score per question, and a live table of recent questions.

## How to run it

Everything runs from one `docker-compose.yml` (Postgres with pgvector, Grafana, and the Streamlit app), plus [uv](https://docs.astral.sh/uv/) for the pipeline scripts. Dependencies are pinned in `uv.lock`.

```bash
cp .env.example .env    # set POSTGRES_PASSWORD; add API keys (below)
make up                 # Postgres + Grafana + app (builds the app image once)
make schema             # create schemas + Grafana role (idempotent)
make model              # download the embedding model (~90 MB, once)
make ingest chunk embed # build the corpus (~440 pages -> ~7k embedded chunks)
```

The app is now at `localhost:8501` and Grafana at `localhost:3000`. For development, `make app` runs the app outside Docker against the same database. All pipeline steps are safe to re-run.

API keys, all optional except one LLM provider: `GEMINI_API_KEY` or `GROQ_API_KEY` (free tiers) or `OPENAI_API_KEY` — the client tries them in that order. `STRIPE_SECRET_KEY` (a **test-mode** key from the [Stripe dashboard](https://dashboard.stripe.com/test/apikeys)) enables the sandbox; without it the app still answers questions.

Reproduce the evaluations with `make eval` (retrieval grid) and `make llm-eval` (prompt comparison); both log to `rag.experiments`.

## Cloud deployment

The live demo runs on free tiers:

- **App** — Streamlit Community Cloud, deployed from this repo.
- **Database** — [Neon](https://neon.tech) serverless Postgres with pgvector, loaded with the same corpus as local. The app connects through `DATABASE_URL` when it is set, so the same code serves both environments.
- **Secrets** — kept in Streamlit's secrets manager, not in the repo.

The local Grafana can read the cloud database too, which is how the dashboard shows real visitor traffic. Create the read-only role on it once:

```bash
DATABASE_URL=<neon connection string> \
GRAFANA_DB_PASSWORD=<pick a password> \
  uv run python -m db.grafana_role
```

Then set `NEON_DB_HOST`, `NEON_DB_NAME`, and `NEON_GRAFANA_DB_PASSWORD` in `.env` and restart Grafana. Leave them unset and the dashboard simply runs against the local database.

## Next steps

- More sandbox flows (refunds, subscriptions with trials) — the flow registry is built to take them.
- Scheduled re-ingestion so the corpus tracks doc changes automatically.
- A cost analysis of serving on paid tiers.

---

<strong>Thank you for taking the time to look at my project. If you have any comments, feedback or suggestions, please reach out.</strong>

#### My contact info:

<strong> linkedIn: www.linkedin.com/in/ali-eddeb </strong>
