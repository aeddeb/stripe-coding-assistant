# Stripe Coding Assistant

Ask a plain-English question about integrating Stripe payments. Get an answer
grounded in the official Stripe docs, with citations — and when the answer is an
executable flow, the assistant **proves it by running it in Stripe's test
sandbox**, showing the real API responses and webhook events.

> Built as a capstone for [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).
> Written for readers who haven't taken the course.

## The problem

A developer integrating payments today has three options, and all of them have
a trust problem:

**1. Read the docs.** Stripe's documentation is excellent — and enormous
(~490 pages in its LLM-ready channel alone, plus the API reference). It's
organized by product, not by *your use case*: the answer to "subscription with
a free trial that prorates on upgrade" is stitched together from four or five
separate pages. Stripe also ships overlapping products (Payment Links,
Checkout, Elements, raw API), and the docs explain each one — but not which
one you should pick.

**2. Ask a generic AI.** LLMs hallucinate plausible-but-wrong parameter names
and pin stale API versions. Stripe's own `llms.txt` opens with a warning to
never trust version numbers from training data — that's Stripe telling you
generic AI gets this wrong. Web search doesn't fix it: the stale web (old blog
posts, answers describing the deprecated Charges API) ranks high and gets
retrieved.

**3. Ask Stack Overflow.** Tens of thousands of `stripe`-tagged questions
prove the demand is real and sustained. But answers are slow to arrive and rot
as the API evolves.

The stakes make this worse than ordinary docs friction. Payments code that is
*mostly right* still loses money: double fulfillment from a mishandled webhook,
missed events, broken refund flows. Most real-world payments bugs live in the
asynchronous parts — payments finish via webhooks, not API responses — which is
exactly where copy-pasted answers fail silently.

## The solution

A retrieval-augmented assistant with one honest claim: **answers you can
verify.**

- **Grounded** — every answer is built from the official docs and cites its
  sources, so you can check the primary text in one click.
- **Proven** — for executable flows, an agent runs the recommended integration
  against Stripe's test-mode sandbox and shows the trace: each API call, the
  status transitions (`requires_capture` → `succeeded`), and the webhook events
  that fired, in order. Answer plus evidence, not answer plus vibes.
- **Measured** — retrieval quality is evaluated against real developer
  questions (curated from Stack Overflow), not just synthetic ones, with
  hit-rate and MRR numbers reported below.

**Doesn't Stripe already have a docs AI?** Yes — and that's evidence the
problem is real. But it's a black box: you can't see how it retrieves, and you
can't measure it. This project's claim is not novelty; it's a transparent,
reproducible, *measurably good* retrieval system on a real corpus — and the
pattern (docs RAG + execution sandbox) generalizes to any API vendor.

## What it looks like

Streamlit chat, two kinds of questions:

1. **How-to** — "charge $50 but hold the money until I ship" → cited answer →
   optional expandable sandbox execution trace.
2. **Concept** — "what happens when a customer disputes?" → cited docs answer.

Every answer takes 👍/👎 feedback, which feeds a monitoring dashboard.

## Build the corpus

Everything runs locally: Postgres (with pgvector) in Docker, Python via
[uv](https://docs.astral.sh/uv/). One-time setup:

```bash
cp .env.example .env    # then set a real POSTGRES_PASSWORD
make up                 # start Postgres in Docker
make schema             # create the serving schema (idempotent)
```

Then the pipeline, in order:

| Command | What it does | Sanity check |
|---------|--------------|--------------|
| `make ingest` | Fetches Stripe's [`llms.txt`](https://docs.stripe.com/llms.txt) index and every page it lists (~440, politely rate-limited), writes raw snapshots under `data/raw/<timestamp>/`, and loads pages into the `stripe_docs` schema with full version history | `select count(*) from stripe_docs.doc_pages;` |
| `make chunk` | Rebuilds `rag.chunks`: splits each page at its markdown headings into embedder-sized pieces, keeping the heading trail as metadata | `select count(*) from rag.chunks;` |
| `make model` | Downloads the ONNX embedding model (~90 MB, once) | — |
| `make embed` | Fills `rag.chunks.embedding` — each chunk is embedded with a `page title — heading path` prefix for context | `select count(*) from rag.chunks where embedding is null;` → 0 |

All steps are safe to re-run: `make ingest` only writes new versions for pages
whose content actually changed, `make chunk` rebuilds wholesale, and
`make embed` fills only missing vectors. `make corpus` = chunk + embed in one
go; `make psql` opens a SQL shell to poke around.

---

**Status: under active development.** Architecture, evaluation results, and
run/deployment instructions land here as each piece ships.

<!--
  Planned sections:
  - Architecture (diagram + component walkthrough)
  - Ingestion pipeline (incremental docs sync, chunking)
  - Retrieval (hybrid vector + full-text search, RRF fusion)
  - Agent (tool-calling loop: doc search + sandbox execution; query rewriting)
  - Evaluation (hit rate / MRR across retrieval approaches; LLM-as-judge on
    real developer questions)
  - Monitoring (user feedback loop + dashboards)
  - Running it (docker-compose quickstart, reproducibility)
  - Deployment (Hugging Face Spaces + Neon Postgres + free-tier LLM providers,
    with rate/output guardrails)
  - Cost analysis
  - Decisions and tradeoffs
-->
