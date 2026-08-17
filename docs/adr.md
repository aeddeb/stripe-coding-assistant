# Architecture Decisions Record

The choices that shaped this project, what they were chosen over, and why —
in plain language. One table, one row per decision (as of Aug 2026; rows are
added as the build progresses).

| # | Decision | Instead of | Why |
|---|----------|------------|-----|
| 1 | **Ingest with dlt, from Stripe's `llms.txt` channel** | A hand-rolled fetch script; a heavy orchestrator (Airflow/Kestra); scraping the HTML docs | `llms.txt` is Stripe's own index published for LLM use, and every page it lists is served as clean markdown — no scraping, no permission gray zone. dlt adds what a bare script lacks (incremental loading, typed tables, schema versioning) for the cost of a library, not a server. Schema stays in dlt's default "evolve" mode with changes logged after every run: our code shapes each row explicitly, so the source can't inject surprise columns. |
| 2 | **Keep every page version (SCD2 history)** | Overwriting pages in place on each sync | Docs change constantly. Storing old versions with validity timestamps makes "what changed since last month" a simple query, and feeds a docs-drift monitoring chart. Stripe's servers send no change headers, so change detection is by content hash: every sync refetches all pages but only writes the ones that actually changed. The exact files as fetched are also kept on disk, as untouched originals. |
| 3 | **Stack Overflow grades the system but never feeds it** | Blending community answers into the answer corpus; grading only with generated questions | SO answers rot as the API evolves — putting them in the corpus would inject the exact staleness this project exists to avoid. But SO *questions* are how developers really phrase problems, so a hand-curated set of real questions is the honest exam. Questions auto-generated from our own docs reuse the docs' wording and inflate retrieval scores; both sets are used and reported separately. |
| 4 | **One Postgres for everything** | Adding a dedicated vector database (Qdrant, Elasticsearch, …) | The corpus is a few thousand chunks. Postgres handles vector similarity (pgvector) and keyword search (built-in full-text) in one place — and hybrid search needs both anyway. One connection string, one backup, one thing to run. Two layers inside it: a raw layer owned by the pipeline (pages, history) and a hand-defined serving layer (chunks, results) that consumers read. At this size vector search runs exact, with no approximate index — results are exact, which keeps method comparisons clean. |
| 5 | **Structure-aware chunking, bounded 200–1,200 characters** | Fixed-size chunks; "semantic" (AI-detected) splitting | Stripe's docs are clean markdown, so their own headings are reliable cut lines: each chunk is one idea, tagged with its heading trail for citations. The ceiling is the embedding model's reading window, not a style choice: all-MiniLM-L6-v2 reads ~256 tokens (≈1,200 characters) and silently ignores the rest, so any larger chunk would be partially invisible to vector search (keyword search still sees all of it). The floor merges crumbs into their neighbor. Chunks are embedded with a "page title — heading" prefix so short chunks carry their context. The strategy stays fixed while search methods are compared: one variable at a time, and no re-embedding the corpus per experiment. |
| 6 | **Small local embedding model: all-MiniLM-L6-v2 via ONNX** | Paid API embeddings; a larger local model | Free, fast on a plain CPU — which matters twice: the dev machine and the free demo host. ONNX is a lighter way to run the same model (a small runtime instead of the full PyTorch stack), shrinking the install by gigabytes. It's also a cheap decision to revisit: swapping models is one name change plus minutes of re-embedding, and the eval numbers decide whether an upgrade earns its cost. |
| 7 | **Free-tier LLMs with fallback: Gemini Flash, then Groq — no card on file** | One paid API provider | The build should cost near zero and the public demo must not be able to surprise-bill: with no card attached, the provider's rate limit *is* the budget cap. Both providers speak the same API, so one client covers them and a rate-limit on the primary swaps providers, not code. Responses are cached on disk keyed by the request, so repeated evaluation runs cost zero API calls. |

## Pending — rows added when their phase is built

- **Agent routing: the LLM's tool-choice is the router** (vs a hardcoded
  question classifier) — written up when the agent loop exists.
- **Real Stripe test-mode sandbox, never a mock** — written up with the
  sandbox tool.
- **Deployment topology: ingest at home · Neon Postgres · Hugging Face
  Space** (incl. the Supabase rejection and budget guardrails) — written up
  at deploy.
