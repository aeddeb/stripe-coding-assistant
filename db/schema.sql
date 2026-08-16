-- Serving-side schema. The dlt ingestion pipeline owns the `stripe_docs`
-- schema (raw pages, SCD2 history); everything derived for retrieval and
-- evaluation lives here in `rag`.
--
-- Idempotent: safe to re-run at any time.
--
-- Apply:  make schema
--   (or)  docker compose exec -T postgres psql -U app -d stripe_assistant < db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag;

-- One row per retrievable chunk. Rebuilt wholesale by `ingestion/chunking.py`
-- (the corpus is small enough that full rebuilds beat incremental bookkeeping),
-- then `ingestion/embed.py` fills the embedding column.
CREATE TABLE IF NOT EXISTS rag.chunks (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    page_url     text NOT NULL,   -- source page (citation target)
    page_title   text,
    section      text,            -- product area from the llms.txt index
    heading_path text,            -- e.g. "Refunds > Partial refunds"
    chunk_index  int  NOT NULL,   -- position of the chunk within its page
    content      text NOT NULL,

    -- 384 dims = all-MiniLM-L6-v2. Filled by ingestion/embed.py.
    embedding    vector(384),

    -- Full-text index input for keyword search. Title and headings are
    -- included so a query can match a chunk by what it is about, not only
    -- by the words inside it.
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(page_title, '') || ' ' ||
            coalesce(heading_path, '') || ' ' ||
            content)
    ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_fts_idx ON rag.chunks USING gin (fts);

-- No ANN index (ivfflat/hnsw) on purpose: at this corpus size (a few
-- thousand rows) exact scan is fast and gives exact results, which matters
-- when comparing retrieval methods.

-- One row per evaluation run: the retrieval configuration tried and the
-- metrics it scored. The comparison table in the README is a query over
-- this table.
CREATE TABLE IF NOT EXISTS rag.experiments (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ran_at      timestamptz NOT NULL DEFAULT now(),
    name        text  NOT NULL,   -- short label, e.g. "hybrid+rrf k=60"
    config      jsonb NOT NULL,   -- full RetrievalConfig, reproducible
    n_questions int,
    hit_rate    double precision,
    mrr         double precision,
    notes       text,
    extra       jsonb             -- anything else: judge scores, latency, ...
);
