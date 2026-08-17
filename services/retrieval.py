"""Retrieval over ``rag.chunks`` — the single search entry point.

Design: one ``search()`` function driven by a config object, so an
evaluation run is a loop over configs and the app serves whatever config
won. Both the eval harness and the app import THIS module — the numbers in
the eval table describe the exact code path users hit.

Status: interface defined, implementation in progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import psycopg

from services.embedder import Embedder


@dataclass(frozen=True)
class RetrievalConfig:
    """Everything that can vary between retrieval experiments."""

    mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    top_k: int = 5
    # Keyword matching: "all" requires every meaningful query word in a
    # chunk (Postgres default, precise but can return nothing for wordy
    # questions); "any" matches chunks containing any query word and
    # lets ranking sort them. A config knob so it can be compared in
    # experiments without touching the search code.
    keyword_match: Literal["all", "any"] = "all"
    # Hybrid fusion (Reciprocal Rank Fusion) constant.
    rrf_k: int = 60
    # How many candidates each arm contributes before fusion/re-ranking.
    candidates: int = 20
    rerank: bool = False
    query_rewrite: bool = False
    # Free-form knobs that don't deserve their own field yet.
    extra: dict = field(default_factory=dict)

    def label(self) -> str:
        parts = [self.mode]
        if self.mode == "hybrid":
            parts.append(f"rrf{self.rrf_k}")
        if self.mode in ("keyword", "hybrid") and self.keyword_match != "all":
            parts.append(f"kw-{self.keyword_match}")
        if self.rerank:
            parts.append("rerank")
        if self.query_rewrite:
            parts.append("rewrite")
        return "+".join(parts)


@dataclass
class Hit:
    chunk_id: int
    page_url: str
    page_title: str | None
    heading_path: str | None
    content: str
    score: float


def search(query: str, cfg: RetrievalConfig, conn: psycopg.Connection) -> list[Hit]:
    """Retrieve the ``cfg.top_k`` most relevant chunks for ``query``.

    Contract:
    - ``keyword``: Postgres full-text search over ``rag.chunks.fts``,
      ranked with ``ts_rank``.
    - ``vector``: cosine similarity between the query embedding
      (``services.embedder``) and ``rag.chunks.embedding``.
    - ``hybrid``: run both arms for ``cfg.candidates`` each, fuse with RRF,
      return the fused top-k.
    - ``rerank`` / ``query_rewrite`` apply on top of any mode.

    Results come back best-first.
    """
    if cfg.query_rewrite:
        # Query rewriting lives in services.router (one LLM call that
        # rewrites/splits the question before retrieval); this flag stays
        # unimplemented at the search layer. Fail loudly rather than let
        # an experiment claim a feature that silently didn't run.
        raise NotImplementedError("query_rewrite happens in services.router")
    if cfg.rerank:
        # Retrieve-then-rerank: overfetch with the base mode, then
        # re-score that pool and keep the top_k (see _rerank).
        pool_cfg = replace(
            cfg, rerank=False, top_k=max(cfg.candidates, 3 * cfg.top_k)
        )
        return _rerank(query, search(query, pool_cfg, conn), conn, cfg.top_k)
    if cfg.mode == "keyword":
        return _keyword(query, conn, cfg.top_k, cfg.keyword_match)
    if cfg.mode == "vector":
        return _vector(query, conn, cfg.top_k)
    if cfg.mode == "hybrid":
        return _hybrid(query, conn, cfg)
    raise NotImplementedError(f"mode {cfg.mode!r} not implemented yet")


# The embedding model is loaded once per process and reused: eval runs call
# search() thousands of times, and reloading ONNX weights per call would
# dominate the runtime.
_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _tsquery_sql(match: Literal["all", "any"]) -> str:
    """SQL fragment that parses the user's question into a tsquery.

    "all": Postgres default — every meaningful word must appear.
    "any": rewrite the parsed query's AND separators to OR, so a chunk
    containing any of the words qualifies and ranking sorts the rest.
    """
    base = "websearch_to_tsquery('english', %(query)s)"
    if match == "all":
        return base
    return f"to_tsquery('english', replace({base}::text, ' & ', ' | '))"


def _keyword(
    query: str,
    conn: psycopg.Connection,
    limit: int,
    match: Literal["all", "any"] = "all",
) -> list[Hit]:
    """Full-text search over ``rag.chunks.fts``, ranked with ``ts_rank``.

    Used both as the standalone ``keyword`` mode and as the keyword arm
    of ``hybrid`` (which calls it with ``cfg.candidates``).
    """
    sql = f"""
        SELECT id, page_url, page_title, heading_path, content,
               ts_rank(fts, q) AS score
        FROM rag.chunks, {_tsquery_sql(match)} AS q
        WHERE fts @@ q
        ORDER BY score DESC
        LIMIT %(limit)s
    """
    rows = conn.execute(sql, {"query": query, "limit": limit}).fetchall()
    return [Hit(*row) for row in rows]


def _vector(query: str, conn: psycopg.Connection, limit: int) -> list[Hit]:
    """Cosine-similarity search over ``rag.chunks.embedding``.

    The question is embedded bare (chunks carried a "title — heading"
    prefix at index time; a question is already self-contained). Score is
    ``1 - cosine distance`` so higher = better, like the other modes.
    Used both as the standalone ``vector`` mode and as the vector arm of
    ``hybrid`` (which calls it with ``cfg.candidates``).
    """
    qvec = _get_embedder().encode(query)
    sql = """
        SELECT id, page_url, page_title, heading_path, content,
               1 - (embedding <=> %(qvec)s) AS score
        FROM rag.chunks
        ORDER BY embedding <=> %(qvec)s
        LIMIT %(limit)s
    """
    rows = conn.execute(sql, {"qvec": qvec, "limit": limit}).fetchall()
    return [Hit(*row) for row in rows]


def _hybrid(query: str, conn: psycopg.Connection, cfg: RetrievalConfig) -> list[Hit]:
    """Fuse keyword and vector rankings with Reciprocal Rank Fusion.

    Scores from the two arms live on different scales, so RRF ignores
    them and uses rank positions only: a chunk earns
    ``1 / (rrf_k + rank)`` points per list it appears on, totals are
    summed, and the fused top-k comes back with the RRF total as
    ``score``. Chunks found by both arms accumulate twice — that overlap
    reward is the point of hybrid search.
    """
    arms = [
        _keyword(query, conn, cfg.candidates, cfg.keyword_match),
        _vector(query, conn, cfg.candidates),
    ]
    totals: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for arm in arms:
        for rank, hit in enumerate(arm, start=1):
            totals[hit.chunk_id] = totals.get(hit.chunk_id, 0.0) + 1 / (cfg.rrf_k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    fused = sorted(first_seen.values(), key=lambda h: totals[h.chunk_id], reverse=True)
    top = fused[: cfg.top_k]
    for hit in top:
        hit.score = totals[hit.chunk_id]
    return top


def _rerank(
    query: str, candidates: list[Hit], conn: psycopg.Connection, top_k: int
) -> list[Hit]:
    """Second-stage re-ranking: re-score an overfetched candidate pool with
    two signals the pool's own ordering may have ignored.

    For every candidate, compute vector cosine similarity AND keyword
    ``ts_rank`` (loose any-term match), min-max normalize each across the
    pool, and combine ``0.7 * vector + 0.3 * keyword``. Semantic evidence
    stays dominant (it won the first-stage eval); the keyword signal breaks
    ties for candidates that literally contain the question's terms —
    error strings, event names, parameter names.

    Cost: two indexed SQL queries over ~3x top_k ids. No model calls
    beyond the one query embedding.
    """
    if not candidates:
        return []
    ids = [h.chunk_id for h in candidates]
    qvec = _get_embedder().encode(query)
    vec_scores = dict(
        conn.execute(
            """
            SELECT id, 1 - (embedding <=> %(qvec)s) AS score
            FROM rag.chunks WHERE id = ANY(%(ids)s)
            """,
            {"qvec": qvec, "ids": ids},
        ).fetchall()
    )
    kw_scores = dict(
        conn.execute(
            f"""
            SELECT id, ts_rank(fts, q) AS score
            FROM rag.chunks, {_tsquery_sql("any")} AS q
            WHERE id = ANY(%(ids)s) AND fts @@ q
            """,
            {"query": query, "ids": ids},
        ).fetchall()
    )

    def normalize(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {i: 1.0 for i in scores}
        return {i: (s - lo) / (hi - lo) for i, s in scores.items()}

    vec_n, kw_n = normalize(vec_scores), normalize(kw_scores)
    combined = {
        h.chunk_id: 0.7 * vec_n.get(h.chunk_id, 0.0) + 0.3 * kw_n.get(h.chunk_id, 0.0)
        for h in candidates
    }
    ranked = sorted(candidates, key=lambda h: combined[h.chunk_id], reverse=True)
    top = ranked[:top_k]
    for hit in top:
        hit.score = combined[hit.chunk_id]
    return top


def _main() -> None:
    """Ad-hoc search from the command line::

        uv run --env-file .env python -m services.retrieval \\
            "why did my webhook fire twice?" --mode hybrid -k 5
    """
    import argparse

    from services.db import connect

    parser = argparse.ArgumentParser(description="Search rag.chunks")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--keyword-match", choices=["all", "any"], default="all")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=20)
    args = parser.parse_args()

    cfg = RetrievalConfig(
        mode=args.mode,
        top_k=args.top_k,
        keyword_match=args.keyword_match,
        rrf_k=args.rrf_k,
        candidates=args.candidates,
    )
    with connect() as conn:
        hits = search(args.question, cfg, conn)
    print(f"[{cfg.label()}] {len(hits)} hits")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank:>2}. {hit.score:.4f}  {hit.page_url}")
        print(f"      {hit.heading_path or hit.page_title or ''}")


if __name__ == "__main__":
    _main()
