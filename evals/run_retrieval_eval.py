"""Retrieval evaluation — score every search configuration against a
ground-truth question set.

An experiment run takes questions whose correct answer page is already
known, sweeps a grid of retrieval configurations, and asks the same thing
of each: did the right page come back, and how near the top? Two numbers
summarise it — hit rate (the share of questions whose answer page appeared
anywhere in the top ``k``) and MRR (mean reciprocal rank, which rewards
ranking that page first rather than fifth). Metrics are reported per
question source and never pooled: questions harvested from Stack Overflow
and questions written against the docs are different difficulties, and one
average across both would hide which is which. Every (configuration,
source) pair becomes one row in ``rag.experiments``.

Run::

    uv run --env-file .env python -m evals.run_retrieval_eval
    uv run --env-file .env python -m evals.run_retrieval_eval --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import psycopg

from evals.experiments import log_experiment
from services.db import connect
from services.retrieval import RetrievalConfig, search

Mode = Literal["keyword", "vector", "hybrid"]
KeywordMatch = Literal["all", "any"]

DEFAULT_GROUND_TRUTH = Path(__file__).with_name("ground_truth.jsonl")

TOP_K_VALUES = (5, 10, 20)

# Mode plus keyword-matching strategy. ``keyword_match`` only changes
# behaviour where a keyword arm runs, so vector search appears once while
# keyword and hybrid appear in both strictnesses.
MODE_VARIANTS: list[tuple[Mode, KeywordMatch]] = [
    ("keyword", "all"),
    ("keyword", "any"),
    ("vector", "all"),
    ("hybrid", "all"),
    ("hybrid", "any"),
]

GRID: list[RetrievalConfig] = [
    RetrievalConfig(mode=mode, keyword_match=match, top_k=top_k)
    for mode, match in MODE_VARIANTS
    for top_k in TOP_K_VALUES
] + [
    # Second-stage re-ranking on top of the two strongest base modes:
    # overfetch 3x top_k, re-score with normalized vector+keyword signals.
    RetrievalConfig(mode=mode, rerank=True, top_k=top_k)
    for mode in ("vector", "hybrid")
    for top_k in TOP_K_VALUES
]


@dataclass(frozen=True)
class Question:
    """One evaluation question and the page that answers it."""

    question: str
    source: str
    answer_page_url: str


@dataclass(frozen=True)
class Score:
    """What one configuration scored on one source's questions."""

    n: int
    hit_rate: float
    mrr: float


def load_ground_truth(path: Path) -> list[Question]:
    """Read the JSONL question set, rejecting rows that cannot be scored."""
    questions: list[Question] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [
                key
                for key in ("question", "source", "answer_page_url")
                if not row.get(key)
            ]
            if missing:
                raise ValueError(f"{path}:{lineno} missing {', '.join(missing)}")
            questions.append(
                Question(
                    question=row["question"],
                    source=row["source"],
                    answer_page_url=row["answer_page_url"],
                )
            )
    if not questions:
        raise ValueError(f"{path} contains no questions")
    return questions


def reciprocal_rank(
    question: Question, cfg: RetrievalConfig, conn: psycopg.Connection
) -> float:
    """Score one question: ``1 / rank`` of the first hit on the answer page.

    Returns 0.0 when the answer page is absent from the results, which is
    what makes the mean of these values MRR and their nonzero share the hit
    rate. A page counts as correct only on an exact URL match, so a chunk
    from a neighbouring page earns nothing.
    """
    try:
        hits = search(question.question, cfg, conn)
    except Exception as exc:
        # Named so a failure mid-sweep says which config and question broke,
        # instead of a bare driver error 200 searches in.
        raise RuntimeError(
            f"search failed for [{cfg.label()} k={cfg.top_k}] {question.question!r}"
        ) from exc
    for rank, hit in enumerate(hits, start=1):
        if hit.page_url == question.answer_page_url:
            return 1 / rank
    return 0.0


def score_by_source(
    cfg: RetrievalConfig, questions: list[Question], conn: psycopg.Connection
) -> dict[str, Score]:
    """Run one configuration over every question, grouped by question source."""
    ranks: dict[str, list[float]] = {}
    for question in questions:
        ranks.setdefault(question.source, []).append(
            reciprocal_rank(question, cfg, conn)
        )
    return {
        source: Score(
            n=len(values),
            hit_rate=sum(1 for value in values if value > 0) / len(values),
            mrr=sum(values) / len(values),
        )
        for source, values in ranks.items()
    }


def print_table(results: list[tuple[RetrievalConfig, str, Score]]) -> None:
    """Print one block per source, best MRR first."""
    label_width = max(len(cfg.label()) for cfg, _, _ in results)
    source_width = max(len(source) for _, source, _ in results)
    header = (
        f"{'config':<{label_width}}  {'k':>3}  {'source':<{source_width}}  "
        f"{'n':>4}  {'hit_rate':>8}  {'mrr':>6}"
    )
    for position, source in enumerate(sorted({source for _, source, _ in results})):
        print(("\n" if position else "") + header)
        print("-" * len(header))
        rows = [row for row in results if row[1] == source]
        for cfg, _, score in sorted(rows, key=lambda row: row[2].mrr, reverse=True):
            print(
                f"{cfg.label():<{label_width}}  {cfg.top_k:>3}  {source:<{source_width}}  "
                f"{score.n:>4}  {score.hit_rate:>8.3f}  {score.mrr:>6.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score retrieval configurations against a ground-truth question set."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help=f"JSONL question set (default: {DEFAULT_GROUND_TRUTH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the results table without writing rows to rag.experiments",
    )
    args = parser.parse_args()
    if not args.ground_truth.exists():
        parser.error(f"ground truth file not found: {args.ground_truth}")

    questions = load_ground_truth(args.ground_truth)
    print(
        f"{len(questions)} questions x {len(GRID)} configs from {args.ground_truth}",
        file=sys.stderr,
    )

    results: list[tuple[RetrievalConfig, str, Score]] = []
    # One connection for the whole sweep: opening a new one per search would
    # cost more than the searches themselves.
    with connect() as conn:
        for position, cfg in enumerate(GRID, start=1):
            print(
                f"[{position}/{len(GRID)}] {cfg.label()} k={cfg.top_k}", file=sys.stderr
            )
            for source, score in score_by_source(cfg, questions, conn).items():
                results.append((cfg, source, score))

    print_table(results)

    if args.dry_run:
        print("\nDry run: nothing written to rag.experiments.")
        return

    for cfg, source, score in results:
        log_experiment(
            name=f"{cfg.label()} k={cfg.top_k} [{source}]",
            config=asdict(cfg) | {"source": source},
            hit_rate=score.hit_rate,
            mrr=score.mrr,
            n_questions=score.n,
            notes=f"retrieval grid over {args.ground_truth.name}",
        )
    print(f"\nLogged {len(results)} rows to rag.experiments.")


if __name__ == "__main__":
    main()
