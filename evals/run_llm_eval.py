"""LLM evaluation: compare answer-prompt variants with an LLM judge.

For every ground-truth question, retrieval runs ONCE with the serving
config; each prompt variant then generates an answer over that identical
context, so the comparison isolates the LLM step. A judge model scores each
answer on two 0-2 scales:

- **faithfulness** — is every claim supported by the provided excerpts?
- **relevance** — does it actually answer the developer's question?

Citation discipline ([n] markers present) is checked programmatically, not
by the judge. Aggregates are logged to ``rag.experiments`` (means land in
the ``extra`` JSON column).

Caveat, stated openly: judge and answerer come from the same provider
chain, so scores are a relative comparison between variants, not an
absolute quality certificate.

Run: make llm-eval  (all calls are disk-cached; re-runs are free)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.experiments import log_experiment
from evals.run_retrieval_eval import Question, load_ground_truth
from services.answer import PROMPT_VARIANTS, generate_answer
from services.db import connect
from services.llm import chat
from services.retrieval import Hit, RetrievalConfig, search

DEFAULT_GROUND_TRUTH = Path(__file__).with_name("ground_truth.jsonl")

# Must match app.app.SERVING_CONFIG — the eval should describe what users get.
SERVING_CONFIG = RetrievalConfig(mode="vector", top_k=10)

CITATION_RE = re.compile(r"\[\d+\]")

JUDGE_PROMPT = """\
You are grading an AI assistant's answer to a Stripe integration question.
You are given the documentation excerpts the assistant saw, the question,
and the answer. Grade two things:

- faithfulness (0-2): 2 = every claim is supported by the excerpts;
  1 = mostly supported but at least one unsupported or embellished claim;
  0 = contradicts the excerpts or is substantially unsupported.
- relevance (0-2): 2 = directly and completely answers the question;
  1 = partially answers or buries the answer; 0 = misses the question.

Reply with ONLY a JSON object: {{"faithfulness": <0|1|2>, "relevance":
<0|1|2>, "reason": "<one short sentence>"}}

EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""


def judge(question: str, context: str, answer: str) -> dict | None:
    raw = chat(
        JUDGE_PROMPT.format(context=context, question=question, answer=answer),
        temperature=0.0,
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
        return {
            "faithfulness": int(scores["faithfulness"]),
            "relevance": int(scores["relevance"]),
            "reason": str(scores.get("reason", "")),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _context_for_judge(hits: list[Hit], max_chars: int = 3000) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(f"[{i}] {hit.page_url}\n{hit.content[:max_chars]}")
    return "\n\n---\n\n".join(parts)


def evaluate_question(question: Question, hits: list[Hit]) -> dict[str, dict]:
    """Generate + judge every prompt variant for one question."""
    context = _context_for_judge(hits)
    results: dict[str, dict] = {}
    for variant, system_prompt in PROMPT_VARIANTS.items():
        answer = generate_answer(question.question, hits, system_prompt)
        verdict = judge(question.question, context, answer)
        results[variant] = {
            "scores": verdict,
            "cited": bool(CITATION_RE.search(answer)),
            "answer_chars": len(answer),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--limit", type=int, default=None, help="cap question count")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="don't log to rag.experiments")
    args = parser.parse_args()

    questions = load_ground_truth(args.ground_truth)
    if args.limit:
        questions = questions[: args.limit]
    print(
        f"{len(questions)} questions x {len(PROMPT_VARIANTS)} prompt variants "
        f"(retrieval: {SERVING_CONFIG.label()} k={SERVING_CONFIG.top_k})",
        file=sys.stderr,
    )

    # Retrieval up front on one connection (psycopg connections are not
    # shared across threads); LLM calls are what parallelism is for.
    with connect() as conn:
        retrieved = [search(q.question, SERVING_CONFIG, conn) for q in questions]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        per_question = list(pool.map(evaluate_question, questions, retrieved))

    for variant in PROMPT_VARIANTS:
        rows = [pq[variant] for pq in per_question]
        judged = [r["scores"] for r in rows if r["scores"] is not None]
        faith = statistics.mean(s["faithfulness"] for s in judged)
        rel = statistics.mean(s["relevance"] for s in judged)
        perfect = sum(
            1 for s in judged if s["faithfulness"] == 2 and s["relevance"] == 2
        ) / len(judged)
        cited = sum(1 for r in rows if r["cited"]) / len(rows)
        chars = statistics.mean(r["answer_chars"] for r in rows)
        unparsed = len(rows) - len(judged)
        print(
            f"\n{variant}: faithfulness {faith:.2f}/2  relevance {rel:.2f}/2  "
            f"both-perfect {perfect:.0%}  cited {cited:.0%}  "
            f"avg {chars:.0f} chars"
            + (f"  ({unparsed} judge parses failed)" if unparsed else "")
        )
        if not args.dry_run:
            log_experiment(
                name=f"llm-eval {variant}",
                config={
                    "variant": variant,
                    "retrieval": SERVING_CONFIG.label(),
                    "top_k": SERVING_CONFIG.top_k,
                },
                n_questions=len(rows),
                notes=f"LLM-as-judge over {args.ground_truth.name}",
                extra={
                    "faithfulness_mean": round(faith, 3),
                    "relevance_mean": round(rel, 3),
                    "both_perfect_rate": round(perfect, 3),
                    "citation_rate": round(cited, 3),
                    "avg_answer_chars": round(chars),
                    "judge_parse_failures": unparsed,
                },
            )
    if not args.dry_run:
        print(f"\nLogged {len(PROMPT_VARIANTS)} rows to rag.experiments.")


if __name__ == "__main__":
    main()
