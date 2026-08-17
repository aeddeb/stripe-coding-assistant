"""Record and list evaluation runs (``rag.experiments``).

Every eval run should end with one ``log_experiment()`` call. Results then
survive as queryable rows instead of terminal scrollback, and the README's
comparison table is a ``SELECT`` away.

List recent runs::

    uv run --env-file .env python -m evals.experiments
"""

from __future__ import annotations

import json

from services.db import connect


def log_experiment(
    name: str,
    config: dict,
    hit_rate: float | None = None,
    mrr: float | None = None,
    n_questions: int | None = None,
    notes: str = "",
    extra: dict | None = None,
) -> int:
    """Insert one experiment row and return its id."""
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO rag.experiments (name, config, n_questions, hit_rate, mrr, notes, extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                name,
                json.dumps(config),
                n_questions,
                hit_rate,
                mrr,
                notes,
                json.dumps(extra) if extra is not None else None,
            ),
        ).fetchone()
        conn.commit()
        return row[0]


def show_recent(limit: int = 20) -> None:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, ran_at::timestamp(0), name, n_questions, hit_rate, mrr, notes
            FROM rag.experiments
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    if not rows:
        print("No experiments logged yet.")
        return
    print(f"{'id':>4}  {'ran_at':19}  {'name':42}  {'n':>4}  {'hit_rate':>8}  {'mrr':>6}  notes")
    for id_, ran_at, name, n, hit_rate, mrr, notes in rows:
        hr = f"{hit_rate:.3f}" if hit_rate is not None else "-"
        mr = f"{mrr:.3f}" if mrr is not None else "-"
        print(f"{id_:>4}  {str(ran_at):19}  {name[:42]:42}  {n or '-':>4}  {hr:>8}  {mr:>6}  {notes or ''}")


if __name__ == "__main__":
    show_recent()
