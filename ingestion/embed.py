"""Fill missing embeddings in ``rag.chunks``.

Embeds only rows where ``embedding IS NULL``, so it is cheap to re-run and
picks up exactly what the last chunk rebuild left unfilled.

What gets embedded is not the raw chunk text but a contextualized version::

    {page_title} — {heading_path}

    {content}

A chunk that says "Provide the ID of the original charge" is ambiguous on
its own; prefixed with "Refunds — Partial refunds" it embeds near refund
questions. The raw text stays untouched in the table — the prefix exists
only in vector space.

Run::

    uv run --env-file .env python -m ingestion.embed
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from services.db import connect
from services.embedder import Embedder, ensure_model

logger = logging.getLogger(__name__)

BATCH_SIZE = 64


def embed_text(page_title: str | None, heading_path: str | None, content: str) -> str:
    header = " — ".join(part for part in (page_title, heading_path) if part)
    return f"{header}\n\n{content}" if header else content


def run() -> None:
    embedder = Embedder(ensure_model())
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, page_title, heading_path, content
            FROM rag.chunks
            WHERE embedding IS NULL
            ORDER BY id
            """
        ).fetchall()
        logger.info("Embedding %d chunks", len(rows))

        for start in tqdm(
            range(0, len(rows), BATCH_SIZE), desc="Embedding chunks", unit="batch"
        ):
            batch = rows[start : start + BATCH_SIZE]
            texts = [embed_text(title, path, content) for _, title, path, content in batch]
            vectors = embedder.encode_batch(texts)
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE rag.chunks SET embedding = %s WHERE id = %s",
                    [(vec, row[0]) for vec, row in zip(vectors, batch)],
                )
            conn.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
