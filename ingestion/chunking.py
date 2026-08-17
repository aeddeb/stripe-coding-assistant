"""Section-aware chunking: whole doc pages → retrievable chunks.

Strategy
--------
Stripe serves its docs as markdown, so heading structure is reliable:

1. Split each page at headings (``#`` … ``####``), tracking the heading
   path (e.g. ``Refunds > Partial refunds``) as retrieval metadata.
2. Sections longer than ``MAX_CHARS`` are split further at paragraph
   boundaries, with the previous paragraph carried over as overlap so a
   sentence's context is never cut mid-thought. A single paragraph that
   itself exceeds ``MAX_CHARS`` (a long table or code block with no blank
   lines) is split at line boundaries, so every chunk stays bounded.
3. Consecutive tiny fragments are merged forward until they reach
   ``MIN_CHARS``, so no chunk is a lone heading or one-liner.

``MAX_CHARS`` is set by the embedding model, not by taste: all-MiniLM-L6-v2
reads ~256 tokens (~1,200 characters) and truncates the rest. A chunk larger
than the model's window is partially invisible to vector search.

Run (rebuilds ``rag.chunks`` from the pages the dlt pipeline landed)::

    uv run --env-file .env python -m ingestion.chunking
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from tqdm import tqdm

from ingestion.config import DATASET_NAME

logger = logging.getLogger(__name__)

MAX_CHARS = 1200  # ≈ the embedder's 256-token window
MIN_CHARS = 200   # merge fragments smaller than this
OVERLAP_CHARS = 200  # carried between splits of an oversized section

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    heading_path: str
    content: str


def split_page(markdown: str) -> list[Chunk]:
    """Split one markdown page into size-bounded, heading-tagged chunks."""
    sections = _split_at_headings(markdown)
    chunks: list[Chunk] = []
    for heading_path, text in sections:
        for piece in _split_long(text):
            chunks.append(Chunk(heading_path=heading_path, content=piece))
    return _merge_small(chunks)


def _split_at_headings(markdown: str) -> list[tuple[str, str]]:
    """Break a page at markdown headings, keeping the heading trail.

    Returns ``(heading_path, section_text)`` pairs in document order.
    """
    stack: list[tuple[int, str]] = []  # (level, title) of open headings
    sections: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        buffer.clear()
        if text:
            path = " > ".join(title for _, title in stack)
            sections.append((path, text))

    in_code_block = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
        match = None if in_code_block else HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return sections


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split an oversized section at paragraph boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    # A paragraph larger than the cap on its own (a table or code block
    # with no blank lines) is split at line boundaries first, so the
    # packing loop below only ever sees pieces that fit.
    paragraphs: list[str] = []
    for para in re.split(r"\n\n+", text):
        if len(para) > max_chars:
            paragraphs.extend(_split_at_lines(para, max_chars))
        else:
            paragraphs.append(para)

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        if size + len(para) > max_chars and current:
            pieces.append("\n\n".join(current))
            # Overlap: carry the tail of the previous piece forward.
            carried = current[-1][-OVERLAP_CHARS:] if current else ""
            current = [carried, para] if carried else [para]
            size = len(carried) + len(para)
        else:
            current.append(para)
            size += len(para)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _split_at_lines(paragraph: str, max_chars: int) -> list[str]:
    """Split a blank-line-free paragraph into pieces of at most ``max_chars``.

    Splits at line boundaries (table rows, lines of code). A single line
    longer than the cap — rare — is sliced at the cap as a last resort.
    """
    pieces: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal size
        if current:
            pieces.append("\n".join(current))
            current.clear()
            size = 0

    for line in paragraph.splitlines():
        if len(line) > max_chars:
            flush()
            for start in range(0, len(line), max_chars):
                pieces.append(line[start : start + max_chars])
            continue
        if size + len(line) > max_chars:
            flush()
        current.append(line)
        size += len(line) + 1  # +1 for the newline joiner
    flush()
    return pieces


def _merge_small(chunks: list[Chunk], min_chars: int = MIN_CHARS) -> list[Chunk]:
    """Fold fragments smaller than ``min_chars`` into the following chunk."""
    merged: list[Chunk] = []
    pending = ""
    for chunk in chunks:
        content = (pending + "\n\n" + chunk.content).strip() if pending else chunk.content
        if len(content) < min_chars:
            pending = content
            continue
        merged.append(Chunk(heading_path=chunk.heading_path, content=content))
        pending = ""
    if pending:
        if merged:
            merged[-1] = Chunk(
                heading_path=merged[-1].heading_path,
                content=merged[-1].content + "\n\n" + pending,
            )
        else:
            merged.append(Chunk(heading_path="", content=pending))
    return merged


def rebuild() -> None:
    """Rebuild ``rag.chunks`` from the current page versions in Postgres.

    Reads the SCD2 table the dlt pipeline maintains (current rows only),
    chunks every page, and replaces the chunks table wholesale. Embeddings
    are cleared by the rebuild; run ``ingestion/embed.py`` afterwards.
    """
    from services.db import connect

    source = os.getenv("SOURCE_DATASET", DATASET_NAME)
    with connect() as conn:
        pages = conn.execute(
            f"""
            SELECT url, title, section, content
            FROM {source}.doc_pages
            WHERE _dlt_valid_to IS NULL
            """
        ).fetchall()
        logger.info("Chunking %d pages from %s.doc_pages", len(pages), source)

        rows = []
        for url, title, section, content in tqdm(pages, desc="Chunking pages", unit="page"):
            for i, chunk in enumerate(split_page(content)):
                rows.append((url, title, section, chunk.heading_path, i, chunk.content))

        insert_sql = """
            INSERT INTO rag.chunks
                (page_url, page_title, section, heading_path, chunk_index, content)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        batch_size = 500
        with conn.cursor() as cur:
            cur.execute("TRUNCATE rag.chunks RESTART IDENTITY")
            for start in tqdm(
                range(0, len(rows), batch_size), desc="Inserting chunks", unit="batch"
            ):
                cur.executemany(insert_sql, rows[start : start + batch_size])
        conn.commit()
        logger.info("Wrote %d chunks (avg %.0f chars)",
                    len(rows),
                    sum(len(r[5]) for r in rows) / max(len(rows), 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rebuild()
