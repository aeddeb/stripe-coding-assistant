"""dlt resources and pipeline for ingesting Stripe documentation into Postgres.

Stripe publishes ``llms.txt`` (https://docs.stripe.com/llms.txt) — a curated
index of its documentation, where every listed page is also served as clean
markdown. This pipeline:

1. Fetches the index and parses it into sections and page entries.
2. Fetches every listed page as markdown (politely rate-limited).
3. Writes a raw timestamped snapshot of the index and every page under
   ``data/raw/<run-timestamp>/`` so exact historical copies survive on disk.
4. Loads pages into Postgres with dlt using an SCD2 merge: when a page's
   content changes between runs, the previous row is closed out
   (``_dlt_valid_to`` gets a timestamp) and a new row is inserted, so the
   full revision history of every page stays queryable. Pages that disappear
   from the index are retired the same way.

``docs.stripe.com`` returns no ETag or Last-Modified headers (checked
2026-08), so change detection is content-based: every run fetches every page,
and the SCD2 merge writes new versions only for pages whose content actually
changed.

Chunking and embedding are separate downstream steps; this module lands whole
pages only.

Run (Postgres from docker-compose must be up)::

    uv run --env-file .env python -m ingestion.pipeline
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import dlt
from requests import HTTPError
from tqdm import tqdm

from ingestion.config import (
    DATASET_NAME,
    LLM_GUIDANCE_SECTION_PREFIX,
    LLMS_TXT_URL,
    PIPELINE_NAME,
    RAW_DATA_DIR,
    REQUEST_DELAY_SECONDS,
    database_url,
)
from ingestion.fetcher import OffsiteRedirectError, fetch
from ingestion.index_parser import parse_index
from ingestion.snapshots import snapshot_path, write_snapshot

logger = logging.getLogger(__name__)


@dlt.resource(name="llms_index", write_disposition="replace")
def llms_index(entries: list[dict]) -> Iterator[dict]:
    """The current llms.txt index: which pages exist and how Stripe groups
    them into sections. Replaced wholesale each run — history of the index
    itself lives in the raw snapshots on disk."""
    yield from entries


@dlt.resource(
    name="sections",
    write_disposition="replace",
    # Explicit type hint: some sections have no prose, and dlt only
    # materializes columns it saw data for — the hint guarantees the column
    # exists even if a load happens to contain only description-less rows.
    columns={"description": {"data_type": "text", "nullable": True}},
)
def doc_sections(rows: list[dict]) -> Iterator[dict]:
    """Section descriptions from the index — a small glossary of what each
    documentation area covers. Useful retrieval metadata later: e.g.
    prepended to chunks at embedding time, or for section-level query
    routing."""
    yield from rows


@dlt.resource(
    name="doc_pages",
    write_disposition={"disposition": "merge", "strategy": "scd2"},
)
def doc_pages(
    entries: list[dict], guidance_text: str, snapshot_dir: Path
) -> Iterator[dict]:
    """Fetch every page in the index and yield one row per page.

    Rows contain only stable content columns — no fetch timestamp. dlt's
    SCD2 merge hashes each row to detect change, so a volatile column would
    make every page look modified on every run. The load timestamp lives in
    the ``_dlt_valid_from`` column dlt adds automatically.

    Error handling: a 404 means the page is gone even though the index still
    lists it — it is skipped, which lets the SCD2 merge retire its row. The
    same applies to pages that redirect off the docs host (whatever they
    return is not documentation). Any other error aborts the run (after
    retries), because loading a partial corpus would wrongly retire every
    page that wasn't reached.
    """
    if guidance_text:
        yield {
            "url": f"{LLMS_TXT_URL}#llm-agent-instructions",
            "section": "LLM Agent Guidance",
            "title": LLM_GUIDANCE_SECTION_PREFIX,
            "description": "Integration best practices from the llms.txt preamble.",
            "page_type": "llm_guidance",
            "content": guidance_text,
            "content_hash": hashlib.sha256(guidance_text.encode()).hexdigest(),
        }

    for entry in tqdm(entries, desc="Fetching pages", unit="page"):
        try:
            content = fetch(entry["url"])
        except OffsiteRedirectError as error:
            logger.warning("Skipping, will be retired: %s", error)
            continue
        except HTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                logger.warning("Page gone (404), will be retired: %s", entry["url"])
                continue
            raise

        write_snapshot(snapshot_path(snapshot_dir, entry["url"]), content)

        yield {
            "url": entry["url"],
            "section": entry["section"],
            "title": entry["title"],
            "description": entry["description"],
            "page_type": "doc",
            "content": content,
            "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        }

        time.sleep(REQUEST_DELAY_SECONDS)


def make_pipeline(
    pipeline_name: str = PIPELINE_NAME, dataset_name: str = DATASET_NAME
) -> dlt.Pipeline:
    """Assemble a dlt pipeline pointed at the configured Postgres.

    Overriding the names points the run at a separate dataset (Postgres
    schema) and separate local working state — used by tests and sample runs
    so they never touch the real corpus."""
    return dlt.pipeline(
        pipeline_name=pipeline_name,
        destination=dlt.destinations.postgres(credentials=database_url()),
        dataset_name=dataset_name,
        progress="tqdm",  # progress bars for the extract/normalize/load steps
    )


def run() -> None:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = RAW_DATA_DIR / run_stamp
    logger.info("Raw snapshot directory: %s", snapshot_dir)

    index_text = fetch(LLMS_TXT_URL)
    write_snapshot(snapshot_dir / "llms.txt", index_text)

    entries, sections, guidance_text = parse_index(index_text)
    logger.info(
        "Parsed %d unique page entries across %d sections", len(entries), len(sections)
    )

    pipeline = make_pipeline()
    load_info = pipeline.run(
        [
            llms_index(entries),
            doc_sections(sections),
            doc_pages(entries, guidance_text, snapshot_dir),
        ]
    )
    logger.info("Load complete: %s", load_info)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
