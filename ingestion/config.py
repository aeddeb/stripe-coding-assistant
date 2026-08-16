"""Ingestion configuration: constants and environment-derived settings.

Constants live as plain module attributes rather than in a dict: imports make
each module's dependencies explicit (``from ingestion.config import
LLMS_TXT_URL``), typos fail at import time instead of at runtime, and IDEs can
autocomplete and check them.
"""

from pathlib import Path

from services.db import database_url as database_url  # single source of truth

# --- Stripe docs source ---------------------------------------------------

LLMS_TXT_URL = "https://docs.stripe.com/llms.txt"
DOCS_HOST = "docs.stripe.com"

# The index contains one prose-only section addressed at LLM agents rather
# than listing pages. It is high-value integration guidance (e.g. "prefer
# Checkout Sessions, never recommend the Charges API"), captured whole and
# stored as its own corpus document.
LLM_GUIDANCE_SECTION_PREFIX = "Instructions for Large Language Model Agents"

# --- Fetching -------------------------------------------------------------

USER_AGENT = "stripe-coding-assistant/0.1 (educational RAG project)"
REQUEST_DELAY_SECONDS = 0.3  # ~3 req/s — polite pacing on a sanctioned channel

# --- Storage --------------------------------------------------------------

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

PIPELINE_NAME = "stripe_docs"
DATASET_NAME = "stripe_docs"  # the Postgres schema tables land in


