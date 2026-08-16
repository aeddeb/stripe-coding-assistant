"""Postgres connection helper — the single owner of connection-string
building, used by ingestion, retrieval, evals, and the app alike.

``DATABASE_URL`` (cloud Postgres) wins when set; otherwise a URL is
assembled from the discrete ``POSTGRES_*`` variables matching the
docker-compose defaults — so switching between local and cloud is one
environment variable.
"""

from __future__ import annotations

import os
from urllib.parse import quote

import psycopg
from pgvector.psycopg import register_vector


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "app")
    password = os.getenv("POSTGRES_PASSWORD", "change-me")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "stripe_assistant")
    # URL-encode credentials: characters like @ or / in a password would
    # otherwise be parsed as part of the URL structure.
    return f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{db}"


def connect() -> psycopg.Connection:
    """Open a connection with pgvector support registered, so ``vector``
    columns read and write as numpy arrays."""
    conn = psycopg.connect(database_url())
    register_vector(conn)
    return conn
