"""Apply ``db/schema.sql`` to the database named by ``DATABASE_URL``.

``make schema`` reaches the local docker Postgres through the container,
which cannot reach a hosted database. This module is the counterpart for
the cloud one: it connects over the network with the project's own driver,
so it needs no ``psql`` binary installed anywhere.

``DATABASE_URL`` must be set explicitly, and there is deliberately no
fallback to the ``POSTGRES_*`` variables. The fallback is what
``services.db`` does, and it is the right behaviour for the application —
but for a script that runs DDL it would mean an unset variable silently
rewrites the local database while the operator believes they are updating
production. Failing is the safer answer.

``db/schema.sql`` is idempotent, so re-running this is safe and is the
intended way to bring a database back in line after a schema change.

Run:  DATABASE_URL='postgresql://...' make schema-cloud
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def target_description(url: str) -> str:
    """Host and database name, with any credentials left out — printed so
    the operator can see which database is about to change."""
    parts = urlparse(url)
    return f"{parts.hostname or '?'}{parts.path or ''}"


def main() -> None:
    url = os.getenv("DATABASE_URL")
    if not url:
        sys.exit(
            "DATABASE_URL is not set.\n"
            "This target applies schema changes to the cloud database, so the "
            "connection string has to be given explicitly:\n"
            "    DATABASE_URL='postgresql://...' make schema-cloud\n"
            "For the local docker database, use `make schema` instead."
        )

    print(f"applying {SCHEMA_PATH.name} to {target_description(url)}")
    # psycopg sends a parameterless statement over the simple query
    # protocol, which accepts the whole multi-statement file in one call.
    with psycopg.connect(url) as conn:
        conn.execute(SCHEMA_PATH.read_text())
        conn.commit()
        columns = [
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'app' AND table_name = 'messages' "
                "ORDER BY ordinal_position"
            ).fetchall()
        ]
    print(f"schema applied. app.messages has {len(columns)} columns:")
    print("  " + ", ".join(columns))


if __name__ == "__main__":
    main()
