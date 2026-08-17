"""Create (or update) ``grafana_ro`` — the read-only Postgres role Grafana
connects as.

Grafana only ever reads: its dashboards are SELECT queries over the
monitoring tables. Giving it a role that *cannot* write means a
misconfigured (or compromised) dashboard cannot touch the data.

Idempotent — safe to re-run; runs as part of ``make schema``.

The password comes from ``GRAFANA_DB_PASSWORD`` (defaults to ``grafana``
for local docker-compose use; override it in ``.env``).
"""

from __future__ import annotations

import os

from services import db

ROLE = "grafana_ro"


def main() -> None:
    password = os.getenv("GRAFANA_DB_PASSWORD", "grafana")
    with db.connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,)
        ).fetchone()
        # Role names can't be parameterized; ROLE is a constant, and the
        # password is quoted server-side via quote_literal-style escaping.
        escaped = password.replace("'", "''")
        if exists:
            conn.execute(f"ALTER ROLE {ROLE} WITH LOGIN PASSWORD '{escaped}'")
        else:
            conn.execute(f"CREATE ROLE {ROLE} WITH LOGIN PASSWORD '{escaped}'")

        # Read-only access to the monitoring tables (app) and the eval
        # results (rag) — SELECT and nothing else. Default privileges cover
        # tables created later by the app user.
        for schema in ("app", "rag"):
            conn.execute(f"GRANT USAGE ON SCHEMA {schema} TO {ROLE}")
            conn.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {ROLE}")
            conn.execute(
                f"ALTER DEFAULT PRIVILEGES IN SCHEMA {schema} "
                f"GRANT SELECT ON TABLES TO {ROLE}"
            )
    print(f"role {ROLE!r} ready: LOGIN + SELECT on app, rag")


if __name__ == "__main__":
    main()
