"""Raw snapshot storage: exact, timestamped copies of everything fetched.

Every pipeline run writes its fetches to ``data/raw/<run-timestamp>/`` before
anything is parsed or loaded. The database holds structured, versioned rows;
these files hold the untouched originals, so any historical version can be
inspected or diffed byte-for-byte later.
"""

import re
from pathlib import Path
from urllib.parse import urlparse


def snapshot_path(snapshot_dir: Path, url: str) -> Path:
    """Map a page URL to a file path inside the snapshot directory.

    Mirrors the URL path (``.../payments/checkout.md`` becomes
    ``pages/payments/checkout.md``). Query strings are folded into the
    filename so URL variants don't collide::

        accept-a-payment.md?platform=web -> accept-a-payment__platform=web.md
    """
    parsed = urlparse(url)
    relative = parsed.path.lstrip("/")
    if parsed.query:
        safe_query = re.sub(r"[^A-Za-z0-9=_-]", "_", parsed.query)
        stem = relative[: -len(".md")] if relative.endswith(".md") else relative
        relative = f"{stem}__{safe_query}.md"
    return snapshot_dir / "pages" / relative


def write_snapshot(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
