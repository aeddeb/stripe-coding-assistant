"""HTTP fetching with retries and a polite User-Agent."""

from urllib.parse import urlparse

from dlt.sources.helpers import requests
from requests import HTTPError

from ingestion.config import USER_AGENT


class OffsiteRedirectError(Exception):
    """A URL redirected to a different host than the one requested.

    Some index entries redirect away from the docs site entirely (e.g.
    ``/samples.md`` 301s to a GitHub org). Whatever such a URL returns is
    not documentation content, so callers treat it like a missing page.
    """


def _offsite(requested_url: str, final_url: str) -> bool:
    return urlparse(final_url).netloc != urlparse(requested_url).netloc


def fetch(url: str) -> str:
    """Fetch a URL as text. Transient failures are retried by the dlt
    requests helper with backoff; persistent failures raise. Raises
    :class:`OffsiteRedirectError` if the response came from a different
    host than requested — including when that host answered with an error
    status (the dlt session raises on status internally, so the redirect
    has to be detected on the error path too)."""
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    except HTTPError as error:
        final = error.response
        if final is not None and _offsite(url, final.url):
            raise OffsiteRedirectError(
                f"{url} redirected off-site to {final.url}"
            ) from error
        raise
    if _offsite(url, response.url):
        raise OffsiteRedirectError(f"{url} redirected off-site to {response.url}")
    response.raise_for_status()
    return response.text
