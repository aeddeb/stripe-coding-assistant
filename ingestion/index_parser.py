"""Parse llms.txt into three buckets: page entries, sections, and the
LLM-guidance text.

llms.txt is one text file containing three kinds of lines. ``parse_index``
reads it once, top to bottom, and sorts every line into one of:

- entries   — ``- [Title](url): description`` lines: the table of contents
- sections  — ``## Heading`` lines plus the prose under them: a glossary of
              what each documentation area covers
- guidance  — the one prose-only section addressed at LLM agents
"""

import logging
import re
from urllib.parse import urlparse

from ingestion.config import DOCS_HOST, LLM_GUIDANCE_SECTION_PREFIX

# Matches index entries like:
#   - [Title](https://docs.stripe.com/page.md): optional description
LINK_LINE = re.compile(
    r"^- \[(?P<title>.+?)\]\((?P<url>[^)\s]+)\)(?::\s*(?P<description>.*))?$"
)

logger = logging.getLogger(__name__)


def parse_index(text: str) -> tuple[list[dict], list[dict], str]:
    """Parse llms.txt into page entries, sections, and the LLM-guidance text.

    Returns ``(entries, sections, llm_guidance_text)``. Each entry has
    ``section``, ``title``, ``url``, ``description`` and ``position`` (order
    in the file). Each section has ``name``, ``description`` (the prose lines
    under its heading — a short definition of what that documentation area
    covers) and ``position``.

    Only markdown pages hosted on docs.stripe.com are kept: external links
    (e.g. support.stripe.com) are skipped, and duplicate URLs listed under
    more than one section are kept once, under the first section they appear
    in.
    """
    entries: list[dict] = []
    sections: list[dict] = []
    prose: dict[str, list[str]] = {}
    guidance_lines: list[str] = []
    seen_urls: set[str] = set()
    section = ""
    in_guidance = False

    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            in_guidance = section.startswith(LLM_GUIDANCE_SECTION_PREFIX)
            if not in_guidance:
                sections.append(
                    {"name": section, "description": None, "position": len(sections)}
                )
                prose[section] = []
            continue

        if in_guidance:
            guidance_lines.append(line)
            continue

        match = LINK_LINE.match(line)
        if not match:
            # Non-link, non-empty lines under a heading are the section's
            # own description — keep them. (Lines before the first heading
            # are the file preamble and are not section prose.)
            if section and line.strip():
                prose[section].append(line.strip())
            continue

        url = match["url"]
        parsed = urlparse(url)
        if parsed.netloc != DOCS_HOST or not parsed.path.endswith(".md"):
            logger.debug("Skipping non-docs link: %s", url)
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)

        entries.append(
            {
                "section": section,
                "title": match["title"],
                "url": url,
                "description": (match["description"] or "").strip() or None,
                "position": len(entries),
            }
        )

    for section_row in sections:
        section_prose = "\n".join(prose[section_row["name"]]).strip()
        section_row["description"] = section_prose or None

    return entries, sections, "\n".join(guidance_lines).strip()
