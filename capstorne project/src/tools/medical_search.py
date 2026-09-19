"""Disease information lookup from trusted public sources.

Two sources, both key-free:
  * MedlinePlus (US National Library of Medicine) via its XML search API
  * WHO fact sheets, fetched by topic slug

Results are returned as plain dicts so Step 4 can embed them into FAISS and
Step 5 can cite them by URL.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

import requests

from src.config import settings

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# MedlinePlus wraps every query-term match in <span class="qtN"> highlight
# markup, so returned fields must be stripped before they reach a prompt.
def _clean(text: str | None) -> str:
    if not text:
        return ""
    unescaped = html.unescape(text)
    without_tags = _TAG_RE.sub(" ", unescaped)
    return _WS_RE.sub(" ", html.unescape(without_tags)).strip()


def _truncate(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def search_medline(query: str, max_results: int | None = None) -> list[dict]:
    """Search MedlinePlus health topics.

    Returns a list of {title, url, source, summary, also_called}. On any
    network or parse failure the list is empty and the caller degrades
    gracefully rather than the agent crashing mid-plan.
    """
    if not query or not query.strip():
        return []

    limit = max_results or settings.search_max_results
    try:
        response = requests.get(
            settings.medline_search_url,
            params={"db": "healthTopics", "term": query.strip(), "retmax": limit},
            timeout=settings.search_timeout,
            headers={"User-Agent": "AgenticHealthcareAssistant/0.1"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except (requests.RequestException, ET.ParseError):
        return []

    results: list[dict] = []
    for document in root.findall(".//document")[:limit]:
        fields = {
            content.get("name"): _clean(content.text)
            for content in document.findall("content")
        }
        summary = fields.get("FullSummary") or fields.get("snippet") or ""
        results.append(
            {
                "title": fields.get("title", "Untitled"),
                "url": document.get("url", ""),
                "source": "MedlinePlus (NLM)",
                "organization": fields.get("organizationName", "National Library of Medicine"),
                "also_called": fields.get("altTitle", ""),
                "summary": _truncate(summary),
            }
        )
    return results


def _slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", (topic or "").lower()).strip()
    return re.sub(r"[\s_]+", "-", slug)


def fetch_who_factsheet(topic: str) -> dict | None:
    """Fetch a WHO fact sheet by topic slug, or None if there isn't one.

    WHO has no public search API, so this guesses the canonical fact-sheet
    URL. Many topics (chronic kidney disease, for one) have no fact sheet at
    all, which is why a missing page is a normal outcome and not an error.
    """
    slug = _slugify(topic)
    if not slug:
        return None

    url = f"{settings.who_factsheet_url.rstrip('/')}/detail/{slug}"
    try:
        response = requests.get(
            url,
            timeout=settings.search_timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AgenticHealthcareAssistant/0.1)"},
        )
        if response.status_code != 200:
            return None
    except requests.RequestException:
        return None

    body = response.text
    # Strip scripts and styles before tags, or their contents leak into the text.
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", body, flags=re.S | re.I)
    paragraphs = [_clean(match) for match in re.findall(r"<p\b[^>]*>(.*?)</p>", body, flags=re.S)]
    text = " ".join(p for p in paragraphs if len(p) > 60)
    if not text:
        return None

    return {
        "title": f"WHO fact sheet: {topic.strip().title()}",
        "url": url,
        "source": "World Health Organization",
        "organization": "WHO",
        "also_called": "",
        "summary": _truncate(text, 2000),
    }


def search_medical_information(query: str, max_results: int | None = None) -> list[dict]:
    """Combined lookup: MedlinePlus topics plus a WHO fact sheet when one exists."""
    results = search_medline(query, max_results)
    factsheet = fetch_who_factsheet(query)
    if factsheet:
        results.append(factsheet)
    return results


def format_search_results(results: list[dict]) -> str:
    """Render results as numbered, citable text for an LLM prompt."""
    if not results:
        return "No results found from MedlinePlus or WHO for this query."

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        block = [f"[{index}] {result['title']} - {result['source']}"]
        if result.get("also_called"):
            block.append(f"    Also called: {result['also_called']}")
        if result.get("url"):
            block.append(f"    URL: {result['url']}")
        if result.get("summary"):
            block.append(f"    {result['summary']}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)
