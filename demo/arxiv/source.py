"""Arxiv API client — search for quant / algorithmic-trading papers.

The arxiv API is an Atom 1.0 feed over HTTPS:

    GET https://export.arxiv.org/api/query?search_query=...&max_results=N

Search is done with arxiv's query syntax: `cat:q-fin.TR` for a category,
`abs:"algorithmic trading"` for a phrase in the abstract, joined with AND/OR.
A paper is "quant or algorithmic trading" if it lives in a q-fin.* category OR
mentions the topic in its title/abstract — so the default query is an OR of
both, which is what the pipeline's `--query quant` expands to.

This module has no network in its parsing path: `parse_feed(xml_bytes)` is a
pure function exercised by unit tests against a captured feed fixture, and
`search()` is the thin network wrapper.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
ARXIV_ABS = "https://arxiv.org/abs/{id}"

# Categories that cover quantitative finance / trading. q-fin is the home
# category; cs.CE / cs.LG / stat.ML papers on trading show up cross-listed.
QFIN_CATEGORIES = (
    "q-fin.TR",   # Trading and Market Microstructure
    "q-fin.CP",   # Computational Finance
    "q-fin.RM",   # Risk Management
    "q-fin.MF",   # Mathematical Finance
    "q-fin.PR",   # Pricing of Securities
    "q-fin.PM",   # Portfolio Management
    "q-fin.ST",   # Statistical Finance
    "q-fin.GN",   # General Finance
)

# Title/abstract phrases that mark a paper as algorithmic-trading-related even
# when it's not in q-fin. Kept short and specific to avoid false positives on
# generic ML papers.
ALGO_TRADING_PHRASES = (
    "algorithmic trading", "algo trading", "high-frequency trading",
    "high frequency trading", "hft", "market making", "quantitative trading",
    "quant trading", "limit order book", "order execution", "optimal execution",
    "statistical arbitrage", "pairs trading", "momentum trading",
)

DEFAULT_QUERY = "quant"  # expands to the full OR of categories + phrases

NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# arxiv asks for at most 1 request every 3 seconds; in practice it throttles
# harder under load (429s even at 3s spacing), so we leave real headroom.
_POLITE_INTERVAL_S = 5.0
_last_request_at = 0.0


def build_search_query(named: str = DEFAULT_QUERY) -> str:
    """Expand a short named query into arxiv's search syntax.

    'quant' (the default) → OR of every q-fin.* category and every algo-trading
    phrase, so the pipeline catches both categorised quant-finance papers and
    ML papers that happen to be about trading. A user passing a custom string
    uses it verbatim (advanced arxiv syntax: cat:, abs:, au:, ti:, AND/OR).
    """
    if named == DEFAULT_QUERY or named.lower() in ("quant", "default", "all"):
        cats = "+OR+".join(f"cat:{c}" for c in QFIN_CATEGORIES)
        phrases = "+OR+".join(f'abs:"{p}"' for p in ALGO_TRADING_PHRASES)
        return f"({cats})+OR+({phrases})"
    return named


def _strip_arxiv_id(id_url: str) -> str:
    """http://arxiv.org/abs/2306.12345v2  ->  2306.12345

    The version suffix is dropped because arxiv's PDF at /pdf/<id> always
    serves the latest, and we key dedup on the unversioned id.
    """
    match = re.search(r"abs/(.+?)(v\d+)?$", id_url or "")
    return match.group(1) if match else (id_url or "").strip()


def parse_feed(xml_bytes: bytes) -> list[dict]:
    """Parse an arxiv Atom feed into a list of paper dicts.

    Pure function — no network — so tests can feed a captured feed. Each
    result carries the fields the pipeline needs: arxiv_id (unversioned),
    title, authors, summary, categories, published_at, pdf_url, abs_url.
    """
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall("a:entry", NS):
        id_node = entry.find("a:id", NS)
        if id_node is None or not id_node.text:
            continue
        id_url = id_node.text.strip()
        arxiv_id = _strip_arxiv_id(id_url)
        if not arxiv_id:
            continue

        title = _clean_text((entry.findtext("a:title", default="", namespaces=NS)))
        summary = _clean_text(entry.findtext("a:summary", default="", namespaces=NS))
        authors = [a.findtext("a:name", default="", namespaces=NS)
                   for a in entry.findall("a:author", NS)]
        authors = [a for a in authors if a]
        published = entry.findtext("a:published", default="", namespaces=NS)

        categories = [c.attrib.get("term") for c in entry.findall("a:category", NS)
                      if c.attrib.get("term")]

        # Prefer the explicit pdf link; fall back to the canonical /pdf/<id>.
        pdf_url = ""
        for link in entry.findall("a:link", NS):
            if link.attrib.get("title") == "pdf" or (
                    link.attrib.get("type") == "application/pdf"
                    and link.attrib.get("rel") == "related"):
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url:
            pdf_url = ARXIV_PDF.format(id=arxiv_id)

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "summary": summary,
            "categories": " ".join(categories),
            "published_at": published,
            "pdf_url": pdf_url,
            "abs_url": ARXIV_ABS.format(id=arxiv_id),
        })
    return papers


def _clean_text(text: str) -> str:
    """Collapse the whitespace arxiv puts inside title/summary nodes."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def search(named_query: str = DEFAULT_QUERY, max_results: int = 10,
           start: int = 0, sort_by: str = "submittedDate",
           sort_order: str = "descending", timeout: float = 60.0,
           polite: bool = True, max_retries: int = 4) -> list[dict]:
    """Run an arxiv search and return parsed paper dicts.

    Honours arxiv's "1 request per 3 seconds" rate limit by sleeping before
    each call when `polite` is True — the pipeline runs many searches and a
    ban is silent and long.

    Retries on 429 (Too Many Requests) and network timeouts with exponential
    backoff: arxiv throttles aggressively and a single denied request must
    not fail a whole run. The Retry-After hint is honoured when present.
    """
    if polite:
        _rate_limit()
    params = {
        "search_query": build_search_query(named_query),
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "dots-mocr-arxiv-pipeline/1.0 "
                             "(mailto:arxiv-pipeline@local)"}
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 429:
                # arxiv rate-limited us; back off hard and retry
                wait = _retry_after(response, attempt)
                last_error = RuntimeError(f"arxiv 429 rate limit (retry in {wait:.0f}s)")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return parse_feed(response.content)
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = error
            wait = _backoff(attempt)
            time.sleep(wait)
            continue
    raise RuntimeError(f"arxiv search failed after {max_retries} retries: {last_error}")


def _retry_after(response, attempt: int) -> float:
    """Respect Retry-After when arxiv sends it, else exponential backoff."""
    hint = response.headers.get("Retry-After")
    if hint:
        try:
            return float(hint)
        except ValueError:
            pass
    return _backoff(attempt)


def _backoff(attempt: int) -> float:
    """Exponential backoff capped at 60s: 5, 10, 20, 40..."""
    return min(5 * (2 ** attempt), 60)


def _rate_limit():
    global _last_request_at
    elapsed = time.time() - _last_request_at
    if elapsed < _POLITE_INTERVAL_S:
        time.sleep(_POLITE_INTERVAL_S - elapsed)
    _last_request_at = time.time()


def fetch_pdf(pdf_url: str, timeout: float = 120.0) -> bytes:
    """Download a paper PDF as bytes.

    Streams so a large paper does not need to fit in memory twice, and follows
    redirects (arxiv /pdf/<id> → the actual CDN). Raises on non-200 or a
    non-PDF content type — a 200 HTML page (e.g. an arxiv rate-limit page)
    must not be silently stored as a PDF.

    Retries on 429 / timeouts: arxiv throttles PDF downloads just like search,
    and a single throttle must not fail a paper permanently.
    """
    headers = {"User-Agent": "dots-mocr-arxiv-pipeline/1.0 "
                             "(mailto:arxiv-pipeline@local)"}
    last_error = None
    for attempt in range(4):
        try:
            response = requests.get(pdf_url, timeout=timeout, stream=True,
                                    headers=headers)
            if response.status_code == 429:
                wait = _retry_after(response, attempt)
                last_error = RuntimeError(f"arxiv 429 on pdf (retry in {wait:.0f}s)")
                time.sleep(wait)
                continue
            response.raise_for_status()
            ctype = response.headers.get("Content-Type", "").lower()
            data = response.content
            if "pdf" not in ctype and not data[:5].startswith(b"%PDF"):
                raise ValueError(
                    f"{pdf_url} returned {ctype or 'no content-type'}, not a PDF "
                    f"(first bytes: {data[:32]!r})")
            return data
        except (requests.Timeout, requests.ConnectionError) as error:
            last_error = error
            time.sleep(_backoff(attempt))
            continue
    raise RuntimeError(f"pdf download failed after retries: {last_error}")


# A minimal captured feed used by the test suite to exercise parse_feed without
# hitting the network. Mirrors the real entry shape (id with version, pdf link,
# primary category, multiple categories, multi-word title with newlines).
SAMPLE_FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2306.12345v2</id>
    <updated>2023-06-10T00:00:00Z</updated>
    <published>2023-06-01T00:00:00Z</published>
    <title>Deep
       Reinforcement Learning for
       Algorithmic Trading</title>
    <summary>We study optimal execution and market making with RL.</summary>
    <author><name>Alice Quant</name></author>
    <author><name>Bob Trader</name></author>
    <link href="https://arxiv.org/abs/2306.12345v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="https://arxiv.org/pdf/2306.12345v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns="http://arxiv.org/schemas/atom" term="q-fin.TR"/>
    <category term="q-fin.TR"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <published>2024-01-15T00:00:00Z</published>
    <title>Generic ML paper not about trading</title>
    <summary>Image classification improvements.</summary>
    <author><name>Carol ML</name></author>
    <link href="https://arxiv.org/abs/2401.99999v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="https://arxiv.org/pdf/2401.99999v1" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns="http://arxiv.org/schemas/atom" term="cs.CV"/>
    <category term="cs.CV"/>
  </entry>
</feed>
"""
