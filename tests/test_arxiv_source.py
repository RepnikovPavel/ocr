"""Tests for demo.arxiv.source — feed parser (no network) + query builder.

The parser is a pure function over a captured Atom feed; the live search()
and fetch_pdf() are exercised on the server, marked `live`.
"""

import pytest

from demo.arxiv import source
from demo.arxiv.source import SAMPLE_FEED, build_search_query, parse_feed


def test_parse_feed_basic_shape():
    papers = parse_feed(SAMPLE_FEED)
    assert len(papers) == 2
    p = papers[0]
    assert p["arxiv_id"] == "2306.12345"  # version stripped
    assert p["title"].startswith("Deep Reinforcement Learning")  # newlines folded
    assert "  " not in p["title"]  # whitespace collapsed
    assert p["authors"] == ["Alice Quant", "Bob Trader"]
    assert "q-fin.TR" in p["categories"]
    assert "cs.LG" in p["categories"]
    assert p["pdf_url"] == "https://arxiv.org/pdf/2306.12345v2"
    assert p["abs_url"] == "https://arxiv.org/abs/2306.12345"
    assert p["published_at"] == "2023-06-01T00:00:00Z"


def test_parse_feed_strips_version_from_id():
    p = parse_feed(SAMPLE_FEED)[0]
    assert "v2" not in p["arxiv_id"]
    # the second entry keeps v1 stripped too
    assert parse_feed(SAMPLE_FEED)[1]["arxiv_id"] == "2401.99999"


def test_parse_feed_title_whitespace_collapsed():
    title = parse_feed(SAMPLE_FEED)[0]["title"]
    assert "\n" not in title
    assert "Deep Reinforcement" in title


def test_parse_feed_pdf_url_fallback_when_missing():
    """A feed with no explicit pdf link falls back to /pdf/<id>."""
    feed = SAMPLE_FEED.replace(
        b'<link title="pdf" href="https://arxiv.org/pdf/2306.12345v2" '
        b'rel="related" type="application/pdf"/>', b"")
    p = parse_feed(feed)[0]
    assert p["pdf_url"] == "https://arxiv.org/pdf/2306.12345"


def test_build_search_query_default_covers_qfin_and_phrases():
    q = build_search_query("quant")
    assert "cat:q-fin.TR" in q
    assert 'abs:"algorithmic trading"' in q
    assert "+OR+" in q
    # both branches (categories OR phrases) are present
    assert q.count("(") >= 2


def test_build_search_query_custom_passes_through():
    # a user-supplied advanced query is used verbatim
    assert build_search_query("abs:limit order book") == "abs:limit order book"


def test_parse_feed_empty_feed():
    empty = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    assert parse_feed(empty) == []


def test_parse_feed_skips_entry_without_id():
    feed = b"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>no id here</title></entry>
  <entry>
    <id>http://arxiv.org/abs/2502.00001v1</id>
    <title>has id</title>
    <link title="pdf" href="https://arxiv.org/pdf/2502.00001v1" rel="related" type="application/pdf"/>
  </entry>
</feed>"""
    papers = parse_feed(feed)
    assert len(papers) == 1
    assert papers[0]["arxiv_id"] == "2502.00001"


@pytest.mark.live
def test_live_search_returns_quant_papers():
    """Hits the real arxiv API. Skipped unless --live is passed on the server."""
    papers = source.search("quant", max_results=3)
    assert len(papers) >= 1
    assert all("arxiv_id" in p for p in papers)
    # at least one result should be q-fin or mention trading in title/summary
    assert any("q-fin" in p["categories"]
               or "trading" in (p["title"] + p["summary"]).lower()
               for p in papers)
