"""The content-addressed store behind agent deduplication.

The cache key is the thing to get right: too loose and an agent receives someone
else's answer, too tight and every resubmission re-parses. Both failure modes
are cheap to write and expensive to notice, so they are pinned here.
"""

import pytest

from demo import docstore


@pytest.fixture()
def store(tmp_path):
    docstore.init(tmp_path / "store.db")
    return docstore


def add(store, sha="a" * 64, mode="prompt_layout_all_en", pages=(0, 1),
        markdown="hello world", filename="paper.pdf"):
    store.remember_document(sha, filename, "pdf", 10, 1234)
    store.store_result(sha256=sha, prompt_mode=mode, pages=list(pages),
                       task_id="t1", job_id="j1", markdown=markdown,
                       pages_done=len(pages), generated_tokens=42, seconds=1.5,
                       filename=filename)


def test_the_same_bytes_prompt_and_pages_hit_the_cache(store):
    add(store)
    assert store.find_result("a" * 64, "prompt_layout_all_en", [0, 1]) is not None


def test_page_order_does_not_create_a_second_entry(store):
    add(store, pages=(0, 1))
    assert store.find_result("a" * 64, "prompt_layout_all_en", [1, 0]) is not None
    assert store.find_result("a" * 64, "prompt_layout_all_en", [1, 1, 0]) is not None


def test_a_different_page_selection_is_a_different_answer(store):
    """An agent that asked for page 5 must not be handed the parse of pages 0-1.

    This is the bug that shipped: status reported 'cached' from another
    selection, the client went for the bundle, and the bundle had never been
    produced.
    """
    add(store, pages=(0, 1))
    assert store.find_result("a" * 64, "prompt_layout_all_en", [5]) is None


def test_a_different_prompt_is_a_different_answer(store):
    add(store, mode="prompt_layout_all_en")
    assert store.find_result("a" * 64, "prompt_ocr", [0, 1]) is None


def test_latest_result_is_a_full_row(store):
    """The bundle needs job_id; a partial row here raised KeyError in production."""
    add(store)
    row = store.find_latest_result("a" * 64, "prompt_layout_all_en")
    assert {"job_id", "task_id", "markdown", "pages_done"} <= set(row)


def test_resubmission_counts_without_duplicating_the_document(store):
    store.remember_document("b" * 64, "x.pdf", "pdf", 3, 10)
    assert store.remember_document("b" * 64, "x.pdf", "pdf", 3, 10) is False
    assert store.get_document("b" * 64)["times_submitted"] == 2


def test_search_finds_words_and_reports_a_snippet(store):
    add(store, markdown="Attention is all you need, said the transformer paper.")
    hits = store.search("transformer")
    assert len(hits) == 1
    assert "[transformer]" in hits[0]["snippet"]


def test_search_survives_hyphens_and_other_fts_operators(store):
    """FTS5 reads '-' as NOT, so a plain query like 'open-source' is a syntax
    error unless it is quoted. Agents search for hyphenated terms constantly."""
    add(store, markdown="An open-source GPT-4V competitor.")
    assert len(store.search("open-source")) == 1
    assert len(store.search("GPT-4V")) == 1


def test_search_for_something_absent_is_empty_not_an_error(store):
    add(store, markdown="nothing relevant here")
    assert store.search("квазипериодический") == []


def test_reindexing_a_document_does_not_duplicate_it(store):
    add(store, markdown="first version")
    add(store, markdown="second version")
    hits = store.search("version")
    assert len(hits) == 1, "FTS5 has no upsert; the old row must be deleted"
    assert store.stats()["cached_results"] == 1


def test_stats_report_reuse(store):
    add(store)
    store.remember_document("a" * 64, "paper.pdf", "pdf", 10, 1234)
    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["submissions"] == 2
    assert stats["cached_results"] == 1


def test_resubmission_refreshes_metadata_not_just_counter(store):
    """A later upload of the same bytes may carry corrected metadata
    (different page count, renamed file). The stored row must reflect the
    latest values, not the first ones — otherwise stale `num_pages` is served
    forever on status/listing responses.
    """
    sha = "c" * 64
    store.remember_document(sha, "old-name.pdf", "pdf", 3, 100)
    store.remember_document(sha, "new-name.pdf", "pdf", 8, 200)
    row = store.get_document(sha)
    assert row["filename"] == "new-name.pdf"
    assert row["num_pages"] == 8
    assert row["size_bytes"] == 200
    assert row["times_submitted"] == 2


def test_search_falls_back_to_substring_when_fts_tokenizer_misses(store):
    """FTS5's unicode61 tokenizer splits on punctuation in ways that can drop
    a term entirely; for those we fall back to a case-insensitive LIKE on
    `body` / `filename` so a user searching for a substring still gets hits.

    Reproduces the 'attention returns 0 results' symptom from issue ocr#11:
    indexing a body whose only token boundary placement ate the queried word.
    """
    add(store, markdown="CompressedSparse Attention architecture, mid-sentence.",
        filename="deepseek.pdf")
    hits = store.search("attention")
    assert len(hits) == 1
    assert "deepseek.pdf" in hits[0]["filename"]
    # LIKE fallback also matches filename substrings
    assert len(store.search("deepseek")) == 1


def test_search_absent_still_empty_with_fallback(store):
    """LIKE fallback must not invent matches: absent terms stay absent."""
    add(store, markdown="nothing relevant here")
    assert store.search("квазипериодический") == []
