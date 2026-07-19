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


def test_find_fullest_result_picks_most_complete_not_most_recent(store):
    """The default bundle for a document parsed at several page selections
    must be the one with the most pages_done, not the one that finished last.
    Regression for `ocrc parse URL > out.zip` returning a 3-page slice after
    a fuller 58-page parse had also been done: the 3-page run landed later
    and find_latest_result (ORDER BY created_at DESC) shadowed it.
    """
    sha = "d" * 64
    store.remember_document(sha, "paper.pdf", "pdf", 10, 1000)
    # First: a big parse covering 8 pages.
    store.store_result(sha256=sha, prompt_mode="prompt_layout_all_en",
                       pages=list(range(8)), task_id="t-big", job_id="j-big",
                       markdown="full parse", pages_done=8, generated_tokens=1000,
                       seconds=100, filename="paper.pdf")
    # Later: a tiny 1-page parse that lands AFTER the big one.
    store.store_result(sha256=sha, prompt_mode="prompt_layout_all_en",
                       pages=[3], task_id="t-small", job_id="j-small",
                       markdown="one page only", pages_done=1, generated_tokens=10,
                       seconds=2, filename="paper.pdf")

    latest = store.find_latest_result(sha, "prompt_layout_all_en")
    fullest = store.find_fullest_result(sha, "prompt_layout_all_en")
    assert latest["task_id"] == "t-small", "latest by created_at is the small one"
    assert fullest["task_id"] == "t-big", "fullest must beat recent-but-small"
    assert fullest["pages_done"] == 8
