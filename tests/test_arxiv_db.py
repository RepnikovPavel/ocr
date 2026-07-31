"""Tests for demo.arxiv.db — papers, runs, steps lifecycle.

Pure SQLite (no network, no boto3). Verifies the DAG step machine: the run's
aggregate counters and terminal status are derived correctly from its steps,
pre-seeding renders the full grid, and refresh preserves parsed state.
"""

import pytest

from demo.arxiv import db
from demo.arxiv.db import STAGES


@pytest.fixture()
def adb(tmp_path):
    db.init(str(tmp_path / "arxiv.db"))
    yield db


def _paper(arxiv_id="2306.12345", **kw):
    return {"arxiv_id": arxiv_id, "title": "Quant Trading " + arxiv_id,
            "authors": ["Alice", "Bob"], "summary": "A summary.",
            "categories": "q-fin.TR q-fin.CP", "published_at": "2023-06-01",
            "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
            "abs_url": f"http://arxiv.org/abs/{arxiv_id}", **kw}


# ----------------------------------------------------------------- papers

def test_upsert_returns_true_only_on_first(adb):
    assert adb.upsert_paper(_paper("1")) is True
    assert adb.upsert_paper(_paper("1", title="revised")) is False
    p = adb.get_paper("1")
    assert p["title"] == "revised"  # metadata refreshed


def test_upsert_preserves_state_on_refresh(adb):
    adb.upsert_paper(_paper("2"))
    adb.set_paper_downloaded("2", "abc" * 22, 4096)
    adb.set_paper_parsed("2")
    # a later metadata refresh from arxiv must NOT wipe the sha/parse state
    adb.upsert_paper(_paper("2", title="revised title"))
    p = adb.get_paper("2")
    assert p["title"] == "revised title"
    assert p["sha256"] == "abc" * 22
    assert p["parsed"] is True
    assert p["storage_status"] == "parsed"


def test_state_transitions(adb):
    adb.upsert_paper(_paper("3"))
    adb.set_paper_downloaded("3", "deadbeef" * 8, 100)
    adb.set_paper_pages("3", 12)
    adb.set_paper_submitted("3", "task-xyz")
    adb.set_paper_stored("3")
    adb.set_paper_parsed("3")
    p = adb.get_paper("3")
    assert p["num_pages"] == 12
    assert p["ocr_task_id"] == "task-xyz"
    assert p["storage_status"] == "parsed"
    assert p["parsed"] is True


def test_status_is_monotonic(adb):
    """A late `stored`/`failed` must never demote an already-parsed paper.

    Stages run out of order across re-runs (a cached parse skips wait_ocr) and
    the executor marks a paper failed on error — without monotonic promotion a
    stray stored/failed could erase a successful parse.
    """
    adb.upsert_paper(_paper("m"))
    adb.set_paper_downloaded("m", "ff" * 32, 1)
    adb.set_paper_parsed("m")
    # these must all be no-ops on the status column
    adb.set_paper_stored("m")
    assert adb.get_paper("m")["storage_status"] == "parsed"
    adb.set_paper_downloaded("m", "ee" * 32, 2)
    assert adb.get_paper("m")["storage_status"] == "parsed"
    adb.set_paper_failed("m")
    assert adb.get_paper("m")["storage_status"] == "parsed"
    assert adb.get_paper("m")["parsed"] is True


def test_failed_only_applies_when_not_parsed(adb):
    """A paper that never reached parsed can be marked failed."""
    adb.upsert_paper(_paper("f"))
    adb.set_paper_downloaded("f", "11" * 32, 1)
    adb.set_paper_failed("f")
    assert adb.get_paper("f")["storage_status"] == "failed"


def test_count_papers(adb):
    adb.upsert_paper(_paper("a"))
    adb.upsert_paper(_paper("b"))
    adb.set_paper_downloaded("a", "x" * 64, 500)
    adb.set_paper_parsed("a")
    c = adb.count_papers()
    assert c["total"] == 2
    assert c["parsed"] == 1
    assert c["stored"] == 1
    assert c["bytes"] == 500
    # by_parser only counts rows in paper_parses (record_parse writes them);
    # set_paper_parsed alone does not, so it's empty here
    assert c["by_parser"] == {}


def test_count_papers_by_parser(adb):
    adb.upsert_paper(_paper("a"))
    adb.upsert_paper(_paper("b"))
    adb.set_paper_downloaded("a", "x" * 64, 500)
    adb.set_paper_downloaded("b", "y" * 64, 700)
    adb.record_parse("a", "dots_mocr", "x" * 64, pages_done=9)
    adb.record_parse("a", "classic_fitz", "x" * 64, pages_done=9)
    adb.record_parse("b", "classic_fitz", "y" * 64, pages_done=5)
    c = adb.count_papers()
    assert c["parsed"] == 2
    assert c["by_parser"] == {"dots_mocr": 1, "classic_fitz": 2}


def test_get_paper_by_sha(adb):
    adb.upsert_paper(_paper("c"))
    adb.set_paper_downloaded("c", "cafe" * 16, 1)
    assert adb.get_paper_by_sha("cafe" * 16)["arxiv_id"] == "c"
    assert adb.get_paper_by_sha("00" * 32) is None


# ----------------------------------------------------------------- runs + steps

def test_create_run_preseeds_all_stages(adb):
    adb.upsert_paper(_paper("p1"))
    adb.upsert_paper(_paper("p2"))
    run = adb.create_run("quant", 2, ["p1", "p2"])
    steps = adb.steps_for_run(run)
    # 2 papers * len(STAGES) step rows
    assert len(steps) == 2 * len(STAGES)
    assert all(s["status"] == "queued" for s in steps)
    # the canonical stage order
    p1_stages = [s["stage"] for s in steps if s["arxiv_id"] == "p1"]
    assert tuple(p1_stages) == STAGES


def test_step_lifecycle_marks_run_running(adb):
    adb.upsert_paper(_paper("p1"))
    run = adb.create_run("quant", 1, ["p1"])
    adb.step_start(run, "p1", "download")
    assert adb.get_run(run)["status"] == "running"


def test_run_done_when_all_steps_terminal(adb):
    adb.upsert_paper(_paper("p1"))
    run = adb.create_run("quant", 1, ["p1"])
    for stage in STAGES:
        adb.step_done(run, "p1", stage)
    r = adb.get_run(run)
    assert r["status"] == "done"
    assert r["done"] == len(STAGES)
    assert r["failed"] == 0
    assert r["finished_at"] is not None


def test_run_error_when_any_step_errored_and_none_pending(adb):
    adb.upsert_paper(_paper("p1"))
    run = adb.create_run("quant", 1, ["p1"])
    adb.step_done(run, "p1", "download")
    adb.step_error(run, "p1", "submit_ocr", "connection refused")
    # remaining stages still queued → run stays running
    assert adb.get_run(run)["status"] == "running"
    for stage in STAGES:
        if stage in ("download", "submit_ocr"):
            continue
        adb.step_skip(run, "p1", stage, "upstream failed")
    r = adb.get_run(run)
    assert r["status"] == "error"
    assert r["failed"] == 1
    assert r["skipped"] == len(STAGES) - 2


def test_step_skip_recomputes_run(adb):
    adb.upsert_paper(_paper("p1"))
    run = adb.create_run("quant", 1, ["p1"])
    adb.step_skip(run, "p1", "download", "already in storage")
    for stage in STAGES:
        if stage == "download":
            continue
        adb.step_done(run, "p1", stage)
    r = adb.get_run(run)
    assert r["status"] == "done"
    assert r["skipped"] == 1


def test_steps_by_paper_groups(adb):
    adb.upsert_paper(_paper("p1"))
    adb.upsert_paper(_paper("p2"))
    run = adb.create_run("quant", 2, ["p1", "p2"])
    by_paper = adb.steps_by_paper(run)
    assert set(by_paper) == {"p1", "p2"}
    assert len(by_paper["p1"]) == len(STAGES)


def test_finish_run_forces_terminal(adb):
    adb.upsert_paper(_paper("p1"))
    run = adb.create_run("quant", 1, ["p1"])
    adb.finish_run(run, status="error", error="runner crashed")
    r = adb.get_run(run)
    assert r["status"] == "error"
    assert r["error"] == "runner crashed"
