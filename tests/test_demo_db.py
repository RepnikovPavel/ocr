import pytest

from demo import db


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    db.init_db(tmp_path / "demo.db")
    yield


def test_session_created_and_reused():
    sid = db.get_or_create_session(None)
    assert db.get_or_create_session(sid) == sid
    assert db.get_or_create_session("unknown") != "unknown"


def test_job_roundtrip():
    sid = db.get_or_create_session(None)
    job_id = db.create_job(sid, "doc.pdf", "pdf", 11)
    job = db.get_job(job_id)
    assert job["filename"] == "doc.pdf"
    assert job["num_pages"] == 11
    assert [j["id"] for j in db.list_jobs(sid)] == [job_id]
    assert db.get_job("missing") is None


def test_task_lifecycle_fifo():
    sid = db.get_or_create_session(None)
    job_id = db.create_job(sid, "doc.pdf", "pdf", 3)
    first = db.create_task(sid, job_id, "prompt_ocr", [0], {})
    second = db.create_task(sid, job_id, "prompt_ocr", [1], {})

    claimed = db.claim_next_task()
    assert claimed["id"] == first
    assert claimed["status"] == "running"
    # claiming again returns the second task, not the running one
    assert db.claim_next_task()["id"] == second
    assert db.claim_next_task() is None

    db.update_task(first, status="done", result=[{"page_no": 0}])
    task = db.get_task(first)
    assert task["status"] == "done"
    assert task["result"] == [{"page_no": 0}]


def test_cancel_queued_and_running():
    sid = db.get_or_create_session(None)
    job_id = db.create_job(sid, "doc.pdf", "pdf", 3)
    queued = db.create_task(sid, job_id, "prompt_ocr", [0], {})
    assert db.request_cancel(queued) == "cancelled"
    assert db.get_task(queued)["status"] == "cancelled"

    running = db.create_task(sid, job_id, "prompt_ocr", [1], {})
    db.claim_next_task()
    assert db.request_cancel(running) == "cancelling"
    assert db.is_cancel_requested(running)
    assert db.get_task(running)["status"] == "running"

    assert db.request_cancel("missing") is None


def test_restart_marks_running_tasks_as_error(tmp_path):
    sid = db.get_or_create_session(None)
    job_id = db.create_job(sid, "doc.pdf", "pdf", 1)
    task_id = db.create_task(sid, job_id, "prompt_ocr", [0], {})
    db.claim_next_task()
    # simulate process restart
    db.init_db(db._DB_PATH)
    task = db.get_task(task_id)
    assert task["status"] == "error"
    assert "restart" in task["error"]
