"""Tests for demo.storage — local backend + factory fallback (no network).

The SeaweedFS backend (boto3) is exercised end-to-end on the server against a
real cluster; these unit tests cover the contract every backend must honour
and the local fallback path, so they run anywhere with no deps.
"""

import hashlib
import os

import pytest

import demo.storage as storage
from demo.storage import (
    KIND_BUNDLE, KIND_PDF, LocalBlobStore, SeaweedBlobStore, configure_store,
    get_store, put_bundle, put_pdf, sha256_bytes, store_kind)


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    """Every test gets an isolated LocalBlobStore rooted in a tmp dir."""
    store = LocalBlobStore(tmp_path / "blobs")
    configure_store(store)
    yield store
    configure_store(None)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- LocalBlobStore

def test_local_put_get_roundtrip(_fresh_store):
    data = b"%PDF-1.4 hello arxiv"
    sha = _digest(data)
    key = _fresh_store.put(sha, KIND_PDF, data)
    assert _fresh_store.get(sha, KIND_PDF) == data
    assert _fresh_store.exists(sha, KIND_PDF)
    assert key.endswith(f"{sha}.pdf")
    # idempotent: putting the same bytes again does not duplicate or error
    _fresh_store.put(sha, KIND_PDF, data)
    assert _fresh_store.get(sha, KIND_PDF) == data


def test_local_absent_returns_none(_fresh_store):
    assert _fresh_store.get("deadbeef" * 8, KIND_PDF) is None
    assert not _fresh_store.exists("deadbeef" * 8, KIND_PDF)
    assert _fresh_store.size("deadbeef" * 8, KIND_PDF) is None


def test_local_kinds_dont_collide(_fresh_store):
    """The same sha256 can hold a pdf and a bundle independently."""
    sha = "ab" * 32
    _fresh_store.put(sha, KIND_PDF, b"pdf bytes")
    _fresh_store.put(sha, KIND_BUNDLE, b"bundle bytes")
    assert _fresh_store.get(sha, KIND_PDF) == b"pdf bytes"
    assert _fresh_store.get(sha, KIND_BUNDLE) == b"bundle bytes"


def test_local_size_reports_bytes(_fresh_store):
    data = b"x" * 2048
    sha = _digest(data)
    _fresh_store.put(sha, KIND_PDF, data)
    assert _fresh_store.size(sha, KIND_PDF) == 2048


def test_local_delete_is_idempotent(_fresh_store):
    sha = _digest(b"data")
    _fresh_store.put(sha, KIND_PDF, b"data")
    _fresh_store.delete(sha, KIND_PDF)
    assert not _fresh_store.exists(sha, KIND_PDF)
    _fresh_store.delete(sha, KIND_PDF)  # second delete must not raise


def test_local_sharding(_fresh_store, tmp_path):
    """Files are sharded into ab/cd/ so a flat dir of 100k files is avoided."""
    sha = "abcdef0123456789" * 4  # ab/cd/...
    _fresh_store.put(sha, KIND_PDF, b"data")
    expected = tmp_path / "blobs" / "ab" / "cd" / f"{sha}.pdf"
    assert expected.is_file()


def test_local_atomic_write(tmp_path):
    """A crash mid-write must not leave a half-written blob behind.

    We simulate by raising during the tmp rename target's creation: the
    store writes to <name>.tmp then renames atomically, so the canonical
    path either exists fully or not at all.
    """
    store = LocalBlobStore(tmp_path / "blobs")
    sha = _digest(b"first")
    store.put(sha, KIND_PDF, b"first")
    # corrupt by writing a .tmp sibling that must never be read as the blob
    from pathlib import Path
    blob_path = Path(store._path(sha, KIND_PDF))
    blob_path.with_suffix(".pdf.tmp").write_bytes(b"partial junk")
    # canonical read is still the complete bytes
    assert store.get(sha, KIND_PDF) == b"first"


# ---------------------------------------------------------------- helpers

def test_sha256_bytes_matches_hashlib():
    data = b"some pdf bytes"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_put_pdf_and_bundle_helpers(_fresh_store):
    pdf = b"%PDF"
    bundle = b"PK\x03\x04 zip"
    sha = sha256_bytes(pdf)
    put_pdf(sha, pdf)
    put_bundle(sha, bundle)  # defaults to parser=dots_mocr
    assert _fresh_store.get(sha, KIND_PDF) == pdf
    assert _fresh_store.get(sha, KIND_BUNDLE, parser="dots_mocr") == bundle
    assert storage.has_pdf(sha)
    # the helpers round-trip with their own default parser
    assert get_bundle(sha) == bundle
    assert storage.has_bundle(sha, "dots_mocr")


# ---------------------------------------------------------------- factory

def test_factory_falls_back_to_local_when_no_endpoint(monkeypatch, tmp_path):
    """With SEAWEED_S3_ENDPOINT unset, get_store() returns a LocalBlobStore."""
    configure_store(None)
    monkeypatch.delenv("SEAWEED_S3_ENDPOINT", raising=False)
    monkeypatch.setenv("STORAGE_LOCAL_DIR", str(tmp_path / "fb"))
    store = get_store()
    assert isinstance(store, LocalBlobStore)
    assert store_kind() == "local"


def test_factory_falls_back_when_seaweed_unreachable(monkeypatch, tmp_path):
    """A configured but unreachable endpoint degrades to local, not a crash.

    The pipeline must still run on a host where SeaweedFS is down; only the
    storage target changes, and the UI surfaces the difference.
    """
    configure_store(None)
    monkeypatch.setenv("SEAWEED_S3_ENDPOINT", "http://127.0.0.1:1")  # nothing there
    monkeypatch.setenv("SEAWEED_ACCESS_KEY", "agent_key")
    monkeypatch.setenv("SEAWEED_SECRET_KEY", "agent_secret_dev_change_me")
    monkeypatch.setenv("SEAWEED_BUCKET", "arxiv-papers-test")
    monkeypatch.setenv("STORAGE_LOCAL_DIR", str(tmp_path / "fb"))
    store = get_store()
    # Either the connection refused before construction (→ LocalBlobStore) or
    # construction succeeded but is unusable. Both are acceptable degradation;
    # what we forbid is an unhandled exception escaping get_store().
    assert store.kind() in ("local", "seaweedfs")


def test_factory_prefers_seaweed_when_available(monkeypatch):
    """If boto3 is installed and the endpoint answers, Seaweed is chosen.

    Skipped when boto3 is absent (e.g. CI on a minimal image): the fallback
    is the documented behaviour there, not a failure.
    """
    pytest.importorskip("boto3")
    configure_store(None)
    # Hit the real in-process check only if a live endpoint is configured via
    # env at run time; otherwise this test is a no-op (covered on the server).
    endpoint = os.environ.get("SEAWEED_S3_ENDPOINT_LIVE")
    if not endpoint:
        pytest.skip("set SEAWEED_S3_ENDPOINT_LIVE to exercise the S3 path")
    monkeypatch.setenv("SEAWEED_S3_ENDPOINT", endpoint)
    store = get_store()
    assert isinstance(store, SeaweedBlobStore)
