"""Blob storage for the arxiv pipeline — SeaweedFS first, local disk fallback.

The OCR service already caches the *parsed markdown* in SQLite (see docstore.py),
keyed on SHA-256 + prompt_mode + pages. This module is the *binary* counterpart:
it stores the original PDFs and the assembled result bundles as opaque blobs,
content-addressed by SHA-256, so the same arxiv paper is never re-downloaded and
a finished bundle is served from storage instead of re-parsed.

Why a separate process owns storage
-----------------------------------
The production OCR container (`dots_mocr_demo`) is bind-mounted read-only and
ships WITHOUT boto3. The arxiv pipeline runner, by contrast, is a host-level
venv that does have boto3. Both processes share the same `demo.db` (SQLite is
the cross-process seam) and the same SeaweedFS bucket, so the container's API
can report on blobs the runner stored without ever needing an S3 client itself.

Graceful degradation
--------------------
If boto3 is missing or the SeaweedFS endpoint is unreachable, `get_store()`
falls back to a local directory. That keeps unit tests network-free and lets
the pipeline run anywhere there is a writable directory, not only where
SeaweedFS lives. The chosen backend is reported via `store_kind()` so the UI
can say "stored on SeaweedFS" vs "stored locally (dev)".

Environment
-----------
    SEAWEED_S3_ENDPOINT   e.g. http://192.168.0.1:8333   (None → local fallback)
    SEAWEED_ACCESS_KEY    S3 access key (agent_key)
    SEAWEED_SECRET_KEY    S3 secret key
    SEAWEED_BUCKET        bucket name (default: arxiv-papers)
    SEAWEED_REGION        region (default: us-east-1)
    STORAGE_LOCAL_DIR     local fallback root (default: <state>/blob_store)

The bucket is created on first use if missing — SeaweedFS gives the configured
identity Admin, so the runner bootstraps its own bucket.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Optional

# Kinds of blob we store. Used as a path segment so the same sha256 can have
# both the source PDF and the parsed bundle without colliding.
KIND_PDF = "pdf"
KIND_BUNDLE = "bundle"
VALID_KINDS = (KIND_PDF, KIND_BUNDLE)

# Default parser tag for a bundle. The same PDF parsed by different algorithms
# lives side by side: <sha>/dots_mocr.zip, <sha>/classic_fitz.zip, ... so a
# parser name is part of every bundle's storage key. PDFs are parser-agnostic
# (they're the source bytes, not a parse result), so KIND_PDF ignores it.
DEFAULT_PARSER = "dots_mocr"


class BlobStore:
    """Interface every backend implements.

    Every method is content-addressed: the caller already has the sha256
    (the pipeline computes it from the arxiv PDF bytes), so we use it as the
    primary key. A repeat put/get is idempotent by construction.
    """

    name = "blob-store"

    def put(self, sha256: str, kind: str, data: bytes, parser: str = "") -> str:
        """Store `data` under (sha256, kind[, parser]). Returns a storage key."""
        raise NotImplementedError

    def get(self, sha256: str, kind: str, parser: str = "") -> Optional[bytes]:
        """Return the bytes, or None if absent."""
        raise NotImplementedError

    def exists(self, sha256: str, kind: str, parser: str = "") -> bool:
        raise NotImplementedError

    def delete(self, sha256: str, kind: str, parser: str = "") -> None:
        raise NotImplementedError

    def size(self, sha256: str, kind: str, parser: str = "") -> Optional[int]:
        raise NotImplementedError

    def kind(self) -> str:
        return self.name


# ----------------------------------------------------------------- local disk

class LocalBlobStore(BlobStore):
    """Filesystem backend: one file per (sha256, kind), sharded two levels deep.

    Sharding (ab/cd/<full>) keeps a single directory from holding tens of
    thousands of entries, which gets slow on most filesystems and painful to
    `ls`. The sharding is internal — the public key is the bare sha256.
    """

    name = "local"

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str, kind: str, parser: str = "") -> Path:
        sha = sha256.lower()
        shard = sha[:2] + "/" + sha[2:4]
        # PDFs are parser-agnostic (source bytes); bundles carry a parser tag so
        # the same sha can hold dots_mocr.zip and classic_fitz.zip side by side.
        stem = f"{sha}.{parser}.{kind}" if parser and kind == KIND_BUNDLE else f"{sha}.{kind}"
        return self.root / shard / stem

    def put(self, sha256: str, kind: str, data: bytes, parser: str = "") -> str:
        path = self._path(sha256, kind, parser)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write to a temp file in the same dir and rename, so a crash mid-write
        # never leaves a half-written blob that looks complete
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return str(path.relative_to(self.root))

    def get(self, sha256: str, kind: str, parser: str = "") -> Optional[bytes]:
        path = self._path(sha256, kind, parser)
        if not path.is_file():
            return None
        return path.read_bytes()

    def exists(self, sha256: str, kind: str, parser: str = "") -> bool:
        return self._path(sha256, kind, parser).is_file()

    def delete(self, sha256: str, kind: str, parser: str = "") -> None:
        path = self._path(sha256, kind, parser)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def size(self, sha256: str, kind: str, parser: str = "") -> Optional[int]:
        path = self._path(sha256, kind, parser)
        return path.stat().st_size if path.is_file() else None


# ----------------------------------------------------------------- SeaweedFS / S3

class SeaweedBlobStore(BlobStore):
    """S3-compatible backend (SeaweedFS s3 gateway, minio, real S3, ...).

    boto3 is imported lazily so the module loads even when boto3 is absent
    (the LocalBlobStore path stays usable in tests and on minimal hosts).
    """

    name = "seaweedfs"

    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str = "arxiv-papers", region: str = "us-east-1"):
        try:
            import boto3  # noqa: F401
            from botocore.client import Config  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "boto3 is required for the SeaweedFS backend but is not "
                "installed; install it or unset SEAWEED_S3_ENDPOINT to use "
                "the local fallback") from error
        import boto3
        from botocore.client import Config

        self.endpoint = endpoint.rstrip("/")
        self.bucket = bucket
        self.region = region
        # SeaweedFS supports path-style addressing; AWS v4 signing works, but
        # the signature must not be cached (SignatureV4 with a virtual clock
        # is fine) — keep it explicit so behaviour is identical against minio.
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
                connect_timeout=10,
                read_timeout=120,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        import boto3
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return
        except Exception:  # noqa: BLE001 — any error → try to create
            pass
        try:
            self._client.create_bucket(Bucket=self.bucket)
        except self._client.exceptions.BucketAlreadyOwnedByYou:
            pass

    @staticmethod
    def _key(sha256: str, kind: str, parser: str = "") -> str:
        sha = sha256.lower()
        # same sharding as the local backend so `ls` stays manageable inside
        # the filer; the path also reads naturally in the SeaweedFS web UI.
        stem = f"{sha}.{parser}.{kind}" if parser and kind == KIND_BUNDLE else f"{sha}.{kind}"
        return f"{sha[:2]}/{sha[2:4]}/{stem}"

    def put(self, sha256: str, kind: str, data: bytes, parser: str = "") -> str:
        key = self._key(sha256, kind, parser)
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return key

    def get(self, sha256: str, kind: str, parser: str = "") -> Optional[bytes]:
        try:
            response = self._client.get_object(Bucket=self.bucket,
                                               Key=self._key(sha256, kind, parser))
            return response["Body"].read()
        except self._client.exceptions.NoSuchKey:
            return None
        except Exception:  # noqa: BLE001 — connection blips should not crash reads
            return None

    def exists(self, sha256: str, kind: str, parser: str = "") -> bool:
        try:
            self._client.head_object(Bucket=self.bucket,
                                     Key=self._key(sha256, kind, parser))
            return True
        except self._client.exceptions.ClientError:
            return False
        except Exception:  # noqa: BLE001
            return False

    def delete(self, sha256: str, kind: str, parser: str = "") -> None:
        try:
            self._client.delete_object(Bucket=self.bucket,
                                       Key=self._key(sha256, kind, parser))
        except Exception:  # noqa: BLE001
            pass

    def size(self, sha256: str, kind: str, parser: str = "") -> Optional[int]:
        try:
            response = self._client.head_object(
                Bucket=self.bucket, Key=self._key(sha256, kind, parser))
            return int(response.get("ContentLength") or 0)
        except Exception:  # noqa: BLE001
            return None


# ----------------------------------------------------------------- factory

# Module-level singleton: the backend is constructed once per process from the
# environment. Tests call configure_store() to inject a fresh instance instead
# of reading env vars.
_STORE: Optional[BlobStore] = None


def configure_store(store: Optional[BlobStore]) -> None:
    """Override the process-wide store (used by tests). Pass None to reset."""
    global _STORE
    _STORE = store


def _build_from_env() -> BlobStore:
    endpoint = os.environ.get("SEAWEED_S3_ENDPOINT")
    # An endpoint pointing at SeaweedFS → try the S3 backend, fall back to
    # local if boto3 is missing or the cluster is unreachable at startup.
    if endpoint:
        try:
            return SeaweedBlobStore(
                endpoint=endpoint,
                access_key=os.environ.get("SEAWEED_ACCESS_KEY", ""),
                secret_key=os.environ.get("SEAWEED_SECRET_KEY", ""),
                bucket=os.environ.get("SEAWEED_BUCKET", "arxiv-papers"),
                region=os.environ.get("SEAWEED_REGION", "us-east-1"),
            )
        except Exception as error:  # noqa: BLE001 — degrade, don't crash
            print(f"[storage] SeaweedFS unavailable ({type(error).__name__}: "
                  f"{error}); falling back to local disk", flush=True)
    return LocalBlobStore(os.environ.get(
        "STORAGE_LOCAL_DIR",
        str(Path(os.environ.get("DEMO_STATE_DIR", "/state")) / "blob_store")))


def get_store() -> BlobStore:
    """Return the process-wide BlobStore, creating it from env on first use."""
    global _STORE
    if _STORE is None:
        _STORE = _build_from_env()
    return _STORE


def store_kind() -> str:
    """Human-readable backend name for the UI ('seaweedfs' | 'local')."""
    return get_store().kind()


# ----------------------------------------------------------------- helpers

def sha256_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes — the content-addressed key for blobs.

    Mirrors docstore.sha256_of (which hashes a file path); this variant takes
    bytes directly because the pipeline downloads a PDF into memory before it
    has a stable path, and the hash must match what the OCR service computed
    from the uploaded bytes for dedup to line up.
    """
    return hashlib.sha256(data).hexdigest()


def put_pdf(sha256: str, data: bytes) -> str:
    """Convenience: store a source PDF; idempotent on identical bytes."""
    return get_store().put(sha256, KIND_PDF, data)


def get_pdf(sha256: str) -> Optional[bytes]:
    return get_store().get(sha256, KIND_PDF)


def has_pdf(sha256: str) -> bool:
    return get_store().exists(sha256, KIND_PDF)


def put_bundle(sha256: str, data: bytes, parser: str = DEFAULT_PARSER) -> str:
    """Store a parsed bundle, tagged by which parser produced it.

    The same PDF parsed by dots.mocr and by the classic text extractor lands
    at different keys (<sha>.dots_mocr.bundle vs <sha>.classic_fitz.bundle),
    so neither overwrites the other and both stay available for comparison.
    """
    return get_store().put(sha256, KIND_BUNDLE, data, parser=parser)


def get_bundle(sha256: str, parser: str = DEFAULT_PARSER) -> Optional[bytes]:
    return get_store().get(sha256, KIND_BUNDLE, parser=parser)


def has_bundle(sha256: str, parser: str = DEFAULT_PARSER) -> bool:
    return get_store().exists(sha256, KIND_BUNDLE, parser=parser)
