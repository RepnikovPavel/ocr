"""Per-page parse cache for the arxiv pipeline (dots.mocr path).

Why per page, not per document
------------------------------
The document-level cache stores one bundle zip per (doc sha256, parser). On a
large document that means: one changed page in a new arxiv version re-OCRs the
WHOLE paper, a failed parse loses all finished pages, and every paper version
is one giant monolithic blob. This module adds the page granularity:

- **identity** — sha256 of the page rendered to PNG at a fixed DPI
  (`page_hashes`). Stable across re-uploads and across arxiv versions whose
  page pixels did not change; independent of PDF metadata churn.
- **artifact** — the OCR service's own 1-page bundle zip, stored losslessly as
  `<page_sha>.<parser>.page` (see demo/storage.py). No format conversion, so
  what comes out of the cache is byte-identical to what the OCR produced.
- **pagemap** — JSON manifest per (doc sha256, parser) listing the page hashes
  in order, stored as `<doc_sha>.<parser>.pagemap`. A re-run whose pagemap and
  page blobs are all present skips OCR entirely and just reassembles.

Assembly (`assemble_bundle`) rebuilds the canonical document bundle
(document.md + meta.json + images/ + layout/) from the 1-page bundles, so
downstream serving (/api/v1/arxiv/papers/{id}/bundle) is unchanged.

Only the dots_mocr (OCR service) path is page-cached — that is the expensive
parse. The classic CPU parsers (classic_fitz, ...) stay monolithic: re-running
them costs seconds, and page granularity would only add moving parts.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from typing import Optional

# Page identity must not drift with caller settings, so the render DPI is a
# module constant, not a parameter callers can bump.
RENDER_DPI = 150

# The worker joins per-page markdown into document.md with exactly this
# separator (demo/worker.py `_record_in_docstore`) — assembly mirrors it.
MD_JOIN = "\n\n"


def page_hashes(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[str]:
    """Content hash of every page, in page order.

    Rendering to PNG first makes the hash insensitive to PDF-level metadata
    edits (producer strings, xref layout) that change the file bytes but not
    the page content — exactly what an arxiv v1 -> v2 bump looks like.
    """
    import fitz  # PyMuPDF — a runner-side dependency, not needed container-side

    hashes = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            png = page.get_pixmap(dpi=dpi).tobytes("png")
            hashes.append(hashlib.sha256(png).hexdigest())
    return hashes


def build_pagemap(page_shas: list[str]) -> bytes:
    """Serialize the pagemap manifest for a document."""
    return json.dumps(
        {"pages": page_shas, "dpi": RENDER_DPI, "version": 1},
    ).encode("utf-8")


def parse_pagemap(data: bytes) -> Optional[list[str]]:
    """Read a pagemap blob; None if it is corrupt or from another scheme."""
    try:
        manifest = json.loads(data)
        pages = manifest["pages"]
        if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
            return None
        return pages
    except Exception:  # noqa: BLE001 — a bad manifest simply means "re-parse"
        return None


def split_page_bundle(bundle_bytes: bytes) -> dict:
    """A 1-page OCR bundle zip -> its parts (markdown, images, layout, meta)."""
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        names = archive.namelist()

        def _read(name):
            return archive.read(name) if name in names else None

        raw_md = _read("document.md")
        raw_meta = _read("meta.json")
        return {
            "markdown": raw_md.decode("utf-8") if raw_md is not None else "",
            "meta": json.loads(raw_meta) if raw_meta is not None else {},
            "images": {
                name[len("images/"):]: archive.read(name)
                for name in names
                if name.startswith("images/") and not name.endswith("/")
            },
            "layout": {
                name[len("layout/"):]: archive.read(name)
                for name in names
                if name.startswith("layout/") and not name.endswith("/")
            },
        }


def assemble_bundle(page_bundles: list[bytes], *, sha256: str,
                    prompt_mode: str, parser: str) -> bytes:
    """Rebuild the canonical whole-document bundle from 1-page bundles.

    The output layout matches what the OCR service serves for a full-document
    parse, so readers cannot tell an assembled bundle from a monolithic one.
    Image and layout names carry their original task/page ids, so merging
    across pages cannot collide; the markdown's relative image links keep
    resolving inside the archive.
    """
    parts = [split_page_bundle(bundle) for bundle in page_bundles]
    markdown = MD_JOIN.join(part["markdown"] for part in parts)
    tokens = sum((part["meta"].get("generated_tokens") or 0) for part in parts)
    seconds = round(
        sum((part["meta"].get("seconds") or 0.0) for part in parts), 2)
    meta = {
        "sha256": sha256,
        "prompt_mode": prompt_mode,
        "pages_done": len(parts),
        "generated_tokens": tokens or None,
        "seconds": seconds or None,
        "parser": parser,
        "assembled_from_pages": True,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document.md", markdown)
        archive.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for part in parts:
            for name, data in part["images"].items():
                archive.writestr(f"images/{name}", data)
            for name, data in part["layout"].items():
                archive.writestr(f"layout/{name}", data)
    return buffer.getvalue()
