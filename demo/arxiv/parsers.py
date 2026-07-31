"""Parser registry: turn a PDF into a result bundle, tagged by which parser
produced it.

The dots.mocr parser already exists — it goes through the OCR service over
HTTP (see pipeline.submit_to_ocr / wait_for_ocr). This module owns the
GPU-less "classic" parsers that extract text straight from the PDF: they are
fast, deterministic, and need no model, which makes them the right baseline
both to seed a corpus quickly and to compare dots.mocr against.

Every parser yields a bundle in the SAME on-disk shape the OCR service
produces (document.md + meta.json [+ images/] [+ layout/]), so the rest of
the pipeline — storage, the /bundle endpoint, the UI — does not care which
parser made a given bundle.

Interface
---------
    parse(pdf_bytes) -> dict  with keys:
        markdown   str          the document text
        meta       dict         at least {parser, pages, pages_done}
        images     list[(name, bytes)]   optional extracted figures

`build_bundle(markdown, meta, images)` packs that into the canonical zip so
the storage path and bundle layout stay identical across parsers.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Callable, Optional

# Registry: parser name -> parse function. The dots.mocr entry is here for
# discoverability but is special-cased in the pipeline (it's the only one that
# goes over the network to the OCR service, not a pure function of the bytes).
DOTS_MOCR = "dots_mocr"
CLASSIC_FITZ = "classic_fitz"
CLASSIC_PDFPLUMBER = "classic_pdfplumber"

REGISTRY: dict[str, Callable[[bytes], dict]] = {}

# Parsers that run locally on the PDF bytes (no OCR service). The pipeline
# uses this set to decide whether to dispatch to the OCR client or call the
# function directly.
LOCAL_PARSERS: set[str] = set()


def register(name: str, fn: Callable[[bytes], dict], *, local: bool = True) -> None:
    REGISTRY[name] = fn
    if local:
        LOCAL_PARSERS.add(name)


def available_parsers() -> list[str]:
    """Names the runner/UI may offer."""
    return sorted(REGISTRY.keys())


def is_local(name: str) -> bool:
    return name in LOCAL_PARSERS


def parse_locally(name: str, pdf_bytes: bytes) -> dict:
    """Run a local parser by name; raises KeyError if unknown."""
    return REGISTRY[name](pdf_bytes)


# ----------------------------------------------------------------- bundle packer

def build_bundle(markdown: str, meta: dict,
                 images: Optional[list[tuple[str, bytes]]] = None) -> bytes:
    """Pack a parse result into the canonical zip layout.

    Matches the OCR service's bundle: document.md + meta.json at the root,
    images/ for figures the markdown references, layout/ for per-page JSON.
    Every bundle carries its parser name in meta.json so a reader can always
    tell what produced it, even after download.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("document.md", markdown)
        archive.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        for name, data in (images or []):
            archive.writestr(f"images/{name}", data)
    return buf.getvalue()


# ----------------------------------------------------------------- classic: fitz

def _parse_fitz(pdf_bytes: bytes) -> dict:
    """PyMuPDF text extraction: get_text() per page, stitched to markdown.

    Fast (no model), deterministic, and the right baseline for born-digital
    arxiv PDFs whose text layer is intact. Markdown is one '# Page N' header
    per page followed by the page's text — coarse but lossless for search and
    comparison against the layout-aware dots.mocr output.
    """
    import fitz

    pages_text = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        num_pages = doc.page_count
        for index in range(num_pages):
            page = doc.load_page(index)
            text = page.get_text("text") or ""
            text = text.strip()
            pages_text.append((index, text))
    markdown = "\n\n".join(f"# Page {i + 1}\n\n{txt}" for i, txt in pages_text)
    return {
        "markdown": markdown,
        "meta": {
            "parser": CLASSIC_FITZ,
            "pages": num_pages,
            "pages_done": num_pages,
            "method": "PyMuPDF get_text",
        },
        "images": [],
    }


register(CLASSIC_FITZ, _parse_fitz, local=True)


# ----------------------------------------------------------------- classic: pdfplumber

def _parse_pdfplumber(pdf_bytes: bytes) -> dict:
    """pdfplumber extraction — better table structure, heavier dependency.

    Falls back to per-page extract_text() but keeps table cells as markdown
    tables when pdfplumber detects them, which matters for papers heavy on
    results tables. Raises a clear error if pdfplumber is not installed
    (it's an optional dependency, unlike PyMuPDF).
    """
    try:
        import pdfplumber
    except ImportError as error:
        raise RuntimeError(
            "classic_pdfplumber needs the 'pdfplumber' package; install it in "
            "the runner venv (pip install pdfplumber)") from error

    pages_md = []
    num_pages = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        num_pages = len(pdf.pages)
        for index, page in enumerate(pdf.pages):
            chunks = []
            # tables first, then the remaining text, so a table's rows stay
            # together rather than being interleaved with surrounding prose
            for table in page.extract_tables() or []:
                chunks.append(_table_to_markdown(table))
            text = (page.extract_text() or "").strip()
            if text:
                chunks.append(text)
            pages_md.append(f"# Page {index + 1}\n\n" + "\n\n".join(chunks).strip())
    return {
        "markdown": "\n\n".join(pages_md),
        "meta": {
            "parser": CLASSIC_PDFPLUMBER,
            "pages": num_pages,
            "pages_done": num_pages,
            "method": "pdfplumber extract_text + tables",
        },
        "images": [],
    }


def _table_to_markdown(table) -> str:
    """Render a pdfplumber table (list of rows) as a markdown table."""
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in (table or [])]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join([header, sep, body])


register(CLASSIC_PDFPLUMBER, _parse_pdfplumber, local=True)
