"""Markdown -> PDF export for the demo results panel.

No new dependencies on purpose: the demo container reuses its image across
deploys (the repo is bind-mounted, `docker restart` picks code changes up), so
a fresh pip package here would force an image rebuild. PyMuPDF (fitz) is
already a core dependency and its Story engine lays out a practical HTML
subset (headings, lists, tables, images resolved through an Archive).

The converter is intentionally a subset: OCR markdown is headings, paragraphs,
lists, tables, code fences, picture links and LaTeX math. Math is shown
verbatim in a monospace span — Story cannot typeset TeX, and the raw source is
more useful in an export than a silently dropped formula.
"""

from __future__ import annotations

import html
import io
import re

import fitz

_CSS = """
body { font-family: sans-serif; font-size: 10pt; }
h1 { font-size: 16pt; } h2 { font-size: 14pt; } h3 { font-size: 12pt; }
code, pre { font-family: monospace; font-size: 9pt; }
pre { background-color: #f2f2f2; padding: 4px; }
.math { font-family: monospace; color: #333333; }
blockquote { color: #555555; }
"""

_IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_LIST_RE = re.compile(r"^([-*+]|\d+[.)])\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\|?[\s:|-]+\|?$")


def _inline(text):
    """Escape, then re-introduce the inline markup the Story engine supports."""
    out = html.escape(text, quote=False)
    out = _IMG_RE.sub(
        lambda m: f'<img src="{m.group(2)}" alt="{m.group(1)}">', out)
    out = _LINK_RE.sub(
        lambda m: (f'<a href="{m.group(2)}">{m.group(1)}</a>'
                   if m.group(2).startswith(("http://", "https://"))
                   else m.group(1)),
        out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    # LaTeX survives escaping untouched; render it verbatim but visually distinct
    out = re.sub(r"\$\$([^$]+)\$\$", r'<span class="math">\1</span>', out)
    out = re.sub(r"\$([^$\n]+)\$", r'<span class="math">\1</span>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"__([^_]+)__", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", out)
    return out


def _table_html(rows):
    def cells(row):
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(rows[0])
    body = [cells(r) for r in rows[1:] if not _TABLE_SEP_RE.match(r)]
    parts = ['<table border="1"><tr>'
             + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>"]
    for row in body:
        parts.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
    parts.append("</table>")
    return "".join(parts)


def _starts_block(stripped):
    return (stripped.startswith("```")
            or _HEADING_RE.match(stripped)
            or _HR_RE.match(stripped)
            or _LIST_RE.match(stripped)
            or stripped.startswith(">")
            or (stripped.startswith("|") and stripped.endswith("|")))


def markdown_to_html(md_text):
    """The supported subset -> HTML body content (no <html> wrapper)."""
    lines = md_text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # the closing fence (or EOF)
            blocks.append("<pre>" + html.escape("\n".join(code)) + "</pre>")
            continue
        match = _HEADING_RE.match(stripped)
        if match:
            level = len(match.group(1))
            blocks.append(f"<h{level}>{_inline(match.group(2))}</h{level}>")
            i += 1
            continue
        if _HR_RE.match(stripped):
            blocks.append("<hr>")
            i += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            blocks.append(_table_html(rows))
            continue
        match = _LIST_RE.match(stripped)
        if match:
            ordered = match.group(1)[0].isdigit()
            items = []
            while i < len(lines):
                item = _LIST_RE.match(lines[i].strip())
                if not item:
                    break
                items.append(f"<li>{_inline(item.group(2))}</li>")
                i += 1
            tag = "ol" if ordered else "ul"
            blocks.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue
        if stripped.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue
        # paragraph: consecutive plain lines. Keep the newlines as <br/> — OCR
        # output carries meaningful line breaks.
        para = []
        while i < len(lines) and lines[i].strip() and not _starts_block(lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        blocks.append(f"<p>{'<br/>'.join(_inline(p) for p in para)}</p>")
    return "\n".join(blocks)


def _render_pdf(html_doc, archive):
    story = fitz.Story(html=html_doc, user_css=_CSS, archive=archive)
    buffer = io.BytesIO()
    writer = fitz.DocumentWriter(buffer)
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (36, 36, -36, -36)
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def markdown_to_pdf(md_text, assets_dir=None, title="document"):
    """Render markdown to PDF bytes.

    `assets_dir` is the directory the markdown's relative image links
    (images/foo.png) resolve against — the task's out dir in the demo.
    An exotic construct must never kill the export: on any Story failure the
    document degrades to verbatim text.
    """
    body = markdown_to_html(md_text)
    html_doc = (f"<html><head><title>{html.escape(title)}</title></head>"
                f"<body>{body}</body></html>")
    archive = fitz.Archive(str(assets_dir)) if assets_dir else None
    try:
        return _render_pdf(html_doc, archive)
    except Exception:
        plain = ("<html><body><p>"
                 + html.escape(md_text).replace("\n", "<br/>")
                 + "</p></body></html>")
        return _render_pdf(plain, None)
