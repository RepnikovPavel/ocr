"""Validate the synthetic edge findings on real data (the master's diploma).

Ground truth is the PDF's own text layer (the true rendered text). For each
chosen page we run dots.mocr layout parsing and measure how faithfully the
model reproduces the page: content-word recall (Cyrillic + Latin + digits),
character-level similarity, and — for algorithm pages — pseudocode keyword
recall. This directly checks whether the synthetic predictions
(algorithms/tables robust, deeply nested fractions break) hold on real pages.
"""

import argparse
import difflib
import json
import re
import time
from pathlib import Path

_CONTENT = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")
_KEYWORDS = ["while", "for", "if", "then", "do", "return", "end",
             "repeat", "until", "procedure", "function"]


def content_tokens(text):
    return set(t.lower() for t in _CONTENT.findall(text or ""))


def norm_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def score_page(model_text, gt_text):
    gt_toks = content_tokens(gt_text)
    got_toks = content_tokens(model_text)
    recall = len(gt_toks & got_toks) / len(gt_toks) if gt_toks else 0.0

    gt_ints = set(int(x) for x in re.findall(r"\d+", gt_text or ""))
    got_ints = set(int(x) for x in re.findall(r"\d+", model_text or ""))
    num_recall = len(gt_ints & got_ints) / len(gt_ints) if gt_ints else None

    similarity = difflib.SequenceMatcher(None, norm_ws(gt_text), norm_ws(model_text)).ratio()

    gt_lower = (gt_text or "").lower()
    present_kw = [k for k in _KEYWORDS if re.search(r"\b" + k + r"\b", gt_lower)]
    got_lower = (model_text or "").lower()
    kw_hit = [k for k in present_kw if re.search(r"\b" + k + r"\b", got_lower)]
    kw_recall = len(kw_hit) / len(present_kw) if present_kw else None

    return {
        "content_word_recall": round(recall, 4),
        "number_recall": round(num_recall, 4) if num_recall is not None else None,
        "char_similarity": round(similarity, 4),
        "algo_keyword_recall": round(kw_recall, 4) if kw_recall is not None else None,
        "keywords_present": present_kw,
        "gt_word_count": len(gt_toks),
        "gt_chars": len(gt_text or ""),
    }


def run(args):
    import fitz

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image
    from dots_mocr.utils.layout_utils import post_process_output

    pages = sorted(int(x) - 1 for x in args.pages.split(","))  # 1-based -> 0-based
    doc = fitz.open(args.pdf)
    gt_by_page = {p: doc[p].get_text() for p in pages}
    doc.close()

    parser = DotsMOCRParser(
        ckpt=args.ckpt, device=args.device, dtype="bfloat16",
        temperature=0.0, max_completion_tokens=args.max_new_tokens,
        dpi=args.dpi, max_pixels=args.max_pixels, num_thread=1,
    )

    rendered = load_pdf_pages(args.pdf, dpi=args.dpi, page_ids=pages)
    results = []
    for page_idx, image in rendered:
        t0 = time.time()
        img = fetch_image(image, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)
        prompt = parser.get_prompt("prompt_layout_all_en", origin_image=image, image=img)
        response = parser._inference(img, prompt, temperature=0.0)
        result = post_process_output(response, "prompt_layout_all_en", image, img,
                                     min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)
        cells, filtered = result if isinstance(result, tuple) else (result, True)
        if not filtered and isinstance(cells, list):
            model_text = "\n".join(str(c.get("text", "")) for c in cells)
            categories = {}
            for c in cells:
                categories[c.get("category", "?")] = categories.get(c.get("category", "?"), 0) + 1
        else:
            model_text = response if isinstance(response, str) else ""
            categories = {}
        metrics = score_page(model_text, gt_by_page[page_idx])
        record = {
            "page": page_idx + 1,
            "kind": args.label_pages.get(str(page_idx + 1), "text"),
            "metrics": metrics,
            "categories": categories,
            "filtered": filtered,
            "seconds": round(time.time() - t0, 2),
        }
        if args.save_outputs:
            record["model_text"] = model_text[:6000]
        results.append(record)
        print(f"[page {page_idx + 1}] word_recall={metrics['content_word_recall']} "
              f"sim={metrics['char_similarity']} "
              f"kw={metrics['algo_keyword_recall']} ({record['seconds']}s)", flush=True)

    summary = {
        "mean_content_word_recall": round(
            sum(r["metrics"]["content_word_recall"] for r in results) / len(results), 4),
        "mean_char_similarity": round(
            sum(r["metrics"]["char_similarity"] for r in results) / len(results), 4),
        "n_pages": len(results),
    }
    algo_pages = [r for r in results if r["metrics"]["algo_keyword_recall"] is not None]
    if algo_pages:
        summary["mean_algo_keyword_recall"] = round(
            sum(r["metrics"]["algo_keyword_recall"] for r in algo_pages) / len(algo_pages), 4)

    report = {
        "config": {"pdf": args.pdf, "dpi": args.dpi, "max_pixels": args.max_pixels},
        "summary": summary,
        "pages": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {out}")


def build_parser():
    parser = argparse.ArgumentParser(description="diploma real-data validation")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pages", required=True, help="1-based comma list")
    parser.add_argument("--labels", default="", help="page:kind comma list, e.g. 11:algorithm,15:table")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=2_200_000)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--save-outputs", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.label_pages = {}
    for item in args.labels.split(","):
        if ":" in item:
            page, kind = item.split(":", 1)
            args.label_pages[page.strip()] = kind.strip()
    run(args)
