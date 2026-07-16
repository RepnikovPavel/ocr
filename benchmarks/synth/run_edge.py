"""Render compiled synthetic PDFs, run dots.mocr, score, summarize the edge.

Consumes the output of compile.py (one PDF per group + manifest.json) plus a
model checkpoint. For each case it runs one layout parse, extracts the
kind-relevant text, scores it, and writes a per-case + aggregated report.
"""

import argparse
import json
import time
from pathlib import Path

from benchmarks.synth import scoring


def _extract(kind, cells, filtered, raw_response):
    """Derive the text a scorer needs from the layout parse of one page."""
    if filtered or not isinstance(cells, list):
        text = raw_response if isinstance(raw_response, str) else ""
        return {"table": text, "formula": text, "algorithm": text, "code": text,
                "full_text": text, "n_cells": None, "categories": {}}

    def cat_text(category):
        return "\n".join(str(c.get("text", "")) for c in cells
                         if c.get("category") == category)

    full_text = "\n".join(str(c.get("text", "")) for c in cells)
    categories = {}
    for c in cells:
        categories[c.get("category", "?")] = categories.get(c.get("category", "?"), 0) + 1
    return {
        "table": cat_text("Table") or full_text,
        "formula": cat_text("Formula") or full_text,
        "algorithm": full_text,
        "code": full_text,
        "full_text": full_text,
        "n_cells": len(cells),
        "categories": categories,
    }


def _infer_page(parser, origin_image, prompt_mode="prompt_layout_all_en"):
    from dots_mocr.utils.image_utils import fetch_image, smart_resize
    from dots_mocr.utils.layout_utils import post_process_output

    image = fetch_image(origin_image, min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)
    prompt = parser.get_prompt(prompt_mode, origin_image=origin_image, image=image)
    response = parser._inference(image, prompt, temperature=0.0)
    result = post_process_output(response, prompt_mode, origin_image, image,
                                 min_pixels=parser.min_pixels, max_pixels=parser.max_pixels)
    if isinstance(result, tuple):
        cells, filtered = result
    else:
        cells, filtered = result, True
    return response, cells, filtered


def run(args):
    import torch  # noqa: F401

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.doc_utils import load_pdf_pages

    pdf_dir = Path(args.pdf_dir)
    manifest = json.loads((pdf_dir / "manifest.json").read_text(encoding="utf-8"))
    kinds = set(args.kinds.split(",")) if args.kinds else None

    parser = DotsMOCRParser(
        ckpt=args.ckpt, device=args.device, dtype="bfloat16",
        temperature=0.0, max_completion_tokens=args.max_new_tokens,
        dpi=args.dpi, max_pixels=args.max_pixels, num_thread=1,
    )

    results = []
    for group, group_cases in manifest["groups"].items():
        if kinds and group_cases and group_cases[0]["kind"] not in kinds:
            continue
        pdf_path = pdf_dir / f"{group}.pdf"
        if not pdf_path.exists():
            print(f"[skip] {pdf_path} missing")
            continue
        pages = load_pdf_pages(str(pdf_path), dpi=args.dpi)
        if len(pages) != len(group_cases):
            print(f"[warn] {group}: {len(pages)} pages != {len(group_cases)} cases; "
                  "scoring by min length")
        for (page_idx, image), case in zip(pages, group_cases):
            t0 = time.time()
            response, cells, filtered = _infer_page(parser, image)
            extracted = _extract(case["kind"], cells, filtered, response)
            metrics = scoring.score_case(case["kind"], extracted[case["kind"]], case["ground_truth"])
            primary = metrics.get(scoring.PRIMARY_METRIC[case["kind"]])
            record = {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "group": group,
                "params": case["params"],
                "primary_metric": scoring.PRIMARY_METRIC[case["kind"]],
                "primary_value": primary,
                "metrics": metrics,
                "categories": extracted["categories"],
                "filtered": filtered,
                "seconds": round(time.time() - t0, 2),
                "output_chars": len(extracted["full_text"]),
            }
            if args.save_outputs:
                record["model_text"] = extracted[case["kind"]][:4000]
            results.append(record)
            print(f"[{group}] {case['case_id']}: "
                  f"{record['primary_metric']}={primary} ({record['seconds']}s)", flush=True)

    summary = summarize(results)
    report = {
        "config": {
            "ckpt": args.ckpt, "dpi": args.dpi, "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens, "device": args.device,
        },
        "summary": summary,
        "cases": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nreport: {out}")


def summarize(results):
    """Per-kind and per-group means, plus the detected 'edge' per family."""
    by_kind = {}
    for r in results:
        by_kind.setdefault(r["kind"], []).append(r["primary_value"] or 0.0)
    kind_summary = {k: round(sum(v) / len(v), 4) for k, v in by_kind.items() if v}

    # edge: for each family (group + a difficulty knob), report score vs difficulty
    families = {}
    for r in results:
        knob = _difficulty_knob(r)
        fam = r.get("params", {}).get("family") or r["group"]
        families.setdefault(fam, []).append((knob, r["case_id"], r["primary_value"] or 0.0))
    edges = {}
    for fam, points in families.items():
        points.sort(key=lambda p: (p[0] is None, p[0]))
        edges[fam] = [
            {"knob": knob, "case_id": cid, "score": round(score, 4)}
            for knob, cid, score in points
        ]
    return {"by_kind": kind_summary, "n_cases": len(results), "families": edges}


def _difficulty_knob(record):
    p = record.get("params", {})
    for key in ("cols", "rows", "level", "n_lines", "n_extra"):
        if key in p:
            return p[key]
    return None


def build_parser():
    parser = argparse.ArgumentParser(description="synthetic edge run for dots.mocr")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kinds", default=None, help="comma list: table,formula,algorithm,code")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=2_200_000)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--save-outputs", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
