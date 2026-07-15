#!/usr/bin/env python3
"""Simple benchmark for dots.mocr: seconds per page on image or PDF.

Usage (inside container or with PYTHONPATH):
  python3 -m demo.bench --ckpt /models --input /path/to/test.pdf --prompt prompt_layout_all_en --pages 5
"""
import argparse
import time
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dots_mocr.cli import DotsMOCRParser
from dots_mocr.utils.prompts import dict_promptmode_to_prompt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--prompt", default="prompt_layout_all_en", choices=list(dict_promptmode_to_prompt.keys()))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--num_thread", type=int, default=4)
    p.add_argument("--max_pages", type=int, default=None, help="limit pages for quick bench")
    args = p.parse_args()

    parser = DotsMOCRParser(
        ckpt=args.ckpt,
        temperature=0.1,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir="/tmp/bench_out",
        device="auto",
    )

    pages = None
    if args.max_pages and args.input.lower().endswith(".pdf"):
        import fitz
        with fitz.open(args.input) as doc:
            pages = list(range(min(args.max_pages, doc.page_count)))

    t0 = time.time()
    res = parser.parse_file(args.input, prompt_mode=args.prompt, pages=pages)
    elapsed = time.time() - t0

    n = max(1, len(res))
    spp = elapsed / n
    print(f"Pages: {n}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"sec/page: {spp:.2f}")
    print("Per page approx (first few):")
    for i, r in enumerate(res[:3]):
        print(f"  page {r.get('page_no', i)}: input {r.get('input_height')}x{r.get('input_width')}")

if __name__ == "__main__":
    main()
