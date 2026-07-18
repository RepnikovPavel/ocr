#!/usr/bin/env python3
"""Benchmark this repository's transformers inference against a vLLM server.

The question this answers: is the hand-tuned path in this repo anywhere near a
production engine on the same card, or is vLLM out of the box already ahead?

Both sides get the identical page, the identical prompt and greedy decoding.
Timing is taken the same way on both: time to first token from the request, then
the steady-state rate over the remaining tokens, so vLLM's streaming response and
our GenerationStats measure the same two things.

    # 1. serve (see docker/Dockerfile.vllm)
    # 2. measure
    python3 -m benchmarks.bench_vllm --ckpt $CKPT --input doc.pdf --pages 0 \
        --vllm-url http://127.0.0.1:8000/v1 --output-dir reports/vllm

Add --concurrency 1,2,4,8 to also measure vLLM's continuous batching, which is
where an engine is expected to pull far ahead of a single-request loop.
"""

import argparse
import base64
import io
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pages", default="0")
    parser.add_argument("--prompt", default="prompt_layout_all_en")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pixels", type=int, default=2_200_000)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="rednote-hilab/dots.mocr")
    parser.add_argument("--concurrency", default="1",
                        help="comma separated request counts to fire at vLLM at once")
    parser.add_argument("--skip-local", action="store_true",
                        help="only measure vLLM (the local side needs the GPU to itself)")
    parser.add_argument("--output-dir", default="reports/vllm")
    return parser


def render_pages(args):
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image

    if Path(args.input).suffix.lower() == ".pdf":
        ids = [int(x) for x in args.pages.split(",") if x.strip()]
        pages = load_pdf_pages(args.input, dpi=args.dpi, page_ids=ids)
    else:
        pages = [(0, fetch_image(args.input))]
    return [(no, fetch_image(img, max_pixels=args.max_pixels)) for no, img in pages]


def image_data_url(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------
# vLLM side
# --------------------------------------------------------------------------

def vllm_once(args, image_url, prompt):
    """One streaming completion. Returns (ttft, decode_tps, tokens, text)."""
    import httpx

    body = {
        "model": args.vllm_model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            # the leading image tags are what the upstream vLLM example sends;
            # without them vLLM v1 inserts a newline in their place
            {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
        ]}],
        "max_completion_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = last_token_at = None
    tokens = 0
    chunks = []
    usage = None
    with httpx.Client(timeout=600.0) as client:
        with client.stream("POST", f"{args.vllm_url}/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if not piece:
                        continue
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    tokens += 1
                    chunks.append(piece)
    wall = time.perf_counter() - started
    if usage and usage.get("completion_tokens"):
        tokens = usage["completion_tokens"]
    decode_tps = None
    if first_token_at and last_token_at and tokens > 1:
        span = last_token_at - first_token_at
        if span > 0:
            decode_tps = (tokens - 1) / span
    return {
        "ttft_seconds": round(first_token_at - started, 3) if first_token_at else None,
        "decode_tps": round(decode_tps, 1) if decode_tps else None,
        "generated_tokens": tokens,
        "wall_seconds": round(wall, 3),
        "response": "".join(chunks),
        "usage": usage,
    }


def vllm_concurrent(args, image_urls, prompt, n):
    """Fire n requests at once — measures continuous batching.

    Each request gets a DIFFERENT page. Identical requests would share vLLM's
    prefix cache, so the run would measure cache hits instead of prefill.
    """
    results = [None] * n
    threads = []
    started = time.perf_counter()

    def worker(index):
        try:
            results[index] = vllm_once(args, image_urls[index % len(image_urls)], prompt)
        except Exception as error:  # noqa: BLE001
            results[index] = {"error": f"{type(error).__name__}: {error}"}

    for index in range(n):
        thread = threading.Thread(target=worker, args=(index,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - started

    ok = [r for r in results if r and not r.get("error")]
    total_tokens = sum(r["generated_tokens"] for r in ok)
    return {
        "concurrency": n,
        "wall_seconds": round(wall, 3),
        "requests_ok": len(ok),
        "total_generated_tokens": total_tokens,
        "aggregate_tps": round(total_tokens / wall, 1) if wall else None,
        "per_request_tps": round(statistics.mean([r["decode_tps"] for r in ok if r["decode_tps"]]), 1)
                           if any(r["decode_tps"] for r in ok) else None,
        "mean_ttft": round(statistics.mean([r["ttft_seconds"] for r in ok if r["ttft_seconds"]]), 3)
                     if any(r["ttft_seconds"] for r in ok) else None,
        "errors": [r["error"] for r in results if r and r.get("error")][:3],
    }


# --------------------------------------------------------------------------
# local (this repository) side
# --------------------------------------------------------------------------

def local_once(parser, image, prompt):
    from dots_mocr.utils.generation_stats import GenerationStats

    stats = GenerationStats()
    response = parser._inference(image, prompt, temperature=0.0, stats=stats)
    return {**stats.to_dict(), "response": response}


def main():
    args = build_arg_parser().parse_args()
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt

    prompt = dict_promptmode_to_prompt[args.prompt]
    pages = render_pages(args)
    report = {"config": vars(args), "pages": [], "vllm_concurrency": []}

    # ---- vLLM ------------------------------------------------------------
    print("=== vLLM ===", flush=True)
    image_urls = [image_data_url(image) for _, image in pages]
    for (page_no, image), image_url in zip(pages, image_urls):
        # warm on a scaled-down variant so the engine is hot but the prefix cache
        # holds nothing that the measured request could reuse
        from dots_mocr.utils.image_utils import fetch_image as _fetch
        vllm_once(args, image_data_url(_fetch(image, max_pixels=300_000)), prompt)
        result = vllm_once(args, image_url, prompt)
        result.update(page_no=page_no, engine="vllm")
        report["pages"].append(result)
        print(f"  page {page_no}: TTFT {result['ttft_seconds']}s, "
              f"{result['generated_tokens']} tok, {result['decode_tps']} tok/s, "
              f"{result['wall_seconds']}s", flush=True)

    for n in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        if n <= 1:
            continue
        summary = vllm_concurrent(args, image_urls, prompt, n)
        report["vllm_concurrency"].append(summary)
        print(f"  concurrency {n}: {summary['aggregate_tps']} tok/s aggregate, "
              f"{summary['per_request_tps']} per request, "
              f"wall {summary['wall_seconds']}s", flush=True)

    # ---- this repository -------------------------------------------------
    if not args.skip_local:
        print("\n=== transformers (this repo) ===", flush=True)
        from dots_mocr.cli import DotsMOCRParser

        local = DotsMOCRParser(
            ckpt=args.ckpt, device="cuda:0", dtype="bfloat16", temperature=0.0,
            max_completion_tokens=args.max_new_tokens, dpi=args.dpi,
            max_pixels=args.max_pixels, num_thread=1)
        for page_no, image in pages:
            local_once(local, image, prompt)         # warm
            result = local_once(local, image, prompt)
            result.update(page_no=page_no, engine="transformers")
            report["pages"].append(result)
            print(f"  page {page_no}: TTFT {result['ttft_seconds']}s, "
                  f"{result['generated_tokens']} tok, {result['decode_tps']} tok/s, "
                  f"{result['wall_seconds']}s", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bench_vllm.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {output_dir / 'bench_vllm.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
