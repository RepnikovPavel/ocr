#!/usr/bin/env python3
"""Compare attention backends for dots.mocr on one machine, same inputs.

Answers two questions at once:

  1. Is flex_attention faster / lighter than sdpa here?
  2. Does it return the *same answer*? Every backend decodes greedily from the
     same pages, and the parent compares the decoded text against the reference
     backend. A speedup that changes the output is not a speedup.

Each backend runs in its own subprocess: torch.compile caches, cuDNN
autotuning and peak-memory counters must not leak between measurements.

    python3 -m benchmarks.bench_attention --ckpt $CKPT --input page.jpg \
        --attn sdpa,flex_attention --output-dir reports/flexattn

    # a few PDF pages, more decode steps for a stable tokens/s
    python3 -m benchmarks.bench_attention --ckpt $CKPT --input doc.pdf \
        --pages 0,1,2 --max-new-tokens 512 --output-dir reports/flexattn
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
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
    parser.add_argument("--input", required=True, help="image or pdf")
    parser.add_argument("--pages", default="0", help="0-based pages for a pdf, e.g. 0,1,2")
    parser.add_argument("--attn", default="sdpa,flex_attention",
                        help="comma separated backends; the first is the reference for output equality")
    parser.add_argument("--llm-attn", default="sdpa",
                        help="language model backend, held constant across the A/B so that "
                             "varying --attn moves only the vision tower")
    parser.add_argument("--prompt", default="prompt_layout_all_en")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1, help="timed passes per page (best is reported)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed passes before measuring; flex_attention compiles on first use")
    parser.add_argument("--output-dir", default="reports/flexattn")
    parser.add_argument("--label", default="")
    # internal
    parser.add_argument("--worker-attn", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    return parser


def load_pages(args):
    """Render the input into a list of (page_no, PIL image)."""
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image

    if Path(args.input).suffix.lower() == ".pdf":
        page_ids = [int(x) for x in args.pages.split(",") if x.strip() != ""]
        return load_pdf_pages(args.input, dpi=args.dpi, page_ids=page_ids)
    return [(0, fetch_image(args.input))]


# --------------------------------------------------------------------------
# worker: one backend, one process
# --------------------------------------------------------------------------

def run_worker(args):
    import torch

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.generation_stats import GenerationStats
    from dots_mocr.utils.image_utils import fetch_image, smart_resize
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt

    backend = args.worker_attn
    result = {"attn_implementation": backend}

    started = time.perf_counter()
    parser = DotsMOCRParser(
        ckpt=args.ckpt,
        device=args.device,
        dtype=args.dtype,
        temperature=0.0,          # greedy: the output must be reproducible
        top_p=1.0,
        max_completion_tokens=args.max_new_tokens,
        dpi=args.dpi,
        max_pixels=args.max_pixels,
        num_thread=1,
        attn_implementation=backend,
        llm_attn_implementation=args.llm_attn,
    )
    torch.cuda.synchronize()
    result["model_load_seconds"] = round(time.perf_counter() - started, 3)
    result["effective_attn_implementation"] = parser.attn_implementation
    result["llm_attn_implementation"] = parser.llm_attn_implementation
    result["vision_attn_class"] = type(parser.model.vision_tower.blocks[0].attn).__name__

    prompt = dict_promptmode_to_prompt[args.prompt]
    pages = []

    for page_no, origin_image in load_pages(args):
        image = fetch_image(origin_image, max_pixels=args.max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)

        # Isolate the vision tower: it is where the quadratic attention lives and
        # where a block-sparse backend is supposed to pay off. Timed separately
        # from the language model so a decode-side regression cannot hide here.
        vision_seconds = time_vision_tower(parser, image, prompt, args)

        torch.cuda.reset_peak_memory_stats()
        cold = None
        for _ in range(max(0, args.warmup)):
            cold_started = time.perf_counter()
            parser._inference(image, prompt, temperature=0.0)
            if cold is None:
                cold = round(time.perf_counter() - cold_started, 3)

        best = None
        for _ in range(max(1, args.repeat)):
            stats = GenerationStats()
            response = parser._inference(image, prompt, temperature=0.0, stats=stats)
            record = stats.to_dict()
            record["response"] = response
            if best is None or (record["wall_seconds"] or 1e9) < (best["wall_seconds"] or 1e9):
                best = record

        best.update({
            "page_no": page_no,
            "input_height": input_height,
            "input_width": input_width,
            "vision_seconds": vision_seconds,
            "first_call_seconds": cold,
            "response_sha256": hashlib.sha256(best["response"].encode("utf-8")).hexdigest(),
            "response_chars": len(best["response"]),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        })
        pages.append(best)

    result["pages"] = pages
    Path(args.worker_result).write_text(json.dumps(result), encoding="utf-8")


def time_vision_tower(parser, image, prompt, args):
    """Median seconds for one vision-tower forward on this page."""
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = parser.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = parser.processor(text=[text], images=image_inputs, videos=video_inputs,
                             padding=True, return_tensors="pt")
    inputs.pop("mm_token_type_ids", None)
    inputs = inputs.to(parser.device)

    tower = parser.model.vision_tower
    pixel_values, grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]
    samples = []
    with torch.inference_mode():
        for index in range(3 + max(0, args.warmup)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            tower(pixel_values, grid_thw)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if index >= max(0, args.warmup):  # drop compile/autotune passes
                samples.append(elapsed)
    return round(statistics.median(samples), 4) if samples else None


# --------------------------------------------------------------------------
# parent: run every backend, compare, report
# --------------------------------------------------------------------------

def environment():
    import torch

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        info["transformers"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gib"] = round(props.total_memory / 2**30, 1)
        info["gpu_sm"] = f"{props.major}.{props.minor}"
    return info


def run_backend(args, backend, result_path):
    command = [sys.executable, "-m", "benchmarks.bench_attention",
               "--ckpt", args.ckpt, "--input", args.input, "--pages", args.pages,
               "--prompt", args.prompt, "--device", args.device, "--dtype", args.dtype,
               "--dpi", str(args.dpi), "--max-pixels", str(args.max_pixels),
               "--max-new-tokens", str(args.max_new_tokens), "--repeat", str(args.repeat),
               "--warmup", str(args.warmup), "--llm-attn", args.llm_attn,
               "--worker-attn", backend, "--worker-result", str(result_path)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    print(f"\n=== {backend} ===", flush=True)
    completed = subprocess.run(command, env=env, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        return {"attn_implementation": backend, "error": f"worker exited {completed.returncode}"}
    return json.loads(result_path.read_text(encoding="utf-8"))


def aggregate(result):
    """Per-backend means across pages (pages are the repeat unit)."""
    pages = [p for p in result.get("pages", []) if p]
    if not pages:
        return {}

    def mean(key):
        values = [p[key] for p in pages if p.get(key) is not None]
        return round(statistics.mean(values), 3) if values else None

    return {
        "pages": len(pages),
        "generated_tokens": mean("generated_tokens"),
        "ttft_seconds": mean("ttft_seconds"),
        "decode_tps": mean("decode_tps"),
        "total_tps": mean("total_tps"),
        "wall_seconds": mean("wall_seconds"),
        "vision_seconds": mean("vision_seconds"),
        "first_call_seconds": mean("first_call_seconds"),
        "peak_memory_gib": max(p.get("peak_memory_gib") or 0 for p in pages),
    }


BBOX_TOLERANCE_PX = 4


def semantic_match(reference_text, candidate_text, bbox_tolerance=BBOX_TOLERANCE_PX):
    """Do two layout outputs say the same thing?

    Byte equality is the wrong bar in bf16, and not only for flex: measured on this
    checkpoint, `sdpa` itself produces two different answers for the same page under
    field-for-field identical configs, differing only in the torch build. Attention
    backends sum in different orders, 42 residual vision layers amplify that, and
    greedy decoding is an argmax — so any step whose top-1/top-2 margin is thinner
    than the perturbation is a coin flip.

    Mostly that surfaces as a +-1 px bbox coordinate, but it can also flip a token
    of text: on one page every backend disagrees about whether a LaTeX fragment
    carries absolute-value bars, and sdpa lands on both sides across builds. So a
    single differing token is not evidence that a backend is wrong.

    The regression criterion is therefore: same blocks, same categories, same text,
    and bboxes within a few pixels — with the understanding that an occasional text
    difference is a property of the model in bf16, not of the backend under test.
    Mask identity is what gets asserted exactly; see tests/test_vision_flex_attention.py.
    """
    if reference_text == candidate_text:
        return True, "identical"
    try:
        reference = json.loads(reference_text)
        candidate = json.loads(candidate_text)
    except (json.JSONDecodeError, TypeError):
        # Not layout JSON — either a text-mode prompt, or output truncated by
        # max_new_tokens mid-object. Report how far the two agree so a truncated
        # run is still interpretable instead of collapsing to a bare "differs".
        common = len(os.path.commonprefix([reference_text, candidate_text]))
        longest = max(len(reference_text), len(candidate_text), 1)
        return False, (f"not parseable as layout JSON (truncated?); "
                       f"common prefix {common}/{longest} chars ({100 * common / longest:.1f}%)")
    if not isinstance(reference, list) or not isinstance(candidate, list):
        return False, "output is not a layout list"
    if len(reference) != len(candidate):
        return False, f"block count {len(candidate)} != {len(reference)}"

    worst = 0
    for index, (want, got) in enumerate(zip(reference, candidate)):
        if want.get("category") != got.get("category"):
            return False, f"block {index}: category {got.get('category')} != {want.get('category')}"
        if (want.get("text") or "") != (got.get("text") or ""):
            return False, f"block {index}: text differs"
        want_bbox, got_bbox = want.get("bbox") or [], got.get("bbox") or []
        if len(want_bbox) != len(got_bbox):
            return False, f"block {index}: bbox arity differs"
        for a, b in zip(want_bbox, got_bbox):
            worst = max(worst, abs(a - b))
    if worst > bbox_tolerance:
        return False, f"bbox drift {worst}px > {bbox_tolerance}px"
    return True, f"same text/categories, max bbox drift {worst}px"


def compare_outputs(results, reference):
    """Per backend: does it say the same thing as the reference backend, page by page?"""
    ref_pages = {p["page_no"]: p for p in results.get(reference, {}).get("pages", [])}
    verdicts = {}
    for backend, result in results.items():
        pages = [p for p in (result.get("pages") or []) if p["page_no"] in ref_pages]
        if backend == reference:
            verdicts[backend] = {"identical": True, "equivalent": True,
                                 "mismatched_pages": [], "compared": len(pages), "notes": []}
            continue
        exact, mismatched, notes = [], [], []
        for page in pages:
            ref = ref_pages[page["page_no"]]
            exact.append(page["response_sha256"] == ref["response_sha256"])
            ok, why = semantic_match(ref["response"], page["response"])
            notes.append(f"p{page['page_no']}: {why}")
            if not ok:
                mismatched.append(page["page_no"])
        verdicts[backend] = {
            "identical": bool(pages) and all(exact),
            "equivalent": bool(pages) and not mismatched,
            "mismatched_pages": mismatched,
            "compared": len(pages),
            "notes": notes,
        }
    return verdicts


def render_markdown(report):
    reference = report["reference_backend"]
    env = report["environment"]
    rows = []
    for backend in report["backends"]:
        summary = report["summary"].get(backend, {})
        verdict = report["output_equality"].get(backend, {})
        if not summary:
            rows.append(f"| `{backend}` | " + " | ".join(["—"] * 7) + " | ошибка |")
            continue
        ref_summary = report["summary"].get(reference, {})

        def speedup(key, higher_is_better=True):
            mine, theirs = summary.get(key), ref_summary.get(key)
            if not mine or not theirs:
                return ""
            ratio = (mine / theirs) if higher_is_better else (theirs / mine)
            return f" ({ratio:.2f}x)" if backend != reference else ""

        if backend == reference:
            equality = "—"
        elif verdict.get("identical"):
            equality = "да (побайтово)"
        elif verdict.get("equivalent"):
            equality = "да (текст и категории; bbox ±px)"
        else:
            equality = f"**НЕТ** (стр. {verdict.get('mismatched_pages')})"
        rows.append(
            f"| `{backend}` "
            f"| {summary.get('vision_seconds')}{speedup('vision_seconds', False)} "
            f"| {summary.get('ttft_seconds')}{speedup('ttft_seconds', False)} "
            f"| {summary.get('decode_tps')}{speedup('decode_tps')} "
            f"| {summary.get('total_tps')}{speedup('total_tps')} "
            f"| {summary.get('wall_seconds')}{speedup('wall_seconds', False)} "
            f"| {summary.get('generated_tokens')} "
            f"| {summary.get('peak_memory_gib')} "
            f"| {equality} |"
        )

    return "\n".join([
        f"### {report['label'] or 'attention backends'}",
        "",
        f"- вход: `{report['config']['input']}` (страницы `{report['config']['pages']}`), "
        f"промпт `{report['config']['prompt']}`",
        f"- max_pixels {report['config']['max_pixels']:,}, max_new_tokens "
        f"{report['config']['max_new_tokens']}, dtype {report['config']['dtype']}, greedy (temperature 0)",
        f"- {env.get('gpu')} ({env.get('gpu_memory_gib')} GiB, sm {env.get('gpu_sm')}), "
        f"torch {env.get('torch')}+cu{env.get('cuda')}, transformers {env.get('transformers')}",
        f"- эталон для сравнения ответов: `{reference}`",
        "",
        "| backend | vision s | TTFT s | decode t/s | total t/s | стр. s | ток. | peak VRAM ГиБ | ответ == эталон |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
    ])


def main():
    args = build_arg_parser().parse_args()

    if args.worker_attn:
        return run_worker(args)

    backends = [x.strip() for x in args.attn.split(",") if x.strip()]
    if not backends:
        raise SystemExit("--attn must list at least one backend")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for backend in backends:
        result_path = output_dir / f"raw_{backend}.json"
        results[backend] = run_backend(args, backend, result_path)

    reference = backends[0]
    report = {
        "label": args.label,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment(),
        "config": {
            "ckpt": args.ckpt, "input": args.input, "pages": args.pages,
            "prompt": args.prompt, "device": args.device, "dtype": args.dtype,
            "dpi": args.dpi, "max_pixels": args.max_pixels,
            "max_new_tokens": args.max_new_tokens, "repeat": args.repeat, "warmup": args.warmup,
        },
        "backends": backends,
        "reference_backend": reference,
        "raw": results,
        "summary": {backend: aggregate(result) for backend, result in results.items()},
        "output_equality": compare_outputs(results, reference),
    }

    json_path = output_dir / "bench_attention.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    (output_dir / "bench_attention.md").write_text(markdown + "\n", encoding="utf-8")

    print("\n" + markdown)
    print(f"\nwrote {json_path} and {output_dir / 'bench_attention.md'}")

    for backend, verdict in report["output_equality"].items():
        for note in verdict.get("notes", []):
            print(f"  {backend}: {note}")

    mismatched = [b for b, v in report["output_equality"].items() if not v["equivalent"]]
    if mismatched:
        print(f"\nWARNING: output is not equivalent to {reference} for: {', '.join(mismatched)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
