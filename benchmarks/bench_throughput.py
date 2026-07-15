#!/usr/bin/env python3
"""dots.mocr throughput benchmark for one PDF on 1..N GPUs.

The 3B model fits on a single 24GB card, so the maximum-throughput setup is
data parallel: one worker process per GPU (pinned via CUDA_VISIBLE_DEVICES),
pages distributed round-robin. The parent process samples nvidia-smi while
the workers run and merges their per-page metrics into one JSON report.

Measured per page: input/output tokens, generate seconds, tokens/s,
JSON validity of the layout output. Per worker: model load time, TTFT
(prefill, max_new_tokens=1). Aggregate: wall seconds, sec/page, pages/min,
aggregate output tokens/s, GPU utilization/power statistics.

Examples:

    # single page, one GPU
    python3 -m benchmarks.bench_throughput --ckpt $CKPT --pdf doc.pdf \
        --gpus 0 --pages 0 --output reports/bench_1page.json

    # full PDF on both 4090s
    python3 -m benchmarks.bench_throughput --ckpt $CKPT --pdf doc.pdf \
        --gpus 0,1 --output reports/bench_2gpu.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--gpus", default="0", help="comma separated GPU ids, e.g. 0,1")
    parser.add_argument("--pages", default="all", help="'all' or comma separated 0-based pages")
    parser.add_argument("--prompt", default="prompt_layout_all_en")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="reports/bench_report.json")
    parser.add_argument("--save-outputs", default=None,
                        help="directory to save raw model responses per page")
    parser.add_argument("--label", default="", help="free-form label stored in the report")
    # internal
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pages", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default="", help=argparse.SUPPRESS)
    return parser


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

def run_worker(args):
    import torch
    from qwen_vl_utils import process_vision_info

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt

    page_ids = [int(x) for x in args.worker_pages.split(",") if x != ""]
    result = {
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pages": [],
    }

    t0 = time.perf_counter()
    parser = DotsMOCRParser(
        ckpt=args.ckpt,
        device="cuda",
        dtype="bfloat16",
        temperature=0.0,
        max_completion_tokens=args.max_new_tokens,
        dpi=args.dpi,
        max_pixels=args.max_pixels,
        num_thread=1,
    )
    torch.cuda.synchronize()
    result["model_load_seconds"] = time.perf_counter() - t0

    processor = parser.processor
    model = parser.model
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prompt = dict_promptmode_to_prompt[args.prompt]

    def preprocess(images):
        messages_batch = [
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            for image in images
        ]
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        image_inputs = []
        for messages in messages_batch:
            batch_images, _ = process_vision_info(messages)
            image_inputs.extend(batch_images)
        inputs = processor(
            text=texts, images=image_inputs, padding=True, return_tensors="pt",
        )
        inputs.pop("mm_token_type_ids", None)
        return inputs.to(parser.device)

    def generate(inputs, max_new_tokens):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        torch.cuda.synchronize()
        return out, time.perf_counter() - start

    t0 = time.perf_counter()
    rendered = load_pdf_pages(args.pdf, dpi=args.dpi, page_ids=page_ids)
    result["render_seconds"] = time.perf_counter() - t0
    images = [
        (page_id, fetch_image(image, max_pixels=args.max_pixels))
        for page_id, image in rendered
    ]

    # warmup: excluded from all metrics (cudnn autotune, memory pools, clocks)
    warm = preprocess([images[0][1]])
    generate(warm, 8)

    # prefill latency: one full prompt, a single new token
    ttft_inputs = preprocess([images[0][1]])
    _, ttft_seconds = generate(ttft_inputs, 1)
    result["ttft_seconds"] = ttft_seconds
    result["ttft_input_tokens"] = int(ttft_inputs.attention_mask.sum().item())

    save_dir = Path(args.save_outputs) if args.save_outputs else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    result["work_start_time"] = time.time()
    work_start = time.perf_counter()
    for chunk_start in range(0, len(images), args.batch_size):
        chunk = images[chunk_start:chunk_start + args.batch_size]
        prep_start = time.perf_counter()
        inputs = preprocess([image for _, image in chunk])
        prep_seconds = time.perf_counter() - prep_start

        out, gen_seconds = generate(inputs, args.max_new_tokens)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        responses = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        input_tokens_rows = inputs.attention_mask.sum(dim=1).tolist()
        output_tokens_rows = (trimmed != pad_id).sum(dim=1).tolist()
        batch_tps = float(sum(output_tokens_rows)) / gen_seconds if gen_seconds > 0 else None

        for row, (page_id, image) in enumerate(chunk):
            response = responses[row]
            valid_json = None
            num_cells = None
            if args.prompt in ("prompt_layout_all_en", "prompt_layout_only_en", "prompt_web_parsing"):
                try:
                    cells = json.loads(response)
                    valid_json = isinstance(cells, list) and len(cells) > 0
                    num_cells = len(cells) if isinstance(cells, list) else None
                except (json.JSONDecodeError, ValueError):
                    valid_json = False
            if save_dir:
                (save_dir / f"page_{page_id:03d}.txt").write_text(response, encoding="utf-8")
            result["pages"].append({
                "page_no": page_id,
                "image_width": image.width,
                "image_height": image.height,
                "batch_size": len(chunk),
                "preprocess_seconds": prep_seconds,
                # true latency: every page in a batch waits for the whole batch
                "generate_seconds": gen_seconds,
                "input_tokens": int(input_tokens_rows[row]),
                "output_tokens": int(output_tokens_rows[row]),
                # per-page rate is only well-defined for batch_size 1
                "output_tokens_per_second": (
                    float(output_tokens_rows[row]) / gen_seconds
                    if gen_seconds > 0 and len(chunk) == 1 else None
                ),
                "batch_output_tokens_per_second": batch_tps,
                "valid_json": valid_json,
                "num_cells": num_cells,
                "response_chars": len(response),
            })
        done = len(result["pages"])
        print(f"[worker gpu={result['visible_devices']}] {done}/{len(images)} pages", flush=True)

    result["work_seconds"] = time.perf_counter() - work_start
    result["work_end_time"] = time.time()
    result["cuda_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    result["device_name"] = torch.cuda.get_device_name(0)

    Path(args.worker_result).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[worker gpu={result['visible_devices']}] done", flush=True)


# --------------------------------------------------------------------------
# GPU sampler
# --------------------------------------------------------------------------

class GpuSampler(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop_event = threading.Event()

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except ValueError:  # nvidia-smi prints "[N/A]" for unsupported fields
            return None

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                )
            except (subprocess.SubprocessError, OSError):
                self._stop_event.wait(self.interval)
                continue
            stamp = time.time()
            for line in out.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 4:
                    continue
                idx = self._to_float(parts[0])
                if idx is None:
                    continue
                self.samples.append({
                    "time": stamp,
                    "gpu": int(idx),
                    "util_pct": self._to_float(parts[1]),
                    "memory_used_mb": self._to_float(parts[2]),
                    "power_w": self._to_float(parts[3]),
                })
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()

    def stats(self, window_start=None, window_end=None):
        stats = {}
        for sample in self.samples:
            if window_start and sample["time"] < window_start:
                continue
            if window_end and sample["time"] > window_end:
                continue
            entry = stats.setdefault(sample["gpu"], {"util": [], "mem": [], "power": []})
            for key, field in (("util", "util_pct"), ("mem", "memory_used_mb"), ("power", "power_w")):
                if sample[field] is not None:
                    entry[key].append(sample[field])

        def summary(values, precision=1):
            if not values:
                return {"mean": None, "max": None, "samples": 0}
            return {
                "mean": round(statistics.fmean(values), precision),
                "max": max(values),
                "samples": len(values),
            }

        return {
            gpu: {
                "util_pct": summary(entry["util"]),
                "memory_used_mb": summary(entry["mem"], 0),
                "power_w": summary(entry["power"]),
            }
            for gpu, entry in stats.items()
        }


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------

def resolve_pages(args):
    if args.pages != "all":
        return [int(x) for x in args.pages.split(",")]
    import fitz

    with fitz.open(args.pdf) as doc:
        return list(range(doc.page_count))


def run_parent(args):
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    pages = resolve_pages(args)
    assignments = {gpu: pages[i::len(gpu_ids)] for i, gpu in enumerate(gpu_ids)}
    assignments = {gpu: chunk for gpu, chunk in assignments.items() if chunk}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = output_path.parent / (output_path.stem + "_workers")
    scratch.mkdir(parents=True, exist_ok=True)

    sampler = GpuSampler()
    sampler.start()

    processes = {}
    wall_start = time.perf_counter()
    start_stamp = time.time()
    for gpu, gpu_pages in assignments.items():
        worker_result = scratch / f"worker_gpu{gpu}.json"
        cmd = [
            sys.executable, "-m", "benchmarks.bench_throughput",
            "--worker",
            "--ckpt", args.ckpt,
            "--pdf", args.pdf,
            "--prompt", args.prompt,
            "--dpi", str(args.dpi),
            "--max-new-tokens", str(args.max_new_tokens),
            "--batch-size", str(args.batch_size),
            "--worker-pages", ",".join(map(str, gpu_pages)),
            "--worker-result", str(worker_result),
        ]
        if args.max_pixels:
            cmd += ["--max-pixels", str(args.max_pixels)]
        if args.save_outputs:
            cmd += ["--save-outputs", str(Path(args.save_outputs) / f"gpu{gpu}")]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(scratch / f"worker_gpu{gpu}.log", "w")
        processes[gpu] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), worker_result, log)

    failures = []
    for gpu, (proc, _, log) in processes.items():
        code = proc.wait()
        log.close()
        if code != 0:
            failures.append((gpu, code))
    wall_seconds = time.perf_counter() - wall_start
    sampler.stop()
    sampler.join(timeout=5)

    if failures:
        for gpu, code in failures:
            print(f"worker on GPU {gpu} failed with code {code}, "
                  f"see {scratch}/worker_gpu{gpu}.log", file=sys.stderr)
        raise SystemExit(1)

    workers = {}
    all_pages = []
    for gpu, (_, worker_result, _) in processes.items():
        data = json.loads(worker_result.read_text())
        workers[gpu] = data
        for page in data["pages"]:
            page["gpu"] = gpu
            all_pages.append(page)
    all_pages.sort(key=lambda p: p["page_no"])

    gen_seconds = [p["generate_seconds"] for p in all_pages]
    output_tokens = sum(p["output_tokens"] for p in all_pages)
    work_seconds = max(w["work_seconds"] for w in workers.values())
    valid_flags = [p["valid_json"] for p in all_pages if p["valid_json"] is not None]
    # GPU stats over the actual inference window only (model load excluded)
    work_window_start = min(w["work_start_time"] for w in workers.values())
    work_window_end = max(w["work_end_time"] for w in workers.values())

    report = {
        "label": args.label,
        "config": {
            "ckpt": args.ckpt,
            "pdf": args.pdf,
            "prompt": args.prompt,
            "dpi": args.dpi,
            "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels,
            "batch_size": args.batch_size,
            "gpus": gpu_ids,
            "pages": pages,
        },
        "aggregate": {
            "num_pages": len(all_pages),
            "num_gpus": len(assignments),
            "wall_seconds": round(wall_seconds, 2),
            "work_seconds_max": round(work_seconds, 2),
            "seconds_per_page_wall": round(wall_seconds / len(all_pages), 2),
            "seconds_per_page_work": round(work_seconds / len(all_pages), 2),
            "seconds_per_page_latency_mean": round(statistics.fmean(gen_seconds), 2),
            "seconds_per_page_latency_median": round(statistics.median(gen_seconds), 2),
            "pages_per_minute": round(60 * len(all_pages) / work_seconds, 2),
            "output_tokens_total": output_tokens,
            "output_tokens_per_second_aggregate": round(output_tokens / work_seconds, 2),
            "ttft_seconds_by_gpu": {
                gpu: round(w["ttft_seconds"], 3) for gpu, w in workers.items()
            },
            "model_load_seconds_by_gpu": {
                gpu: round(w["model_load_seconds"], 1) for gpu, w in workers.items()
            },
            "valid_json_pages": sum(1 for flag in valid_flags if flag),
            "checked_json_pages": len(valid_flags),
            "gpu_stats_inference_window": sampler.stats(
                window_start=work_window_start, window_end=work_window_end,
            ),
            "gpu_stats_full_run": sampler.stats(window_start=start_stamp),
        },
        "workers": workers,
        "pages_detail": all_pages,
    }

    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    agg = report["aggregate"]
    print(json.dumps({"label": args.label, **agg}, indent=2))
    print(f"\nreport written to {output_path}")


def main():
    args = build_arg_parser().parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
