#!/usr/bin/env python3
"""Where does a page actually go? Per-stage, per-operator profile of dots.mocr.

Answers, on a real document:
  * which stage owns the wall clock (preprocess / vision tower / prefill / decode);
  * which operators inside each stage own it, ranked;
  * how many FLOPs each stage really executes, and what fraction of the card's
    compute and memory bandwidth that is (the roofline verdict: is this op limited
    by arithmetic or by moving bytes?);
  * whether the hot shapes are static across pages and decode steps, which decides
    whether CUDA graphs / static kernels are even applicable.

Wall-clock numbers are taken WITHOUT the profiler attached (it distorts them);
the profiler is then run separately for the operator breakdown.

    python3 -m benchmarks.profile_stages --ckpt $CKPT --input doc.pdf --pages 0,1 \
        --output-dir reports/profile
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

BYTES_PER_ELEMENT = 2  # bf16 everywhere in this model


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--pages", default="0,1")
    parser.add_argument("--prompt", default="prompt_layout_all_en")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pixels", type=int, default=2_200_000)
    parser.add_argument("--decode-steps", type=int, default=64,
                        help="decode steps to profile (the real page decodes ~1300)")
    parser.add_argument("--attn", default="flex_attention")
    parser.add_argument("--llm-attn", default="sdpa")
    parser.add_argument("--output-dir", default="reports/profile")
    return parser


# --------------------------------------------------------------------------
# device capability
# --------------------------------------------------------------------------

def device_roofline(device):
    import torch

    props = torch.cuda.get_device_properties(device)
    # GDDR: bus width x memory clock x 2 (DDR)
    bandwidth = props.memory_bus_width / 8 * props.memory_clock_rate * 1e3 * 2 / 1e9
    # Ada (sm_89) bf16 tensor core: 256 MAC/cycle/SM -> 512 FLOP/cycle/SM
    peak_bf16 = props.multi_processor_count * 512 * props.clock_rate * 1e3 / 1e12
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "peak_bandwidth_gbs": round(bandwidth, 1),
        "peak_bf16_tflops": round(peak_bf16, 1),
        "total_memory_gib": round(props.total_memory / 2**30, 2),
    }


# --------------------------------------------------------------------------
# analytical FLOP / byte model (cross-check on the profiler)
# --------------------------------------------------------------------------

def analytic_costs(config, vision_tokens, prompt_tokens, kv_len):
    """FLOPs and weight bytes per stage, derived from the geometry alone."""
    v = config.vision_config
    vd, vl, vh = v.embed_dim, v.num_hidden_layers, v.num_attention_heads
    v_int = v.intermediate_size
    hd = config.hidden_size
    ll, lh, lkv = config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads
    l_int, vocab = config.intermediate_size, config.vocab_size
    head_dim = hd // lh

    def gemm(m, n, k):
        return 2 * m * n * k

    S = vision_tokens
    vision_per_layer = (
        gemm(S, 3 * vd, vd)          # qkv
        + 2 * 2 * S * S * (vd // vh) * vh  # attention scores + context
        + gemm(S, vd, vd)            # proj
        + 2 * gemm(S, v_int, vd)     # fc1 + fc3 (SwiGLU up)
        + gemm(S, vd, v_int)         # fc2 (down)
    )
    vision_flops = vision_per_layer * vl

    def lm_layer_flops(q_len, kv):
        q_proj = gemm(q_len, hd, hd)
        kv_proj = 2 * gemm(q_len, lkv * head_dim, hd)
        o_proj = gemm(q_len, hd, hd)
        attn = 2 * 2 * q_len * kv * head_dim * lh
        mlp = 2 * gemm(q_len, l_int, hd) + gemm(q_len, hd, l_int)
        return q_proj + kv_proj + o_proj + attn + mlp

    prefill_flops = lm_layer_flops(prompt_tokens, prompt_tokens) * ll + gemm(1, vocab, hd)
    decode_flops = lm_layer_flops(1, kv_len) * ll + gemm(1, vocab, hd)

    lm_weights = (
        ll * (hd * hd + 2 * lkv * head_dim * hd + hd * hd + 3 * hd * l_int)
        + vocab * hd            # lm_head
    ) * BYTES_PER_ELEMENT
    vision_weights = vl * (3 * vd * vd + vd * vd + 3 * vd * v_int) * BYTES_PER_ELEMENT
    kv_bytes = ll * 2 * lkv * head_dim * kv_len * BYTES_PER_ELEMENT

    return {
        "vision_tflops": vision_flops / 1e12,
        "prefill_tflops": prefill_flops / 1e12,
        "decode_tflops_per_token": decode_flops / 1e12,
        "lm_weight_bytes": lm_weights,
        "vision_weight_bytes": vision_weights,
        "kv_bytes_at_ctx": kv_bytes,
    }


# --------------------------------------------------------------------------
# operator profiling
# --------------------------------------------------------------------------

def shape_bytes(shapes):
    total = 0
    for shape in shapes or []:
        if not shape:
            continue
        n = 1
        for dim in shape:
            n *= max(int(dim), 1)
        total += n * BYTES_PER_ELEMENT
    return total


def profile_region(fn, warmup=2, active=3):
    """Run fn under the profiler and aggregate CUDA time / FLOPs / shapes per op."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_flops=True) as prof:
        for _ in range(active):
            fn()
        torch.cuda.synchronize()

    ops = defaultdict(lambda: {"cuda_us": 0.0, "count": 0, "flops": 0.0,
                               "bytes": 0.0, "shapes": set()})
    for event in prof.key_averages(group_by_input_shape=True):
        if event.self_device_time_total <= 0:
            continue
        entry = ops[event.key]
        entry["cuda_us"] += event.self_device_time_total
        entry["count"] += event.count
        entry["flops"] += (event.flops or 0)
        entry["bytes"] += shape_bytes(event.input_shapes) * event.count
        if event.input_shapes:
            entry["shapes"].add(str(event.input_shapes)[:110])

    for entry in ops.values():
        entry["cuda_us"] /= active
        entry["count"] = entry["count"] // active
        entry["flops"] /= active
        entry["bytes"] /= active
        entry["shapes"] = sorted(entry["shapes"])[:3]
    return dict(ops)


def rank_ops(ops, roofline, limit=14):
    rows = []
    total_us = sum(o["cuda_us"] for o in ops.values()) or 1.0
    for name, o in sorted(ops.items(), key=lambda kv: -kv[1]["cuda_us"])[:limit]:
        seconds = o["cuda_us"] / 1e6
        tflops = (o["flops"] / seconds / 1e12) if (seconds > 0 and o["flops"]) else None
        gbs = (o["bytes"] / seconds / 1e9) if seconds > 0 else None
        verdict = "-"
        if tflops and gbs:
            compute_util = tflops / roofline["peak_bf16_tflops"]
            memory_util = gbs / roofline["peak_bandwidth_gbs"]
            verdict = "compute" if compute_util > memory_util else "memory"
            verdict += f" ({max(compute_util, memory_util) * 100:.0f}%)"
        elif gbs:
            verdict = f"memory ({gbs / roofline['peak_bandwidth_gbs'] * 100:.0f}%)"
        rows.append({
            "op": name,
            "ms": round(o["cuda_us"] / 1000, 3),
            "pct": round(100 * o["cuda_us"] / total_us, 1),
            "calls": o["count"],
            "gflop": round(o["flops"] / 1e9, 2) if o["flops"] else None,
            "tflops": round(tflops, 1) if tflops else None,
            "gbs": round(gbs) if gbs else None,
            "bound": verdict,
            "shapes": o["shapes"][:1],
        })
    return rows, total_us / 1000


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    import torch
    from qwen_vl_utils import process_vision_info

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.image_utils import fetch_image
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt

    roofline = device_roofline(args.device)
    print(json.dumps(roofline, indent=1))

    parser = DotsMOCRParser(
        ckpt=args.ckpt, device=args.device, dtype=args.dtype, temperature=0.0,
        max_completion_tokens=2048, dpi=args.dpi, max_pixels=args.max_pixels,
        num_thread=1, attn_implementation=args.attn,
        llm_attn_implementation=args.llm_attn,
    )
    model, processor = parser.model, parser.processor
    prompt = dict_promptmode_to_prompt[args.prompt]
    page_ids = [int(x) for x in args.pages.split(",") if x.strip()]
    pages = (load_pdf_pages(args.input, dpi=args.dpi, page_ids=page_ids)
             if Path(args.input).suffix.lower() == ".pdf" else [(0, fetch_image(args.input))])

    report = {"roofline": roofline, "config": vars(args), "pages": []}

    for page_no, origin in pages:
        print(f"\n{'=' * 70}\nPAGE {page_no}\n{'=' * 70}", flush=True)
        page = {"page_no": page_no}

        # ---- stage 1: preprocess (CPU) -------------------------------------
        started = time.perf_counter()
        image = fetch_image(origin, max_pixels=args.max_pixels)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
        inputs.pop("mm_token_type_ids", None)
        inputs = inputs.to(parser.device)
        torch.cuda.synchronize()
        page["preprocess_s"] = round(time.perf_counter() - started, 4)

        pixel_values, grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]
        input_ids = inputs["input_ids"]
        vision_tokens, prompt_tokens = pixel_values.shape[0], input_ids.shape[-1]
        page.update(vision_tokens=vision_tokens, prompt_tokens=prompt_tokens,
                    image_size=[image.width, image.height])
        print(f"vision tokens {vision_tokens}, prompt tokens {prompt_tokens}")

        tower = model.vision_tower

        # ---- stage 2: vision tower -----------------------------------------
        def run_vision():
            with torch.inference_mode():
                tower(pixel_values, grid_thw)

        run_vision()
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run_vision()
            torch.cuda.synchronize(); samples.append(time.perf_counter() - t0)
        page["vision_s"] = round(statistics.median(samples), 4)

        # ---- stage 3: prefill ----------------------------------------------
        def run_prefill():
            with torch.inference_mode():
                return model(input_ids=input_ids, pixel_values=pixel_values,
                             image_grid_thw=grid_thw,
                             attention_mask=inputs["attention_mask"], use_cache=True)

        out = run_prefill()
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            run_prefill()
            torch.cuda.synchronize(); samples.append(time.perf_counter() - t0)
        page["prefill_s"] = round(statistics.median(samples), 4)
        page["prefill_minus_vision_s"] = round(page["prefill_s"] - page["vision_s"], 4)

        # ---- stage 4: decode -------------------------------------------------
        past = out.past_key_values
        next_token = out.logits[:, -1:].argmax(-1)
        cache_len = prompt_tokens

        def one_step():
            nonlocal past
            with torch.inference_mode():
                step = model(input_ids=next_token, past_key_values=past, use_cache=True)
                return step

        for _ in range(4):
            one_step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.decode_steps):
            one_step()
        torch.cuda.synchronize()
        decode_total = time.perf_counter() - t0
        page["decode_ms_per_token"] = round(1000 * decode_total / args.decode_steps, 3)
        page["decode_tps"] = round(args.decode_steps / decode_total, 1)

        # ---- analytic model --------------------------------------------------
        page["analytic"] = analytic_costs(model.config, vision_tokens, prompt_tokens,
                                          cache_len)

        # ---- operator breakdown ---------------------------------------------
        page["ops"] = {}
        for stage, fn in (("vision", run_vision), ("prefill", run_prefill),
                          ("decode_step", one_step)):
            ops = profile_region(fn, warmup=2, active=3)
            rows, total_ms = rank_ops(ops, roofline)
            page["ops"][stage] = {"total_ms": round(total_ms, 3), "top": rows}
            print(f"\n--- {stage}: {total_ms:.2f} ms of CUDA work ---")
            for r in rows[:8]:
                print(f"  {r['pct']:5.1f}%  {r['ms']:8.3f} ms  x{r['calls']:<5} "
                      f"{r['op'][:44]:44} {str(r['tflops'] or '-'):>7} TF/s  "
                      f"{str(r['gbs'] or '-'):>6} GB/s  {r['bound']}")

        # a full page: decode dominates once you multiply by the real token count
        page["projected_page"] = {
            "decode_tokens_typical": 1300,
            "decode_s": round(1300 * page["decode_ms_per_token"] / 1000, 2),
        }
        report["pages"].append(page)

    # ---- shape dynamism ------------------------------------------------------
    report["shape_analysis"] = shape_analysis(report["pages"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile_stages.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {output_dir / 'profile_stages.json'}")
    return 0


def shape_analysis(pages):
    """Do the hot shapes change between pages? Decides CUDA-graph applicability."""
    out = {}
    for stage in ("vision", "prefill", "decode_step"):
        per_page = {}
        for page in pages:
            for row in page["ops"].get(stage, {}).get("top", [])[:6]:
                per_page.setdefault(row["op"], []).append(row["shapes"][0] if row["shapes"] else "")
        out[stage] = {
            op: {"static": len(set(v)) == 1, "variants": sorted(set(v))[:2]}
            for op, v in per_page.items()
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
