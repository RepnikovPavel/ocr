#!/usr/bin/env python3
"""Do all four attention backends give the same answer?

Backends: sdpa, flash_attention_2, flex_attention (this repo's vision tower) and
vLLM (a separate engine entirely). Each runs in the environment it needs — the
local venv, the project container, the vLLM container — so this script does one
backend per invocation and a final pass that compares whatever has been collected.

There is no golden file to compare against: the upstream project publishes only
rendered PNGs of its showcase results, no machine-readable expected output, and no
statement about determinism. So the reference here is `flash_attention_2`, because
that is what the checkpoint's own config.json names, and the real assertion is
mutual agreement rather than agreement with an authority.

    # one backend at a time, into a shared directory
    python3 -m benchmarks.agreement_matrix collect --backend sdpa --ckpt $CKPT \
        --images img1.jpg img2.png --out reports/agreement
    python3 -m benchmarks.agreement_matrix collect --backend vllm \
        --vllm-url http://127.0.0.1:8000/v1 --images ... --out reports/agreement

    # then the matrix
    python3 -m benchmarks.agreement_matrix compare --out reports/agreement
"""

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

TRANSFORMERS_BACKENDS = ("sdpa", "flash_attention_2", "flex_attention", "eager")
REFERENCE = "flash_attention_2"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="run one backend over the images")
    collect.add_argument("--backend", required=True,
                         choices=TRANSFORMERS_BACKENDS + ("vllm",))
    collect.add_argument("--ckpt", help="required for the transformers backends")
    collect.add_argument("--images", nargs="+", required=True)
    collect.add_argument("--prompt", default="prompt_layout_all_en")
    collect.add_argument("--max-pixels", type=int, default=1_000_000,
                         help="must be identical across backends; sdpa OOMs on 12 GiB above ~1.5 Mpx")
    collect.add_argument("--max-new-tokens", type=int, default=6144)
    collect.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    collect.add_argument("--vllm-model", default="rednote-hilab/dots.mocr")
    collect.add_argument("--label", default="", help="suffix to keep two runs of one backend apart")
    collect.add_argument("--out", default="reports/agreement")

    compare = sub.add_parser("compare", help="build the agreement matrix from collected runs")
    compare.add_argument("--out", default="reports/agreement")
    compare.add_argument("--reference", default=REFERENCE)
    return parser


def load_images(paths, max_pixels):
    from dots_mocr.utils.image_utils import fetch_image

    return [(Path(p).name, fetch_image(p, max_pixels=max_pixels)) for p in paths]


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def collect(args):
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt

    prompt = dict_promptmode_to_prompt[args.prompt]
    images = load_images(args.images, args.max_pixels)
    results = {}

    if args.backend == "vllm":
        from benchmarks.bench_vllm import image_data_url, vllm_once

        class Shim:
            vllm_url, vllm_model = args.vllm_url, args.vllm_model
            max_new_tokens = args.max_new_tokens
        for name, image in images:
            out = vllm_once(Shim, image_data_url(image), prompt)
            results[name] = {"response": out["response"],
                             "generated_tokens": out["generated_tokens"]}
            print(f"  {name}: {out['generated_tokens']} tokens", flush=True)
    else:
        from dots_mocr.cli import DotsMOCRParser

        parser = DotsMOCRParser(
            ckpt=args.ckpt, device="cuda:0", dtype="bfloat16", temperature=0.0,
            max_completion_tokens=args.max_new_tokens, max_pixels=args.max_pixels,
            num_thread=1, attn_implementation=args.backend)
        # what actually ran, not what was asked for — the parser may demote
        effective = parser.attn_implementation
        if effective != args.backend:
            raise SystemExit(f"asked for {args.backend} but the parser resolved {effective}; "
                             "the run would be mislabelled")
        from dots_mocr.utils.generation_stats import GenerationStats

        for name, image in images:
            stats = GenerationStats()
            response = parser._inference(image, prompt, temperature=0.0, stats=stats)
            results[name] = {"response": response,
                             "generated_tokens": stats.generated_tokens}
            print(f"  {name}: {stats.generated_tokens} tokens", flush=True)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = args.backend + (f"__{args.label}" if args.label else "")
    payload = {
        "backend": args.backend,
        "label": args.label,
        "config": {"prompt": args.prompt, "max_pixels": args.max_pixels,
                   "max_new_tokens": args.max_new_tokens},
        "environment": environment(),
        "responses": results,
    }
    path = output_dir / f"run_{key}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def environment():
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------

def compare(args):
    from benchmarks.bench_attention import semantic_match

    output_dir = Path(args.out)
    runs = {}
    for path in sorted(output_dir.glob("run_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = payload["backend"] + (f"__{payload['label']}" if payload.get("label") else "")
        runs[key] = payload
    if len(runs) < 2:
        raise SystemExit(f"need at least two collected runs in {output_dir}, found {len(runs)}")

    images = sorted(set.intersection(*(set(r["responses"]) for r in runs.values())))
    print(f"backends: {', '.join(runs)}")
    print(f"images:   {', '.join(images)}\n")

    def unpack(run, image):
        """Older runs stored a bare string; newer ones a dict with the token count."""
        entry = run["responses"][image]
        if isinstance(entry, str):
            return entry, None
        return entry["response"], entry.get("generated_tokens")

    pairs = {}
    truncated = set()
    for key, run in runs.items():
        cap = run["config"]["max_new_tokens"]
        for image in images:
            _, tokens = unpack(run, image)
            # a response cut off at the cap is not a disagreement, it is a
            # measurement that never finished — counting it as one manufactures
            # divergence out of an under-sized token budget
            if tokens is not None and tokens >= cap:
                truncated.add(image)

    for a, b in combinations(sorted(runs), 2):
        verdicts = {}
        for image in images:
            text_a, _ = unpack(runs[a], image)
            text_b, _ = unpack(runs[b], image)
            if image in truncated:
                verdicts[image] = {"equivalent": None, "identical": text_a == text_b,
                                   "detail": "inconclusive: output hit max_new_tokens"}
                continue
            ok, why = semantic_match(text_a, text_b)
            verdicts[image] = {"equivalent": ok, "identical": text_a == text_b, "detail": why}
        pairs[f"{a} vs {b}"] = verdicts

    if truncated:
        print(f"truncated at max_new_tokens, excluded from the verdict: "
              f"{', '.join(sorted(truncated))}\n")

    width = max(len(k) for k in pairs) + 2
    print(f"{'pair':{width}} {'equivalent':>12} {'identical':>10}   differing images")
    for pair, verdicts in pairs.items():
        judged = [v for v in verdicts.values() if v["equivalent"] is not None]
        eq = sum(1 for v in judged if v["equivalent"])
        ident = sum(1 for v in verdicts.values() if v["identical"])
        bad = [i for i, v in verdicts.items() if v["equivalent"] is False]
        print(f"{pair:{width}} {eq:>7}/{len(judged):<4} {ident:>7}/{len(images):<3}   "
              f"{', '.join(bad) if bad else '—'}")

    reference = args.reference
    summary = {"pairs": pairs, "reference": reference, "images": images,
               "truncated_excluded": sorted(truncated),
               "backends": {k: v["environment"] for k, v in runs.items()}}
    if reference in runs:
        print(f"\nagainst the reference ({reference}, the backend named in the checkpoint config):")
        for key in sorted(runs):
            if key == reference:
                continue
            verdicts = pairs.get(f"{min(key, reference)} vs {max(key, reference)}", {})
            judged = [v for v in verdicts.values() if v["equivalent"] is not None]
            eq = sum(1 for v in judged if v["equivalent"])
            print(f"  {key:22} {eq}/{len(judged)} images equivalent")
    else:
        print(f"\nreference {reference} was not collected — matrix only")

    path = output_dir / "agreement.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    disagreeing = [p for p, v in pairs.items() if any(x["equivalent"] is False for x in v.values())]
    return 1 if disagreeing else 0


def main():
    args = build_arg_parser().parse_args()
    return collect(args) if args.command == "collect" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
