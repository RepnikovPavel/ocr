#!/usr/bin/env python3
"""Token budget of one page: how many tokens the image becomes, and how many
tokens are then generated.

This is the accounting that explains the profile. A page enters as tens of
thousands of patch tokens, is compressed 4x by the patch merger before it reaches
the language model, and is answered with a comparatively small number of generated
tokens — but those generated tokens are the expensive ones, because each of them
costs a full pass over the model weights while the whole prompt is processed in a
single batched pass.

    python3 -m benchmarks.token_budget --ckpt $CKPT --input doc.pdf
"""

import argparse
import json
import os
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--resolutions", default="500000,1000000,1500000,2200000")
    parser.add_argument("--modes", default="prompt_layout_all_en,prompt_layout_only_en,prompt_ocr")
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--output-dir", default="reports/profile")
    return parser


def main():
    args = build_arg_parser().parse_args()
    import torch
    from qwen_vl_utils import process_vision_info

    from dots_mocr.cli import DotsMOCRParser
    from dots_mocr.utils.doc_utils import load_pdf_pages
    from dots_mocr.utils.generation_stats import GenerationStats
    from dots_mocr.utils.image_utils import fetch_image
    from dots_mocr.utils.prompts import dict_promptmode_to_prompt

    parser = DotsMOCRParser(
        ckpt=args.ckpt, device="cuda:0", dtype="bfloat16", temperature=0.0,
        max_completion_tokens=args.max_new_tokens, dpi=args.dpi, num_thread=1)
    config = parser.model.config
    vision_config = config.vision_config
    patch = vision_config.patch_size
    merge = vision_config.spatial_merge_size
    image_token_id = config.image_token_id

    origin = (load_pdf_pages(args.input, dpi=args.dpi, page_ids=[args.page])[0][1]
              if Path(args.input).suffix.lower() == ".pdf" else fetch_image(args.input))

    def prepare(image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = parser.processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = parser.processor(text=[text], images=image_inputs, videos=video_inputs,
                                  padding=True, return_tensors="pt")
        inputs.pop("mm_token_type_ids", None)
        return inputs.to(parser.device)

    report = {"geometry": {"patch_size": patch, "spatial_merge_size": merge}, "rows": []}
    layout_prompt = dict_promptmode_to_prompt["prompt_layout_all_en"]

    print(f"патч {patch}x{patch}, слияние {merge}x{merge} "
          f"-> один токен языковой модели = {patch*merge}x{patch*merge} пикселей\n")
    print(f"{'max_pixels':>11} {'рендер':>12} {'Мп':>5} {'патч-токенов':>13} "
          f"{'в языковую модель':>18} {'весь промпт':>12} {'сжатие':>8}")

    for target in [int(x) for x in args.resolutions.split(",")]:
        image = fetch_image(origin, max_pixels=target)
        inputs = prepare(image, layout_prompt)
        vision_tokens = inputs["pixel_values"].shape[0]
        prompt_tokens = inputs["input_ids"].shape[-1]
        lm_image_tokens = int((inputs["input_ids"] == image_token_id).sum())
        pixels = image.width * image.height
        report["rows"].append({
            "max_pixels": target, "render": [image.width, image.height],
            "megapixels": round(pixels / 1e6, 2), "vision_tokens": vision_tokens,
            "lm_image_tokens": lm_image_tokens, "prompt_tokens": prompt_tokens,
            "text_tokens": prompt_tokens - lm_image_tokens,
        })
        print(f"{target:11,} {f'{image.width}x{image.height}':>12} {pixels/1e6:5.2f} "
              f"{vision_tokens:13,} {lm_image_tokens:18,} {prompt_tokens:12,} "
              f"{vision_tokens/max(lm_image_tokens,1):7.1f}x")

    # ---- generated tokens per mode, at the demo's default resolution -------
    print(f"\n{'режим':>26} {'сгенерировано':>14} {'секунд':>8} {'t/s':>7} "
          f"{'на 1 токен промпта':>19}")
    image = fetch_image(origin, max_pixels=2_200_000)
    inputs = prepare(image, layout_prompt)
    base_prompt_tokens = inputs["input_ids"].shape[-1]
    report["generation"] = []
    for mode in args.modes.split(","):
        prompt = dict_promptmode_to_prompt[mode]
        stats = GenerationStats()
        parser._inference(image, prompt, temperature=0.0, stats=stats)
        d = stats.to_dict()
        report["generation"].append({"mode": mode, **d})
        print(f"{mode:>26} {d['generated_tokens']:14,} {d['wall_seconds']:8.2f} "
              f"{d['decode_tps'] or 0:7.1f} {d['generated_tokens']/base_prompt_tokens:19.2f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "token_budget.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {output_dir / 'token_budget.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
