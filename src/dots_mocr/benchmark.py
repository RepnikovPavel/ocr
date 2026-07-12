import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


PROMPT_MODES = (
    "prompt_layout_all_en",
    "prompt_layout_only_en",
    "prompt_ocr",
    "prompt_grounding_ocr",
    "prompt_web_parsing",
    "prompt_scene_spotting",
    "prompt_image_to_svg",
    "prompt_general",
)


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark one dots.mocr image with an offline checkpoint")
    parser.add_argument("--ckpt", required=True, help="Local checkpoint path")
    parser.add_argument("--input", "--input-path", "--input_path", dest="input", required=True, help="Input image path")
    parser.add_argument("--prompt", choices=PROMPT_MODES, default="prompt_layout_all_en")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument(
        "--attn-implementation",
        "--attn_implementation",
        dest="attn_implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_completion_tokens",
        dest="max_new_tokens",
        type=int,
        default=16384,
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="./benchmark_output")
    parser.add_argument("--metrics", help="Metrics JSON path; defaults to OUTPUT_DIR/metrics.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--custom-prompt", "--custom_prompt", dest="custom_prompt")
    parser.add_argument("--bbox", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--min-pixels", "--min_pixels", dest="min_pixels", type=int)
    parser.add_argument("--max-pixels", "--max_pixels", dest="max_pixels", type=int)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--fitz-preprocess", action="store_true")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def rss_current_bytes():
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def rss_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def cuda_metrics(torch, device):
    if device != "cuda":
        return {
            "cuda_current_memory_bytes": None,
            "cuda_current_reserved_bytes": None,
            "cuda_peak_memory_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    cuda_device = torch.device("cuda")
    return {
        "cuda_current_memory_bytes": int(torch.cuda.memory_allocated(cuda_device)),
        "cuda_current_reserved_bytes": int(torch.cuda.memory_reserved(cuda_device)),
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
    }


def version_metrics(torch):
    values = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dots_mocr": distribution_version("repnikov-dots-mocr"),
        "torch": torch.__version__,
        "transformers": distribution_version("transformers"),
        "qwen_vl_utils": distribution_version("qwen-vl-utils"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    return values


def prepare_and_generate(parser, args, torch):
    from qwen_vl_utils import process_vision_info

    from dots_mocr.utils.consts import MAX_PIXELS, MIN_PIXELS
    from dots_mocr.utils.image_utils import fetch_image, get_image_by_fitz_doc

    origin_image = fetch_image(args.input)
    min_pixels = args.min_pixels
    max_pixels = args.max_pixels
    if args.prompt == "prompt_grounding_ocr":
        min_pixels = min_pixels or MIN_PIXELS
        max_pixels = max_pixels or MAX_PIXELS
    if args.fitz_preprocess:
        image = get_image_by_fitz_doc(origin_image, target_dpi=args.dpi)
        image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
    prompt_text = parser.get_prompt(
        args.prompt,
        bbox=args.bbox,
        origin_image=origin_image,
        image=image,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        custom_prompt=args.custom_prompt,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = parser.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = parser.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs.pop("mm_token_type_ids", None)
    if "attention_mask" in inputs:
        input_tokens = int(inputs.attention_mask.sum().item())
    else:
        input_tokens = int(inputs.input_ids.numel())
    inputs = inputs.to(parser.device)
    generation_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        generation_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        generated_ids = parser.model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_tokens = sum(int(ids.numel()) for ids in generated_ids_trimmed)
    response = parser.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "response": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_text": prompt_text,
        "original_width": origin_image.width,
        "original_height": origin_image.height,
        "processed_width": image.width,
        "processed_height": image.height,
    }


def run(args):
    ckpt = Path(args.ckpt).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    if not input_path.is_file():
        raise FileNotFoundError(f"input image not found: {input_path}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.prompt == "prompt_grounding_ocr" and args.bbox is None:
        raise ValueError("--bbox is required for prompt_grounding_ocr")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt = str(ckpt)
    args.input = str(input_path)

    import torch

    from dots_mocr.cli import DotsMOCRParser

    if args.cpu_threads is not None:
        if args.cpu_threads <= 0:
            raise ValueError("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    load_start = total_start
    with contextlib.redirect_stdout(sys.stderr):
        parser = DotsMOCRParser(
            ckpt=args.ckpt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_completion_tokens=args.max_new_tokens,
            num_thread=1,
            dpi=args.dpi,
            output_dir=str(output_dir),
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            attn_implementation=args.attn_implementation,
            device=args.device,
            dtype=args.dtype,
        )
    if resolved_device == "cuda":
        torch.cuda.synchronize()
    load_end = time.perf_counter()

    inference_start = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        generated = prepare_and_generate(parser, args, torch)
    if resolved_device == "cuda":
        torch.cuda.synchronize()
    inference_end = time.perf_counter()
    total_end = inference_end

    load_seconds = load_end - load_start
    inference_seconds = inference_end - inference_start
    output_tokens = generated["output_tokens"]
    response_path = output_dir / "response.txt"
    response_path.write_text(generated["response"], encoding="utf-8")
    parameter_count = sum(parameter.numel() for parameter in parser.model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in parser.model.parameters() if parameter.requires_grad)
    result = {
        "model": str(ckpt),
        "model_class": type(parser.model).__name__,
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "device": parser.device,
        "dtype": str(parser.dtype).removeprefix("torch."),
        "prompt": args.prompt,
        "prompt_text": generated["prompt_text"],
        "input": str(input_path),
        "output": str(response_path),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": total_end - total_start,
        "input_tokens": generated["input_tokens"],
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / inference_seconds if inference_seconds > 0 else None,
        "rss_current_bytes": rss_current_bytes(),
        "rss_peak_bytes": rss_peak_bytes(),
        "original_width": generated["original_width"],
        "original_height": generated["original_height"],
        "processed_width": generated["processed_width"],
        "processed_height": generated["processed_height"],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "versions": version_metrics(torch),
    }
    result.update(cuda_metrics(torch, parser.device))
    if parser.device == "cuda":
        properties = torch.cuda.get_device_properties(torch.device("cuda"))
        result["cuda_device_name"] = properties.name
        result["cuda_device_total_memory_bytes"] = int(properties.total_memory)
        result["cuda_compute_capability"] = f"{properties.major}.{properties.minor}"
    else:
        result["cuda_device_name"] = None
        result["cuda_device_total_memory_bytes"] = None
        result["cuda_compute_capability"] = None
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    metrics_path = Path(args.metrics).expanduser().resolve() if args.metrics else Path(args.output_dir).expanduser().resolve() / "metrics.json"
    if str(args.metrics) != "-":
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
