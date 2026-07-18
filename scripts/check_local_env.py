#!/usr/bin/env python3
"""Diagnose a local (no-Docker) dots.mocr environment.

Run it after scripts/setup_local.sh, or any time inference behaves oddly:

    python3 scripts/check_local_env.py [--ckpt /path/to/snapshot]

Exits non-zero if anything that would break inference is wrong. The transformers
major version is a hard check: this port targets 5.x (the API the upstream code was
written against), and 4.x is not a supported configuration.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

MIN_TRANSFORMERS_MAJOR = 5

problems = []
warnings_ = []
notes = []


def check(label, value, ok=True, fix=None, fatal=True):
    """Report one probe. `fatal=False` marks a condition that degrades inference
    but does not prevent it (no CUDA, no flex kernel), so it warns instead of
    failing the whole check — run_local.sh gates startup on the exit code."""
    status = "ok " if ok else ("FAIL" if fatal else "warn")
    print(f"  {status}  {label}: {value}")
    if not ok:
        message = f"{label}: {value}" + (f"\n         fix: {fix}" if fix else "")
        (problems if fatal else warnings_).append(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("CKPTDIR") or os.environ.get("DOTS_MOCR_CKPT"))
    args = ap.parse_args()

    print(f"interpreter: {sys.executable}")
    print(f"repo:        {REPO_ROOT}")

    print("\npython")
    check("version", ".".join(map(str, sys.version_info[:3])), sys.version_info >= (3, 10),
          "use python >= 3.10")

    print("\ntorch")
    try:
        import torch
    except ImportError as error:
        check("import", str(error), False, "scripts/setup_local.sh")
        return report()
    check("version", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    check("cuda available", cuda_ok, cuda_ok,
          "CPU-only works but is very slow; expect minutes per page", fatal=False)
    if cuda_ok:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            check(f"cuda:{index}", f"{props.name}, sm_{props.major}{props.minor}, "
                                  f"{free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB")
            if total / 2**30 < 8:
                notes.append(f"cuda:{index} has < 8 GiB: the 3B model in bf16 needs ~6.6 GiB of weights alone")

    print("\nflex attention")
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention  # noqa: F401
        check("torch.nn.attention.flex_attention", "importable")
    except ImportError as error:
        check("torch.nn.attention.flex_attention", str(error), False,
              "needs torch >= 2.5; the sdpa backend still works (--attn_implementation sdpa)",
              fatal=False)

    print("\ntransformers")
    try:
        import transformers
    except ImportError as error:
        check("import", str(error), False, "scripts/setup_local.sh")
        return report()
    major = int(transformers.__version__.split(".")[0])
    check("version", transformers.__version__, major >= MIN_TRANSFORMERS_MAJOR,
          f"pip install 'transformers=={os.environ.get('DOTS_MOCR_TRANSFORMERS', '5.5.4')}' — "
          "this port targets transformers 5.x")

    print("\ndots_mocr")
    try:
        from dots_mocr.transformers_patch import register_transformers
        register_transformers()
        check("register_transformers()", "ok")
    except Exception as error:  # noqa: BLE001 - surface anything to the user
        check("register_transformers()", f"{type(error).__name__}: {error}", False)

    for module, package in (("fitz", "PyMuPDF"), ("cairosvg", "CairoSVG"),
                            ("qwen_vl_utils", "qwen-vl-utils"), ("fastapi", "fastapi"),
                            ("uvicorn", "uvicorn"), ("multipart", "python-multipart")):
        try:
            __import__(module)
            check(package, "importable")
        except ImportError:
            check(package, "missing", False, f"pip install {package}")

    if args.ckpt:
        print("\ncheckpoint")
        ckpt = Path(args.ckpt)
        check("path", str(ckpt), ckpt.is_dir(), "pass --ckpt or set CKPTDIR")
        if ckpt.is_dir():
            index = ckpt / "model.safetensors.index.json"
            check("weights", "model.safetensors.index.json present" if index.is_file() else "missing",
                  index.is_file(), "run scripts/download_checkpoint.sh")
            config = ckpt / "config.json"
            check("config.json", "present" if config.is_file() else "missing", config.is_file())
    else:
        notes.append("no --ckpt/CKPTDIR given: checkpoint not verified")

    return report()


def report():
    print()
    for note in notes:
        print(f"note: {note}")
    for warning in warnings_:
        print(f"warning: {warning}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("environment looks good" + (" (with warnings)" if warnings_ else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
