#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


OLD_PROCESSOR = "configuration_dots.DotsVLProcessor"
NEW_PROCESSOR = "configuration_dots_ocr.DotsVLProcessor"


def load_json(path: Path) -> tuple[bytes, Any]:
    data = path.read_bytes()
    return data, json.loads(data.decode("utf-8"))


def replace_processor(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        if value == OLD_PROCESSOR:
            return NEW_PROCESSOR, True
        return value, False
    if isinstance(value, list):
        changed = False
        result = []
        for item in value:
            new_item, item_changed = replace_processor(item)
            result.append(new_item)
            changed = changed or item_changed
        return result, changed
    if isinstance(value, dict):
        changed = False
        result = {}
        for key, item in value.items():
            new_item, item_changed = replace_processor(item)
            result[key] = new_item
            changed = changed or item_changed
        return result, changed
    return value, False


def encode_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def build_changes(checkpoint: Path) -> list[tuple[Path, bytes, bytes, int]]:
    config_path = checkpoint / "config.json"
    preprocessor_path = checkpoint / "preprocessor_config.json"
    config_bytes, config = load_json(config_path)
    preprocessor_bytes, preprocessor = load_json(preprocessor_path)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: top-level JSON value must be an object")
    changes = []
    if "auto_map" in config:
        config = dict(config)
        del config["auto_map"]
        changes.append(
            (config_path, config_bytes, encode_json(config), config_path.stat().st_mode & 0o7777)
        )
    preprocessor, preprocessor_changed = replace_processor(preprocessor)
    if preprocessor_changed:
        changes.append(
            (
                preprocessor_path,
                preprocessor_bytes,
                encode_json(preprocessor),
                preprocessor_path.stat().st_mode & 0o7777,
            )
        )
    return changes


def apply_changes(changes: list[tuple[Path, bytes, bytes, int]]) -> None:
    for path, original, updated, mode in changes:
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():
            atomic_write(backup, original, mode)
        atomic_write(path, updated, mode)
        print(f"updated {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser()
    try:
        if not checkpoint.is_dir():
            raise ValueError(f"checkpoint directory not found: {checkpoint}")
        changes = build_changes(checkpoint)
        if args.check:
            if changes:
                for path, _, _, _ in changes:
                    print(f"needs update {path}")
                return 1
            print(f"checkpoint is ready: {checkpoint}")
            return 0
        apply_changes(changes)
        if not changes:
            print(f"checkpoint is already ready: {checkpoint}")
        return 0
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
