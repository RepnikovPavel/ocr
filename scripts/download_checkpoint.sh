#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 dots.mocr|dots.mocr-svg [hf-cache-dir]" >&2
    exit 2
fi

model="$1"
case "$model" in
    dots.mocr|dots.mocr-svg) ;;
    *)
        echo "unsupported model: $model" >&2
        exit 2
        ;;
esac

cache_root="${2:-${HF_HOME:-/mnt/nvme/huggingface}}"
model_id="rednote-hilab/$model"
snapshot_dir="$cache_root/models--rednote-hilab--$model/snapshots/main"
base_url="https://huggingface.co/$model_id/resolve/main"

common_files=(
    config.json
    chat_template.json
    configuration_dots.py
    generation_config.json
    merges.txt
    model.safetensors.index.json
    modeling_dots_ocr.py
    modeling_dots_vision.py
    preprocessor_config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
    NOTICE
)
files=("${common_files[@]}")
if [[ "$model" == "dots.mocr-svg" ]]; then
    files+=(.gitattributes README.md)
fi
shards=(
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
)

mkdir -p "$snapshot_dir"
cd "$snapshot_dir"

download() {
    local remote_name="$1"
    local local_name="${2:-$1}"
    wget -c --show-progress --output-document="$local_name" "$base_url/$remote_name"
}

for file in "${files[@]}"; do
    download "$file"
done
download "dots.mocr%20LICENSE%20AGREEMENT" "dots.mocr LICENSE AGREEMENT"
for shard in "${shards[@]}"; do
    download "$shard"
done

required=("${files[@]}" "dots.mocr LICENSE AGREEMENT" "${shards[@]}")
for file in "${required[@]}"; do
    if [[ ! -s "$file" ]]; then
        echo "missing or empty file: $snapshot_dir/$file" >&2
        exit 1
    fi
done

printf '%s\n' "$snapshot_dir"
