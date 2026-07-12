#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DOCKER_BUILDKIT=1 docker build \
    --pull=false \
    --network=host \
    --build-arg BASE_IMAGE=trtllm:1.3.0rc20 \
    --tag dots-mocr:trtllm-1.3.0rc20 \
    --file "$ROOT/docker/Dockerfile" \
    "$ROOT"
