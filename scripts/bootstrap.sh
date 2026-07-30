#!/usr/bin/env bash
# bootstrap.sh — one-command server deploy, zero decisions for the operator.
#
#   bash scripts/bootstrap.sh                 # auto-detect checkpoint + CUDA, deploy, verify
#   bash scripts/bootstrap.sh --bind lan      # publish on 0.0.0.0 (trusted LAN) instead of SSH tunnel
#   bash scripts/bootstrap.sh --dry-run       # preflight + auto-detect only, deploy nothing
#   bash scripts/bootstrap.sh --down          # stop the deployment
#
# WHY THIS EXISTS
# Every other launcher here needs you to already KNOW three things: where the
# dots.mocr checkpoint lives on this host, which CUDA your driver is (so the
# image actually starts — the trtllm cu13x build silently refuses on a CUDA 12
# driver like a 4090's), and which image tags match. Figuring that out from
# first principles is what used to cost a deploy agent ~150K tokens. This
# script finds the checkpoint, detects the driver, picks compatible images,
# bakes the ones that are missing, hands everything to deploy_server.sh, then
# runs doctor.sh and prints the one command your colleagues must run.
#
# The parse cache is NOT something this (or any deploy) sets up: it is built
# into the demo (demo/docstore.py, SHA-256 in SQLite) and starts with the demo
# container. There is no separate cache component and no SeaweedFS to deploy.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ---- pretty printers (mirror doctor.sh's style) -----------------------------
if [ -t 1 ]; then
    G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; D=$'\033[2m'; O=$'\033[0m'
else
    G=""; R=""; Y=""; D=""; O=""
fi
ok()   { echo "${G}OK   $*${O}"; }
note() { echo "${D}     $*${O}"; }
die()  { echo "${R}FAIL $*${O}" >&2; exit 1; }
step() { echo; echo "${G}==>$*${O}"; }

have() { command -v "$1" >/dev/null 2>&1; }

# ---- args -------------------------------------------------------------------
BIND=localhost
DRY_RUN=0
ACTION=up
while [ $# -gt 0 ]; do
    case "$1" in
        --bind)
            shift; [ $# -gt 0 ] || die "--bind needs a value: localhost|lan"
            case "$1" in localhost|127.0.0.1) BIND=localhost ;; lan|0.0.0.0|*) BIND=lan ;; esac ;;
        --bind=*)       case "${1#--bind=}" in localhost|127.0.0.1) BIND=localhost ;; *) BIND=lan ;; esac ;;
        --dry-run|-n)   DRY_RUN=1 ;;
        --down)         ACTION=down ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        *)              die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

# Hand --down straight to deploy_server.sh: there is nothing to detect when
# the goal is to stop.
if [ "$ACTION" = down ]; then
    step "stopping the deployment"
    exec env STATE="${STATE:-$ROOT/server_state}" \
        bash "$ROOT/scripts/deploy_server.sh" --down
fi

echo "=== dots.mocr bootstrap ==="
[ "$DRY_RUN" = 1 ] && note "DRY RUN: detect + preflight only, nothing will be deployed"

# ---- 1. preflight (fail fast, before any slow step) -------------------------
step "preflight"
have docker      || die "docker not found. Install Docker + the NVIDIA Container Toolkit."
have nvidia-smi  || die "nvidia-smi not found. This deploy needs an NVIDIA GPU + driver."

# free space on the filesystem holding $ROOT (weights + state need GBs)
free_gb=$(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [ "${free_gb:-0}" -ge 1 ] && [ "$free_gb" -lt 30 ]; then
    echo "${Y}WARN only ${free_gb} GiB free under $ROOT — weights are ~6 GiB and state grows${O}"
else
    ok "${free_gb:-?} GiB free under $ROOT"
fi

# ports the deployment uses
DEMO_PORT="${DEMO_PORT:-8601}"
VLLM_PORT="${VLLM_PORT:-8000}"
port_busy() { ss -ltn "sport = :$1" 2>/dev/null | tail -n +2 | grep -q .; }
for p in "$DEMO_PORT" "$VLLM_PORT"; do
    # the existing containers occupying these are fine — a re-deploy — but an
    # UNRELATED process (another service) is exactly the 404-trap deploy_server.sh
    # warns about; refuse early instead of letting it surface as a confusing error.
    if port_busy "$p"; then
        if docker ps --format '{{.Ports}}' 2>/dev/null | grep -q ":$p->"; then
            note "port $p held by a docker container (likely ours — redeploy)"
        else
            die "port $p is busy with something other than our containers. Free it or set DEMO_PORT/VLLM_PORT."
        fi
    fi
done
ok "ports ${DEMO_PORT} (demo) and ${VLLM_PORT} (vLLM) are clear"

# ---- 2. find the checkpoint -------------------------------------------------
step "locate the dots.mocr checkpoint"
# Explicit override always wins.
CKPT="${CKPT:-${DOTS_MOCR_CKPT:-${CKPTDIR:-}}}"
if [ -n "$CKPT" ] && [ -d "$CKPT" ]; then
    ok "checkpoint (from \$CKPT): $CKPT"
else
    # Scan the places this snapshot actually lives on our hosts: HF cache layout
    # (models--rednote-hilab--dots.mocr/snapshots/<rev>) under /mnt and the home
    # cache, plus a couple of flat-mount variants. First hit with weights wins.
    found=""
    for base in \
        "$HOME/.cache/huggingface/hub" \
        /mnt/nvme /mnt/nvme2 /mnt/data2 /mnt/hf /mnt/models \
        /root/.cache/huggingface/hub ; do
        [ -d "$base" ] || continue
        # -L follows the snapshot symlinks HF creates; index.json is the marker
        # that this dir really holds weights, not an empty/incomplete download.
        hit=$(find -L "$base" -type d -name "models--rednote-hilab--dots.mocr" 2>/dev/null | head -1 || true)
        [ -n "$hit" ] || continue
        snap="$hit/snapshots"
        if [ -d "$snap" ]; then
            for rev in "$snap"/*/; do
                if [ -f "${rev}model.safetensors.index.json" ] || \
                   ls "${rev}"*.safetensors >/dev/null 2>&1; then
                    found="$(cd "$rev" && pwd -P)"; break
                fi
            done
        fi
        [ -n "$found" ] && break
    done
    if [ -n "$found" ]; then
        CKPT="$found"
        ok "checkpoint (auto-detected): $CKPT"
    else
        die "dots.mocr checkpoint not found. Either:
  - pass the path: CKPT=/path/to/snapshot bash $0
  - download it:   bash $ROOT/scripts/download_checkpoint.sh"
    fi
fi
# Validate it actually has weights, not just a stray dir.
if ! { [ -f "$CKPT/model.safetensors.index.json" ] || ls "$CKPT"/*.safetensors >/dev/null 2>&1; }; then
    die "$CKPT has no .safetensors weights. Is it a complete snapshot? (download_checkpoint.sh)"
fi

# ---- 3. detect the CUDA driver → pick image tags ---------------------------
# This is the step that used to eat tokens: the trtllm:1.3.0rc20 image is a
# cu13x build and will NOT start on a CUDA 12.x driver (every GeForce 4090 in
# driver <580). We detect the driver once and pick tags known to work.
step "detect CUDA driver and pick images"
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 | tr -dc '0-9')
driver=${driver:-0}
note "driver major version: ${driver}"

# Defaults match docker/compose.server.yml. FORCE_* let an operator override.
VLLM_IMAGE="${FORCE_VLLM_IMAGE:-}"
DOTS_MOCR_IMAGE="${FORCE_DEMO_IMAGE:-}"

if [ -z "$VLLM_IMAGE" ] || [ -z "$DOTS_MOCR_IMAGE" ]; then
    if [ "$driver" -ge 580 ]; then
        # CUDA 13-capable driver: the repo's default trtllm build works.
        [ -z "$VLLM_IMAGE" ]       && VLLM_IMAGE="dots-mocr:vllm"
        [ -z "$DOTS_MOCR_IMAGE" ]  && DOTS_MOCR_IMAGE="dots-mocr:trtllm-1.3.0rc20"
        note "driver >= 580 → CUDA 13 path (default trtllm/vllm images)"
    else
        # CUDA 12.x driver (4090 etc.): cu13x builds refuse to start. vLLM ships a
        # CUDA-12 runtime itself; the demo must run on the bench-cu126 image.
        [ -z "$VLLM_IMAGE" ]       && VLLM_IMAGE="vllm/vllm-openai:v0.17.1"
        [ -z "$DOTS_MOCR_IMAGE" ]  && DOTS_MOCR_IMAGE="dots-mocr:bench-cu126"
        note "driver < 580 → CUDA 12 path (vllm/vllm-openai + bench-cu126 demo)"
    fi
fi
ok "vLLM image : $VLLM_IMAGE"
ok "demo image : $DOTS_MOCR_IMAGE"

# ---- 4. bake missing images -------------------------------------------------
# build.sh builds the 60 GB trtllm image (slow); the two images a CUDA-12 host
# needs are tiny and quick, so build them directly rather than touching build.sh.
step "ensure images exist (build if missing)"
ensure_image() {
    local img="$1" dockerfile="$2" ctx="$3"
    if docker image inspect "$img" >/dev/null 2>&1; then
        ok "$img present"
    else
        echo "${D}     building $img from $dockerfile ...${O}"
        DOCKER_BUILDKIT=1 docker build -q -t "$img" -f "$ROOT/$dockerfile" "$ROOT/$ctx" \
            || die "failed to build $img (see docker output above)"
        ok "$img built"
    fi
}
case "$VLLM_IMAGE" in
    dots-mocr:vllm)        ensure_image "dots-mocr:vllm"        docker/Dockerfile.vllm docker/ ;;
    vllm/vllm-openai*)     ok "$VLLM_IMAGE is a public image (docker will pull on up)" ;;
    *)                     note "custom vLLM image $VLLM_IMAGE — assumed already present" ;;
esac
case "$DOTS_MOCR_IMAGE" in
    dots-mocr:bench-cu126)        ensure_image "dots-mocr:bench-cu126" docker/Dockerfile.bench docker/ ;;
    dots-mocr:trtllm*)            ensure_image "$DOTS_MOCR_IMAGE" docker/Dockerfile . ;;
    *)                            note "custom demo image $DOTS_MOCR_IMAGE — assumed already present" ;;
esac

[ "$DRY_RUN" = 1 ] && {
    echo
    echo "==>${G} dry run complete — would deploy with:${O}"
    note "CKPT=$CKPT"
    note "DOTS_MOCR_IMAGE=$DOTS_MOCR_IMAGE  VLLM_IMAGE=$VLLM_IMAGE"
    note "DEMO_PORT=$DEMO_PORT  VLLM_PORT=$VLLM_PORT  bind=$BIND"
    exit 0
}

# ---- 5. deploy (existing script does the real work) ------------------------
step "deploy (vLLM + demo via deploy_server.sh)"
STATE="${STATE:-$ROOT/server_state}"
# Pick the first free GPU for vLLM. deploy_server.sh / compose default to GPU 0;
# honoring VLLM_GPU lets an operator pin it without editing two places.
: "${VLLM_GPU:=0}"

# translate our bind mode into the address deploy_server.sh / uvicorn expect
case "$BIND" in
    lan)        export DOTS_MOCR_BIND=0.0.0.0 ;;
    localhost)  export DOTS_MOCR_BIND=127.0.0.1 ;;
esac

export CKPT STATE DOTS_MOCR_IMAGE VLLM_IMAGE VLLM_GPU DEMO_PORT VLLM_PORT
bash "$ROOT/scripts/deploy_server.sh"

# ---- 6. verify --------------------------------------------------------------
step "verify with doctor.sh"
DEMO_PORT=$DEMO_PORT VLLM_PORT=$VLLM_PORT STATE="$STATE" \
    bash "$ROOT/scripts/doctor.sh" || echo "${Y}WARN doctor reported issues — the service may still be warming up (vLLM ~40 s)${O}"

# ---- 7. print the one command colleagues run -------------------------------
step "ready"
case "$BIND" in
    lan)
        # best-effort: the address colleagues reach this host at on the LAN
        host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        [ -n "$host_ip" ] || host_ip="<server-ip>"
        echo "Service is on the LAN at ${G}http://${host_ip}:${DEMO_PORT}${O} (no auth — trusted network only)."
        echo "Colleagues run, once:"
        echo
        echo "  ${G}curl -fsSL https://raw.githubusercontent.com/RepnikovPavel/ocrc/main/install.sh | sh && OCRC_SERVER=http://${host_ip}:${DEMO_PORT} ocrc parse doc.pdf${O}"
        ;;
    localhost)
        echo "Service is on 127.0.0.1:${DEMO_PORT} (behind an SSH tunnel — not exposed)."
        echo "Colleagues run, once, to use it from their laptop:"
        echo
        echo "  ${G}curl -fsSL https://raw.githubusercontent.com/RepnikovPavel/ocrc/main/install.sh | sh${O}"
        echo "  ${G}ssh -N -L ${DEMO_PORT}:127.0.0.1:${DEMO_PORT} <server>${O}   # in one terminal, leave it open"
        echo "  ${G}OCRC_SERVER=http://127.0.0.1:${DEMO_PORT} ocrc parse doc.pdf${O}"
        ;;
esac
echo
echo "On this server itself:  ${G}OCRC_SERVER=http://127.0.0.1:${DEMO_PORT} ocrc parse doc.pdf${O}"
echo "Logs:  docker logs -f dots_mocr_demo   |   Stop:  bash $0 --down"
