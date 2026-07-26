#!/usr/bin/env bash
# doctor.sh — quick health check for a dots.mocr deployment.
#
# Run this FIRST when something looks off ("ocrc hangs", "queue stuck",
# "operation not permitted", "demo :8601 does not answer"). It tells you what
# is wrong in plain text and, where it can, the exact command to fix it.
#
#   scripts/doctor.sh                       # auto-detect local/server
#   DEMO_PORT=8601 scripts/doctor.sh        # force demo port
#   VLLM_PORT=8010 scripts/doctor.sh        # force vLLM port (server deploy)
#   ssh server scripts/doctor.sh            # run ON the server over ssh
#
# The script is local-only: it curls DEMO_HOST:DEMO_PORT from where it runs.
# To check a remote deployment, ssh there first (the host that runs the demo
# container) and run it there — that is where docker ps, /state and GPU live.
#
# Exit code: 0 = healthy, 1 = at least one critical check failed.
# Dependencies: bash, curl, python3. docker and nvidia-smi are used if present.
set -u

# ---- pretty printers --------------------------------------------------------
GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
[ -t 1 ] || { GREEN=""; RED=""; YEL=""; DIM=""; OFF=""; }

ok()   { echo "${GREEN}OK   $*${OFF}"; }
warn() { echo "${YEL}WARN $*${OFF}"; WARN_CNT=$((WARN_CNT+1)); }
fail() { echo "${RED}FAIL $*${OFF}"; FAIL_CNT=$((FAIL_CNT+1)); }
note() { echo "${DIM}     $*${OFF}"; }
WARN_CNT=0; FAIL_CNT=0

# ---- detect where we are ----------------------------------------------------
# 1) explicit env wins; 2) docker ps hints at a server deploy; 3) dev defaults.
DEMO_PORT="${DEMO_PORT:-${PORT:-8601}}"
DEMO_HOST="${DEMO_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-rednote-hilab/dots.mocr}"
STATE_DIR="${STATE_DIR:-${DEMO_STATE_DIR:-}}"

have() { command -v "$1" >/dev/null 2>&1; }

# Try to learn the state dir from a running demo container (more reliable than
# guessing). Only relevant on a server deploy.
if [ -z "$STATE_DIR" ] && have docker; then
    sd=$(docker inspect dots_mocr_demo \
         --format '{{range .Mounts}}{{if (eq .Destination "/state")}}{{.Source}}{{end}}{{end}}' \
         2>/dev/null || true)
    [ -n "$sd" ] && STATE_DIR="$sd"
fi
# Likewise, prefer the actual VLLM_PORT the demo is wired to, if reachable.
if [ -z "${VLLM_PORT_EXPLICIT:-}" ] && have docker; then
    vp=$(docker inspect dots_mocr_demo \
         --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
         | awk -F= '$1=="DEMO_VLLM_URL"{print $2}' \
         | sed -nE 's#http://[^:/]*:([0-9]+)/?.*#\1#p' | head -1)
    [ -n "$vp" ] && VLLM_PORT="$vp"
fi
[ -z "$STATE_DIR" ] && STATE_DIR="./server_state"

echo "=== dots.mocr doctor ==="
note "demo=${DEMO_HOST}:${DEMO_PORT} vllm=:${VLLM_PORT} state=${STATE_DIR}"
echo

# ---- 1. demo container (only if docker is usable) ---------------------------
if have docker; then
    line=$(docker ps -a --filter name=dots_mocr_demo \
            --format '{{.Names}}|{{.Status}}|{{.Ports}}' 2>/dev/null | head -1)
    if [ -z "$line" ]; then
        note "no dots_mocr_demo container on this host (maybe a dev run without docker)"
    else
        name=${line%%|*}; rest=${line#*|}; status=${rest%%|*}
        case "$status" in
            Up*)    ok "demo container: $status" ;;
            Exited*) fail "demo container is $status — start it:"
                     note "    docker start dots_mocr_demo" ;;
            *)      warn "demo container: $status" ;;
        esac
    fi
fi

# ---- 2. demo /healthz -------------------------------------------------------
hz=$(curl -sS -m 5 -o /dev/null -w '%{http_code}' "http://${DEMO_HOST}:${DEMO_PORT}/healthz" 2>/dev/null || echo 000)
case "$hz" in
    200) ok "/healthz → 200" ;;
    503) fail "/healthz → 503 (worker thread is dead). Fast fix:"
            note "    docker restart dots_mocr_demo"
            note "    the in-process watchdog (server.py) should already be restarting it; if not, this is real" ;;
    000) fail "/healthz unreachable on ${DEMO_HOST}:${DEMO_PORT}."
            if have docker; then
                note "    is the demo container running? docker ps | grep dots_mocr_demo"
            else
                note "    is the demo started? scripts/run_local_vllm.sh (or run_local.sh)"
            fi
            note "    if the server is remote: open an SSH tunnel first, see ocr/AGENTS.md" ;;
    *)   warn "/healthz → HTTP $hz (unexpected)" ;;
esac

# parse worker_alive / model_state out of /healthz (python3 if available, else grep)
if [ "$hz" = "200" ] && have python3; then
    body=$(curl -sS -m 5 "http://${DEMO_HOST}:${DEMO_PORT}/healthz" 2>/dev/null || echo "")
    read_field() { python3 -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(d.get('$1',''))" <<<"$body" 2>/dev/null; }
    wa=$(read_field worker_alive)
    ms=$(read_field model_state)
    paused=$(read_field paused)
    err=$(read_field model_error)
    if [ "$wa" = "False" ]; then
        fail "worker_alive=false — worker thread died; docker restart dots_mocr_demo"
    elif [ -n "$wa" ]; then
        ok "worker_alive=true  model_state=$ms  paused=$paused"
    fi
    [ -n "$err" ] && [ "$err" != "None" ] && [ "$err" != "null" ] && warn "model_error: $err"
fi

# ---- 3. vLLM serving the right model ----------------------------------------
vm=$(curl -sS -m 5 "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || echo "")
if have python3 && [ -n "$vm" ] && printf '%s' "$vm" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)' 2>/dev/null; then
    if printf '%s' "$vm" | grep -q "$VLLM_MODEL"; then
        ok "vLLM on :${VLLM_PORT} serving $VLLM_MODEL"
    else
        fail "vLLM is up but $VLLM_MODEL not in /v1/models; got:"
        note "    $(printf '%s' "$vm" | head -c 200)"
    fi
else
    fail "vLLM unreachable on 127.0.0.1:${VLLM_PORT}."
    if have docker; then
        note "    docker ps | grep dots_vllm  — if Exited: docker start dots_vllm"
    else
        note "    start vLLM: scripts/run_local_vllm.sh (brings up both)"
    fi
fi

# ---- 4. queue length + store stats ------------------------------------------
if have python3; then
    q=$(curl -sS -m 5 "http://${DEMO_HOST}:${DEMO_PORT}/api/v1/queue" 2>/dev/null || echo "")
    qlen=$(python3 -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(len(d.get('queue',[])))" <<<"$q" 2>/dev/null || echo "?")
    case "$qlen" in
        0)        ok "queue empty" ;;
        [1-9]*)   warn "queue has $qlen task(s) waiting — check model_state above; if loaded, parsing is just slow" ;;
        *)        note "queue: $q" ;;
    esac
fi

# ---- 5. state dir / sqlite permissions --------------------------------------
if [ -n "$STATE_DIR" ]; then
    if [ -d "$STATE_DIR" ]; then
        perm=$(stat -c '%a %U:%G' "$STATE_DIR" 2>/dev/null || stat -f '%A %u:%g' "$STATE_DIR" 2>/dev/null || echo "?")
        ok "state dir $STATE_DIR ($perm)"
        # check the db files specifically — the recurring incident was root:root on demo.db
        if ls "$STATE_DIR"/demo.db* >/dev/null 2>&1; then
            while read -r f; do
                [ -z "$f" ] && continue
                fp=$(stat -c '%a %U:%G' "$f" 2>/dev/null || stat -f '%A %u:%g' "$f" 2>/dev/null || echo "?")
                # warn if owned by root AND not world-writable — that is the trap
                owner=${fp#* }
                mode=${fp%% *}
                case "$owner" in
                    root:*) [ "${mode#??}" -lt 6 ] 2>/dev/null && warn "$f is $fp — SQLite may fail WAL checkpoint; fix: sudo chmod -R a+rwX $STATE_DIR" ;;
                esac
            done < <(ls -1 "$STATE_DIR"/demo.db* 2>/dev/null)
        fi
    else
        warn "state dir $STATE_DIR does not exist yet (normal before first deploy)"
    fi
fi

# ---- 6. GPU -----------------------------------------------------------------
if have nvidia-smi; then
    gpus=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true)
    if [ -n "$gpus" ]; then
        echo
        note "GPU snapshot:"
        while IFS= read -r line; do echo "     $line"; done <<<"$gpus"
    fi
fi

echo
if [ "$FAIL_CNT" -gt 0 ]; then
    echo "${RED}=== $FAIL_CNT critical, $WARN_CNT warnings ===${OFF}"
    exit 1
fi
if [ "$WARN_CNT" -gt 0 ]; then
    echo "${YEL}=== healthy with $WARN_CNT warning(s) ===${OFF}"
else
    echo "${GREEN}=== all checks passed ===${OFF}"
fi
exit 0
