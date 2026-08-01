#!/usr/bin/env bash
# Restart the GLM-OCR demo container after sustained idle to free ALL its VRAM.
#
# Why: GLM-OCR runs in-process (transformers). The worker unloads the weights
# fine (model_state -> stopped), but a Python process that once initialized CUDA
# keeps its ~540 MiB primary context until the process exits. empty_cache() does
# not release it — only process death does. dots.mocr avoids this because its
# vLLM idle-unload stops the whole container; GLM has no such server, so this
# watcher is the equivalent for the in-process engine.
#
# Safe: it only restarts when the worker already reports stopped AND no task is
# running, and only after IDLE_RESTART_S of that state. A restart takes ~2 s and
# the next request reloads the model (~1.3 s) — the cost of full VRAM release.
#
#   nohup bash scripts/glm_idle_watcher.sh &
set -u

CONTAINER="${GLM_DEMO_CONTAINER:-glm_demo}"
HEALTH="http://127.0.0.1:${GLM_DEMO_PORT:-8603}/healthz"
IDLE_RESTART_S="${IDLE_RESTART_S:-300}"   # restart after this many seconds stopped
POLL_S="${POLL_S:-60}"
GLM_GPU="${GLM_DEMO_GPU:-1}"
stopped_since=0

# Read the idle condition from /healthz in one python invocation. Quoted via a
# single-quoted heredoc so the string literals survive verbatim (an earlier
# version lost its quotes to bash heredoc expansion and silently always read -1).
is_idle() {
  curl -sf --max-time 3 "$HEALTH" 2>/dev/null | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    idle = (d.get("worker_alive") and d.get("model_state") == "stopped"
            and not d.get("current_task_id"))
    print(1 if idle else 0)
except Exception:
    print(-1)
'
}

echo "[watcher] watching $CONTAINER (gpu$GLM_GPU); restart after ${IDLE_RESTART_S}s stopped"
while true; do
  sleep "$POLL_S"
  state="$(is_idle)"
  if [ "$state" = "1" ]; then
    stopped_since=$((stopped_since + POLL_S))
    if [ "$stopped_since" -ge "$IDLE_RESTART_S" ]; then
      # Confirm VRAM is actually still held before churning (a just-restarted
      # demo reports stopped too, with ~0 VRAM — nothing to free there).
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
              -i "$GLM_GPU" 2>/dev/null | head -1 | tr -d "[:space:]")"
      if [ -n "$used" ] && [ "$used" -gt 100 ]; then
        echo "[watcher] $(date +%T) $CONTAINER idle ${stopped_since}s, VRAM=${used}MiB -> docker restart"
        docker restart "$CONTAINER" >/dev/null 2>&1 && echo "[watcher] restarted, VRAM will drop on next poll"
        stopped_since=0
      else
        echo "[watcher] idle but VRAM already low (${used}MiB), skip"
        stopped_since=0
      fi
    else
      echo "[watcher] $(date +%T) stopped ${stopped_since}s / ${IDLE_RESTART_S}s"
    fi
  else
    # not idle (loaded/running) or unreachable -> reset the counter
    stopped_since=0
  fi
done
