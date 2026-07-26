# AGENTS.md — dots.mocr for AI agents

Low-token brief for agents working in this repo. Mirrors the style of
[`ocrc/prompt.txt`](https://github.com/RepnikovPavel/ocrc/blob/main/prompt.txt).
Read this first; the human-facing [README.md](README.md) has the long version
and [docs/architecture.md](docs/architecture.md) has the deep model math.

## What this is / is not

- The dots.mocr **web demo + agent API**: a FastAPI service (port 8601) that
  takes PDFs/images and returns parsed markdown + layout JSON. Two engines:
  `vllm` (default, ~1.85× faster, separate container `dots_vllm`) and
  `transformers` (in-process, no extra server).
- The companion CLI [`ocrc`](https://github.com/RepnikovPavel/ocrc) lives in a
  separate repo — install it from there, do not duplicate its logic here.
- Use **dots.mocr** (the main model). Do not invest in `dots.mocr-svg` —
  README §"Какая из двух моделей" says why (broken XML on text pages, ~3× slower).

## The OCRC_SERVER question (read this before any "connection refused")

`ocrc` (and most clients) default to `http://127.0.0.1:8601`. That address means
"on the same host where the demo is running." Three real situations:

1. **You are on the demo host itself** (e.g. you ssh'd to the server): the
   default works as-is.
2. **You are on a different machine** (your laptop, a CI runner): open an SSH
   tunnel first, then point `OCRC_SERVER` at the local end of it:
   ```sh
   ssh -N -L 8601:127.0.0.1:8601 server          # in one terminal, leave it
   OCRC_SERVER=http://127.0.0.1:8601 ocrc parse paper.pdf   # in another
   ```
   Pick a high random local port if 8601 may collide: `-L 41729:127.0.0.1:8601`,
   then `OCRC_SERVER=http://127.0.0.1:41729`.
3. **No route at all**: there is no public HTTP URL. tuna (`ru.tuna.am`) is an
   SSH-only gateway; it does not publish web ports without extra setup.

## Fast path: parse a document

```sh
curl -fsSL https://raw.githubusercontent.com/RepnikovPavel/ocrc/main/install.sh | sh
OCRC_SERVER=http://127.0.0.1:8601 ocrc parse paper.pdf          # → ./ocrc-out/...
OCRC_SERVER=http://127.0.0.1:8601 ocrc parse https://arxiv.org/pdf/XXXX.YYYYY > out.zip
OCRC_SERVER=http://127.0.0.1:8601 ocrc queue                    # see waiting tasks
```

`ocrc parse` blocks until done (~10-30 s/page). Re-submitting the same file is
free (content-addressed cache); do not build your own dedup. Full CLI reference:
[`ocrc/prompt.txt`](https://github.com/RepnikovPavel/ocrc/blob/main/prompt.txt).

## Diagnose first

When anything looks off ("ocrc hangs", "queue stuck", "operation not
permitted", "/healthz 503"), **run the doctor before changing code**:

```sh
scripts/doctor.sh                                 # local
ssh server 'bash /mnt/nvme2/ocr-flex/scripts/doctor.sh'   # on the server
```

It checks the demo container, `/healthz`, vLLM, the queue, **sqlite file
permissions** (the recurring root:root WAL-checkpoint trap) and the GPU, then
prints the exact fix command. Exit non-zero = at least one critical failure.

## Bring the service up

Three scenarios, simplest first:

```sh
# 1) local, vLLM engine (default, needs a checkpoint + GPU)
CKPTDIR=/path/to/dots.mocr scripts/run_local_vllm.sh

# 2) local, in-process transformers engine (no separate vLLM container)
CKPTDIR=/path/to/dots.mocr scripts/run_local.sh

# 3) server deploy (docker compose: dots_mocr_demo + dots_vllm)
CKPT=/path/to/snapshot STATE=/path/to/state scripts/deploy_server.sh
```

Both bring the demo up at `http://127.0.0.1:8601`. You need a checkpoint at
`$CKPTDIR`/`$CKPT` — get it with `scripts/download_checkpoint.sh` then
`scripts/prepare_checkpoint.py`.

## Validate without a GPU

```sh
python3 -m pytest tests -m "not gpu"           # fast unit/regression tests
MAX_NEW_TOKENS=8 scripts/test_dots_mocr_cpu.sh # one short end-to-end on synthetic data
```

`scripts/check_local_env.py` verifies torch/transformers versions before you
waste time on a broken env.

## Code map (1 line per file)

| File | What |
| --- | --- |
| `demo/server.py` | FastAPI app, env config, `WORKER` singleton, `/healthz`, the `/api/*` browser routes |
| `demo/worker.py` | `DemoWorker(Thread)`: owns the model, lazy load/unload, picks tasks from the queue, manages the vLLM container via docker.sock |
| `demo/db.py` | SQLite (sessions/jobs/tasks), WAL, short-lived connections |
| `demo/agent_api.py` | `/api/v1` router for agents: `documents`, `documents/{sha}/bundle`, `queue`, `events` (SSE), `search`, `stats` |
| `demo/docstore.py` | content-addressed store (SHA-256 + prompt mode + pages), FTS5 search |
| `src/dots_mocr/cli.py` | `DotsMOCRParser`, in-process inference (HF generate) |
| `src/dots_mocr/model/vllm_parser.py` | subclass that swaps load+generate for HTTP to a vLLM server |
| `src/dots_mocr/utils/prompts.py` | the 7 prompt modes (layout_all/only, ocr, grounding_ocr, web_parsing, scene_spotting, general) |
| `src/dots_mocr/transformers_patch/` | ported upstream modeling code |
| `docker/compose.server.yml` | server deploy: `vllm` + `demo` services, docker.sock mount, `restart: unless-stopped` |
| `scripts/deploy_server.sh` | one-shot server deploy (chmod's the state dir to avoid the sqlite trap) |
| `scripts/doctor.sh` | health check — run this first when debugging |

## Key env vars

| Var | Default | Meaning |
| --- | --- | --- |
| `DEMO_STATE_DIR` | `/state` (in container) | where `demo.db` lives; back up this one file |
| `DEMO_ENGINE` | `vllm` | `vllm` or `transformers` |
| `DEMO_VLLM_URL` | `http://127.0.0.1:8000/v1` | vLLM OpenAI endpoint |
| `DEMO_VLLM_CONTAINER` | `dots_vllm` | container the worker stops/starts on idle-unload (needs docker.sock) |
| `DEMO_IDLE_UNLOAD_S` | `180` | seconds idle before the model is unloaded |
| `DEMO_VARIANT` | `mocr` | `mocr` (default) or `svg` (do not develop) |
| `DEMO_PORT` / `PORT` | `8601` | demo HTTP port |
| `DEMO_SKIP_WATCHDOG` | unset | set to `1` to disable the worker watchdog thread (debug only) |

## State and backup

All queue + parsed-document metadata is one SQLite file: `$DEMO_STATE_DIR/demo.db`
(with `-shm`/`-wal` sidecars). Backup = copy those three files. The parsed
images/markdown bundles live in `$DEMO_STATE_DIR/jobs/<task_id>/`.

## Common agent mistakes (each has bitten us at least once)

1. **The worker thread silently died and the queue froze.** Symptom: `/api/v1/queue`
   shows tasks `queued` forever, GPU idle, but the demo container is "healthy".
   The container's `/healthz` now reports `worker_alive: false` (and HTTP 503),
   so docker restarts it; the in-process watchdog in `server.py` also rebuilds
   the worker. If you see this anyway, `docker restart dots_mocr_demo`.
2. **First task after idle takes ~30 s extra.** Idle-unload stops the vLLM
   container; the first new task has to wake it and reload the weights. Not a
   bug — wait or call `POST /api/model/start` to pre-warm.
3. **`DEMO_VLLM_CONTAINER` set but `/var/run/docker.sock` not mounted.** Idle-unload
   then frees only the weights (~6 GiB), leaving ~700 MiB of CUDA context
   resident. Check `docker inspect dots_mocr_demo` for the sock mount.
4. **`demo.db*` owned by `root:root` with mode `644`.** Eventually crashes the
   worker with `sqlite3.OperationalError: unable to open database file` when WAL
   tries to checkpoint. `scripts/doctor.sh` flags this; fix is
   `sudo chmod -R a+rwX $DEMO_STATE_DIR`. `deploy_server.sh` now does this
   preventively.
5. **`> out 2>&1` corrupts the bundle.** `ocrc` refuses rather than mix zip bytes
   with its own log. Use `> out.zip 2> log.txt` or `--quiet`.
6. **Re-submitting the same document is not a mistake.** The store is
   content-addressed; the second call is a fast cache lookup. Don't build your
   own cache.
7. **A different service on port 8000 answering 404 reads as "vLLM broken".**
   `deploy_server.sh` checks `/v1/models` for the exact model name; trust that
   over a bare "port answered".
8. **Binding `0.0.0.0` exposes the demo with no auth.** Default bind is
   `127.0.0.1` (behind SSH). Only set `DOTS_MOCR_BIND=0.0.0.0` on a trusted LAN.
9. **dots.mocr-svg is not worth touching.** See README. Use dots.mocr.
10. **transformers 4.x does not work.** Need 5.x with bfloat16; `check_local_env.py`
    enforces it.

## Making changes

- Branch → PR → merge into `main`. No separate CONTRIBUTING file yet.
- Inference code lives under `src/dots_mocr/`; the model weights and math are
  fixed — optimisations must preserve outputs. Regression guard:
  `tests/test_vision_flex_attention.py` and `tests/test_gpu_integration.py`
  (`-m gpu`).
- Run `python3 -m pytest tests -m "not gpu"` before pushing; the docker build
  is expensive and CI does not run it.
- Sign commits with a trailer so contributions are traceable:
  ```
  Agent: ZCode (GLM-5.2)
  ```
- After touching `demo/*.py`, restart the demo container on the server — the
  image bind-mounts the repo (`$REPO_ROOT:/opt/dots-mocr:ro`), so `git pull` +
  `docker restart dots_mocr_demo` is enough; no rebuild.

## Architecture pointer

For the model side (FLOPs, shapes, where the compute goes, regression
baselines): [docs/architecture.md](docs/architecture.md). For CUDA engineering
uplift: [docs/uplift/](docs/uplift/). This file intentionally does not duplicate
that material.
