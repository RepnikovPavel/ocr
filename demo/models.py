"""Registry of the models the demo can switch between at runtime.

The demo used to be one model per server instance (dots.mocr or dots.mocr-svg,
picked by `DEMO_VARIANT`). The model selector widens that to several models
served side by side, so a human can parse the *same* document with each and
eyeball the quality / speed difference in one tab.

Each model is a hot vLLM server on its own GPU (on a 2x4090 host: dots.mocr on
GPU0, GLM-OCR on GPU1), and switching the selector in the UI just repoints the
worker at the other server — no weight reload, no VRAM churn. The two servers
serve different prompt-mode namespaces (dots.mocr's layout skills vs GLM-OCR's
Text/Formula/Table triggers), so the demo's prompt-mode dropdown and the
docstore cache key both keep the answers apart.

The config is read from the environment so the same image runs on a laptop
(one vLLM) and the server (two). Anything unset falls back to the single-model
defaults the worker already had, so a deploy that does not set the new vars
keeps behaving exactly as before.
"""

import os


# dots.mocr's seven layout skills, as in the `mocr` variant of demo/server.py.
# Kept here (not imported from server.py) to avoid a demo<->demo import cycle:
# server.py builds its prompt-mode list from this registry now.
DOTS_PROMPT_MODES = [
    "prompt_layout_all_en",
    "prompt_layout_only_en",
    "prompt_ocr",
    "prompt_grounding_ocr",
    "prompt_web_parsing",
    "prompt_scene_spotting",
    "prompt_general",
]

# GLM-OCR's cue-driven modes. They are NOT aliases of dots.mocr's skills —
# GLM-OCR does not emit layout JSON with bboxes, only recognition output — so
# they live in their own namespace and the docstore key stays unambiguous.
GLM_PROMPT_MODES = [
    "glm_text_recognition",
    "glm_formula_recognition",
    "glm_table_recognition",
    "glm_general",
]


def _modes_with_defaults(modes):
    """Attach the per-mode UI metadata the demo already expects.

    default_temperature / page_seconds_estimate were hard-coded tables in
    demo/worker.py; for the selector we surface them per model so each model's
    dropdown shows its own priors. GLM-OCR is far faster per page (0.9B +
    recognition-only, no layout JSON post-processing), hence the lower numbers.
    """
    from demo.worker import PAGE_SECONDS_ESTIMATE, default_temperature

    return [
        {
            "mode": mode,
            "default_temperature": default_temperature(mode),
            "page_seconds_estimate": PAGE_SECONDS_ESTIMATE.get(mode, 45),
        }
        for mode in modes
    ]


def _build_models():
    """The model registry, as a stable {id: spec} dict.

    `parser_factory` is a lazy import + construct callable so importing this
    module never pulls torch/vllm — the worker only imports the chosen one.
    """
    def _dots_factory(url, model, common):
        from dots_mocr.model.vllm_parser import VllmDotsMOCRParser

        return VllmDotsMOCRParser(
            vllm_url=url, vllm_model=model, device="vllm", dtype="auto", **common,
        )

    def _glm_factory(url, model, common):
        from dots_mocr.model.glm_parser import GlmOcrVllmParser

        return GlmOcrVllmParser(
            vllm_url=url, vllm_model=model, device="vllm", dtype="auto", **common,
        )

    return {
        "dots_mocr": {
            "label": "dots.mocr (3.0B)",
            "engine": "vllm",
            "vllm_url": os.environ.get("DEMO_VLLM_URL_DOTS") or os.environ.get("DEMO_VLLM_URL") or "http://127.0.0.1:8000/v1",
            "vllm_model": os.environ.get("DEMO_VLLM_MODEL_DOTS") or os.environ.get("DEMO_VLLM_MODEL") or "rednote-hilab/dots.mocr",
            "vllm_container": os.environ.get("DEMO_VLLM_CONTAINER_DOTS") or os.environ.get("DEMO_VLLM_CONTAINER"),
            "prompt_modes": DOTS_PROMPT_MODES,
            "default_mode": "prompt_layout_all_en",
            "parser_factory": _dots_factory,
        },
        "glm_ocr": {
            "label": "GLM-OCR (0.9B)",
            "engine": "vllm",
            "vllm_url": os.environ.get("DEMO_VLLM_URL_GLM") or "http://127.0.0.1:8001/v1",
            "vllm_model": os.environ.get("DEMO_VLLM_MODEL_GLM") or "glm-ocr",
            "vllm_container": os.environ.get("DEMO_VLLM_CONTAINER_GLM") or "glm_vllm",
            "prompt_modes": GLM_PROMPT_MODES,
            "default_mode": "glm_text_recognition",
            "parser_factory": _glm_factory,
        },
    }


def get_models():
    """Return the registry, re-read each call so env overrides take effect.

    Rebuilding is cheap (no imports until a factory runs) and lets the worker
    pick up a config change in the same process, which the deploy scripts rely
    on when they exec the server with a fresh environment.
    """
    return _build_models()


def get_model(model_id):
    """Look up one spec, or None. The caller (worker.select_model) validates."""
    return get_models().get(model_id)


def default_model_id():
    """Startup model, overridable by DEMO_MODEL; dots.mocr keeps old behavior."""
    return os.environ.get("DEMO_MODEL", "dots_mocr")


def models_public():
    """The subset /api/state exposes to the browser: id + label + modes.

    Parser factories and vLLM internals are server-side only — the UI needs the
    label for the dropdown and the prompt-mode list to redraw the skills row.
    """
    return [
        {
            "id": mid,
            "label": spec["label"],
            "prompt_modes": _modes_with_defaults(spec["prompt_modes"]),
            "default_mode": spec["default_mode"],
        }
        for mid, spec in get_models().items()
    ]
