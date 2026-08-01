"""In-process (transformers) GLM-OCR parser — the no-vLLM path.

Why this exists: GLM-OCR needs a current vLLM (>=0.19) whose C runtime is built
for CUDA 13, but the demo's 4090 host runs a driver/container-toolkit that won't
load cu13 images and won't run a hand-built cu126 vLLM (its `_C_stable_libtorch`
is still cu13). The model card's own path — transformers >= 5.1 with
`AutoModelForImageTextToText` — works on the same cu126 stack dots.mocr's
transformers engine already runs on (bench-cu126: torch 2.11+cu126,
transformers 5.5.4, both proven to init CUDA here).

So this is the GLM-OCR counterpart of DotsMOCRParser for the `transformers`
engine: same GenerationStats / abort / artifact plumbing (inherited), only
loading and generation differ — GLM-OCR is a native transformers architecture
(no remote code), and its chat template is applied through apply_chat_template
with the image as a content part (no qwen_vl_utils).
"""

import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import threading

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from dots_mocr.cli import DotsMOCRParser
from dots_mocr.utils.generation_stats import GenerationStats


class GlmOcrTransformersParser(DotsMOCRParser):
    """DotsMOCRParser loading GLM-OCR natively (no dots transformers_patch)."""

    # The parent's _load_model calls register_transformers() (the dots.mocr
    # remote-code patch) and reads a dots_ocr config — wrong for GLM-OCR, which
    # is a native `glm_ocr` architecture resolved straight by transformers.
    def _load_model(self, ckpt):
        # device/dtype/attn are already resolved by the parent __init__.
        device_map = self._resolve_device_map()
        print(f"[glm] using device_map: {device_map}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt,
            torch_dtype=self.dtype,
            device_map=device_map,
            local_files_only=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(ckpt, local_files_only=True)
        print(f"[glm] GLM-OCR loaded from {ckpt}, device={self.device}, "
              f"dtype={self.dtype}")

    def _resolve_dtype(self, dtype):
        # GLM-OCR ships bf16; match dots.mocr's resolution so the demo's
        # dtype selector and the worker's "bfloat16 on cuda" default still apply.
        if dtype == "auto":
            is_cuda = isinstance(self.device, str) and self.device.startswith("cuda")
            return torch.bfloat16 if is_cuda else torch.float32
        return getattr(torch, dtype)

    def _inference(self, image, prompt, temperature=None, stats=None):
        """One generate() call, mirroring DotsMOCRParser._inference's shape.

        Differences from the parent: GLM-OCR's processor takes the image as a
        content 'url'/'image' part and applies its own chat template (which
        injects <|begin_of_image|>); there is no qwen_vl_utils step and no
        mm_token_type_ids to pop beyond token_type_ids.
        """
        temperature = self.temperature if temperature is None else temperature
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        inputs.pop("token_type_ids", None)

        generation_kwargs = {"max_new_tokens": self.max_completion_tokens}
        if temperature > 0:
            generation_kwargs.update(do_sample=True, temperature=temperature,
                                     top_p=self.top_p)

        abort_event = self.abort_event
        if abort_event is not None or stats is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _StepCriteria(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    if stats is not None:
                        stats.record_token()
                    return abort_event is not None and abort_event.is_set()

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([_StepCriteria()])

        with self._generate_lock, torch.inference_mode():
            if stats is not None:
                stats.start(prompt_tokens=int(inputs.input_ids.shape[-1]))
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[-1]:]
        if stats is not None:
            stats.finish(
                generated_tokens=int(generated_ids_trimmed.shape[-1]),
                aborted=abort_event is not None and abort_event.is_set(),
            )
        # skip_special_tokens=True gives clean markdown/LaTeX for the demo's
        # renderer (the card uses False only to inspect raw tokens).
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
