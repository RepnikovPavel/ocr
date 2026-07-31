#!/usr/bin/env python3
# Capture reference greedy ids by running the HF model's forward() directly,
# one step at a time (prefill then decode), instead of model.generate() —
# generate() breaks under transformers 5.5.4 because the checkpoint's
# prepare_inputs_for_generation assumes an older cache_position API. The manual
# loop reproduces exactly what the C++ engine does, so the ids are a fair
# parity target.
#
#   docker exec dots_mocr_demo python3 /opt/dots-mocr/cpp/tools/capture_reference_manual.py \
#       /state/ref 16
# (assumes capture_inputs.py already wrote pixel_values.bin + input_ids.txt +
# grid_thw.txt into the same dir)
import os, sys, json
import numpy as np
import torch

def main():
    ref_dir = sys.argv[1]
    max_new  = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    ids = list(map(int, open(os.path.join(ref_dir, "input_ids.txt")).read().split()))
    gt = list(map(int, open(os.path.join(ref_dir, "grid_thw.txt")).read().split()))
    Nv = gt[0]*gt[1]*gt[2]
    pv = np.fromfile(os.path.join(ref_dir, "pixel_values.bin"), dtype=np.float32).reshape(Nv, 588)

    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
    cfg = AutoConfig.from_pretrained("/models", local_files_only=True, trust_remote_code=True)
    cfg.vision_config.attn_implementation = "sdpa"
    use_cuda = torch.cuda.is_available()
    dev = "cuda:0" if use_cuda else "cpu"
    # Load in bf16: the vision forward unconditionally casts input to bf16, so
    # an fp32 load just creates a dtype mismatch at the conv.
    model = AutoModelForCausalLM.from_pretrained(
        "/models", config=cfg, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, device_map={"": dev},
        local_files_only=True, trust_remote_code=True)
    # The checkpoint stores the patch-embed conv bias as fp32 while the conv
    # weight and the rest of the model load as bf16; the vision forward casts
    # pixel_values to the weight dtype (bf16) and then chokes on the fp32 bias.
    # Align the bias to the weight dtype before running.
    # The checkpoint's vision forward forces `hidden_states.bfloat16()`
    # unconditionally (modeling_dots_vision.py forward(bf16=True)), so the
    # patch-embed Conv2d always sees bf16 input. The conv bias loads as fp32
    # while the weight is bf16 — align the bias so the conv doesn't reject it.
    # Keep everything in bf16 (the model's native dtype).
    mdtype = torch.bfloat16
    model = model.to(torch.bfloat16)
    pe = model.vision_tower.patch_embed.patchifier.proj
    pe.bias.data = pe.bias.data.to(pe.weight.dtype)
    for p in model.vision_tower.parameters():
        if p.dtype != mdtype:
            p.data = p.data.to(mdtype)
    model.eval()

    # The vision RoPE inv_freq buffer is non-persistent and loads as garbage
    # (zeros/denormals); recompute it the way VisionRotaryEmbedding.__init__
    # does so the reference actually applies rotary embedding. Without this the
    # captured ids would be from a broken (no-RoPE) forward.
    vr = model.vision_tower.rotary_pos_emb
    half = vr.inv_freq.shape[0]            # 32
    dim = half * 2                          # 64
    theta = getattr(vr, "theta", 10000.0)
    with torch.no_grad():
        vr.inv_freq.copy_(1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=vr.inv_freq.device) / dim)))
    print("inv_freq[:4]:", model.vision_tower.rotary_pos_emb.inv_freq[:4].tolist(), flush=True)

    input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
    pixel_values = torch.tensor(pv, dtype=mdtype, device=dev)
    grid_thw = torch.tensor([gt], dtype=torch.long, device=dev)
    EOS = {151643, 151672, 151673}

    # Prefill: run prepare_inputs_embeds + decoder with use_cache=True.
    with torch.inference_mode():
        img_mask = (input_ids == cfg.image_token_id)
        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            grid_thw=grid_thw, img_mask=img_mask)
        out = model.model(
            inputs_embeds=inputs_embeds, use_cache=True,
            return_dict=True)
        hidden = out.last_hidden_state
        past = out.past_key_values
        logits = model.lm_head(hidden[:, -1:])
        next_id = int(logits[0].argmax(-1).item())

    gen = []
    p = AutoProcessor.from_pretrained("/models", local_files_only=True, trust_remote_code=True)
    while next_id not in EOS and len(gen) < max_new:
        gen.append(next_id)
        # decode step: embed one token, feed with past KV
        emb = model.model.embed_tokens(torch.tensor([[next_id]], device=dev))
        with torch.inference_mode():
            out = model.model(inputs_embeds=emb, past_key_values=past,
                              use_cache=True, return_dict=True)
            past = out.past_key_values
            logits = model.lm_head(out.last_hidden_state[:, -1:])
            next_id = int(logits[0].argmax(-1).item())
    if next_id not in EOS:
        gen.append(next_id)

    with open(os.path.join(ref_dir, "ref_ids.txt"), "w") as f:
        for t in gen: f.write("%d\n" % t)
    print("captured %d ref ids: %s" % (len(gen), gen[:20]), flush=True)
    print("ref text:", p.batch_decode([gen], skip_special_tokens=True)[0][:200], flush=True)

if __name__ == "__main__":
    main()
