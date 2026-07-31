// Minimal loader test: open a checkpoint dir, list a few known tensors.
#include "safetensors_loader.h"
#include <cassert>
#include <cstdio>
#include <string>

using namespace dots;

int main(int argc, char** argv) {
    std::string ckpt = argc > 1 ? argv[1] : ".";
    ModelWeights w = ModelWeights::load(ckpt);
    if (w.tensors.empty()) { fprintf(stderr, "skip: no checkpoint at %s\n", ckpt.c_str()); return 77; }

    // Spot-check a handful of tensors the model needs.
    assert(w.has("lm_head.weight"));
    assert(w.has("model.embed_tokens.weight"));
    assert(w.has("model.norm.weight"));
    assert(w.has("vision_tower.merger.mlp.0.weight"));
    assert(w.has("vision_tower.blocks.0.attn.qkv.weight"));
    assert(w.has("model.layers.0.self_attn.q_proj.weight"));
    assert(w.has("model.layers.0.self_attn.q_proj.bias"));

    // embed shape must be [V, H] = [151936, 1536].
    auto s = w.shape("model.embed_tokens.weight");
    assert(s.size() == 2 && s[0] == 151936 && s[1] == 1536);

    printf("OK loader: %zu tensors, embed=[%d,%d]\n", w.tensors.size(), s[0], s[1]);
    return 0;
}
