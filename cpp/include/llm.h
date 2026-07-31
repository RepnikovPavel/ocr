// Qwen2 LLM decoder (DotsOCRForCausalLM body): 28 pre-norm transformer blocks
// with GQA (12 Q heads, 2 KV heads) + lm_head.
//
// Two execution modes share the same weights:
//   - prefill: process the whole prompt [S, hidden] in one pass, fill the
//     KV-cache, return the last-position logits;
//   - decode : one new token id -> embed -> 28 blocks (single query against the
//     growing KV-cache) -> lm_head -> greedy argmax.
//
// Weight layout (safetensors, bf16; q/k/v have bias, o_proj/mlp have none):
//   model.embed_tokens.weight                    : [V, 1536]
//   model.layers.{0..27}.input_layernorm.weight  : [1536]
//   model.layers.{0..27}.self_attn.{q,k,v}_proj.{weight,bias} : q/k/v [1536,1536] / [256,1536]
//   model.layers.{0..27}.self_attn.o_proj.weight : [1536, 1536]
//   model.layers.{0..27}.post_attention_layernorm.weight : [1536]
//   model.layers.{0..27}.mlp.gate_proj.weight    : [8960, 1536]
//   model.layers.{0..27}.mlp.up_proj.weight      : [8960, 1536]
//   model.layers.{0..27}.mlp.down_proj.weight    : [1536, 8960]
//   model.norm.weight                            : [1536]
//   lm_head.weight                               : [V, 1536]   (untied)
//
// GQA: k/v are projected to [S, n_kv=2, 128]; in attention each KV head is
// shared by g=6 query heads. The decode kernel takes n_heads/n_kv_heads and
// handles the expansion internally.
#pragma once

#include "config.h"
#include "safetensors_loader.h"
#include "tensor.h"

#include <memory>
#include <vector>

namespace dots {

struct LLMWeights {
    bf16* embed = nullptr;          // [V, H]
    bf16* norm_w = nullptr;         // [H]
    bf16* lm_head = nullptr;        // [V, H]

    static constexpr int NUM_BLOCKS = 28;
    struct Block {
        bf16* in_ln_w;              // [H]
        bf16* q_w; bf16* q_b;       // [H,H], [H]
        bf16* k_w; bf16* k_b;       // [kv_dim,H], [kv_dim]   kv_dim = n_kv*hd
        bf16* v_w; bf16* v_b;
        bf16* o_w;                  // [H,H]
        bf16* post_ln_w;            // [H]
        bf16* gate_w;               // [I,H]
        bf16* up_w;                 // [I,H]
        bf16* down_w;               // [H,I]
    };
    Block blocks[NUM_BLOCKS];

    DeviceBuffer storage;
};

// Per-request KV-cache: one buffer per layer for K and V, capacity-bounded.
struct KVCache {
    int max_seq = 0;       // capacity
    int n_kv_heads = 0;
    int head_dim = 0;
    // Per layer: K[layer] and V[layer], each [max_seq, n_kv, head_dim] bf16.
    std::vector<bf16*> k;
    std::vector<bf16*> v;
    DeviceBuffer storage;
    int current_len = 0;   // number of valid positions written so far

    void alloc(int n_layers, int max_seq_, int n_kv_heads_, int head_dim_, int device);
};

class LLM {
public:
    static std::unique_ptr<LLM> load(const ModelWeights& w, const LLMConfig& cfg,
                                     int device = 0);

    // Prefill: inputs_embeds [S, H] (already has vision scattered in). Fills
    // kv_cache from position 0, returns the last-row logits [V] bf16 on device.
    // logits_buf must hold vocab bf16.
    void prefill(const void* inputs_embeds, int S, KVCache& kv,
                 void* logits_buf);

    // Decode one token: embedding [1,H] -> forward -> logits [V].
    void decode_step(const void* embed_1h, KVCache& kv, void* logits_buf);

    const LLMConfig& config() const { return cfg_; }
    LLMWeights& weights() { return *w_; }

private:
    LLMConfig cfg_;
    std::unique_ptr<LLMWeights> w_;
    int device_ = 0;
};

}  // namespace dots
