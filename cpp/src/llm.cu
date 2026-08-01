// Qwen2 LLM forward (prefill + decode). See llm.h for weight layout.
//
// Block math (pre-norm, GQA):
//   h = x + o_proj( attn( rope(q_proj(norm1(x))), k_cache, v_cache ) )
//   y = h + down_proj( swiglu( gate_proj(norm2(h)), up_proj(norm2(h)) ) )
// Prefill: T=S rows processed with flash_attention (causal), K/V written to the
//   cache at positions 0..S-1.
// Decode: T=1, single query attends to the growing cache via decode_attention.
#include "llm.h"
#include "kernels.h"
#include "tc_gemm.h"
#include "attention.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace dots {

// ---- KVCache ---------------------------------------------------------------
void KVCache::alloc(int n_layers, int max_seq_, int n_kv_heads_, int head_dim_, int device) {
    max_seq = max_seq_; n_kv_heads = n_kv_heads_; head_dim = head_dim_;
    current_len = 0;
    size_t per_layer = (size_t)max_seq * n_kv_heads * head_dim;
    size_t total = per_layer * n_layers * 2;  // K + V
    storage = DeviceBuffer(total * sizeof(bf16), device);
    bf16* base = (bf16*)storage.ptr;
    k.resize(n_layers); v.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        k[i] = base + (size_t)i * per_layer;
        v[i] = base + ((size_t)n_layers + i) * per_layer;
    }
}

// ---- weight loading --------------------------------------------------------
std::unique_ptr<LLM> LLM::load(const ModelWeights& w, const LLMConfig& cfg, int device) {
    auto llm = std::make_unique<LLM>();
    llm->cfg_ = cfg;
    llm->device_ = device;
    llm->w_ = std::make_unique<LLMWeights>();

    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int kv_dim = cfg.num_key_value_heads * cfg.head_dim();
    const int L = cfg.num_hidden_layers;

    size_t total = 0;
    total += (size_t)V * H;                 // embed
    total += H;                             // final norm
    total += (size_t)V * H;                 // lm_head (untied)
    for (int b = 0; b < L; ++b) {
        total += H;
        total += (size_t)H * H + H;         // q w,b
        total += (size_t)kv_dim * H + kv_dim; // k w,b
        total += (size_t)kv_dim * H + kv_dim; // v w,b
        total += (size_t)H * H;             // o
        total += H;                         // post_ln
        total += (size_t)I * H;             // gate
        total += (size_t)I * H;             // up
        total += (size_t)H * I;             // down
    }

    DeviceBuffer& S = llm->w_->storage;
    S = DeviceBuffer(total * sizeof(bf16), device);
    bf16* base = (bf16*)S.ptr;
    size_t off = 0;
    auto take = [&](size_t n) -> bf16* { bf16* p = base + off; off += n; return p; };

    llm->w_->embed = take((size_t)V * H);
    w.copy_to_device("model.embed_tokens.weight", llm->w_->embed, Dtype::BF16);
    llm->w_->norm_w = take(H);
    w.copy_to_device("model.norm.weight", llm->w_->norm_w, Dtype::BF16);
    llm->w_->lm_head = take((size_t)V * H);
    w.copy_to_device("lm_head.weight", llm->w_->lm_head, Dtype::BF16);

    char nm[160];
    for (int b = 0; b < L; ++b) {
        auto& B = llm->w_->blocks[b];
        B.in_ln_w = take(H);
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", b);
        w.copy_to_device(nm, B.in_ln_w, Dtype::BF16);

        B.q_w = take((size_t)H * H); B.q_b = take(H);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", b);
        w.copy_to_device(nm, B.q_w, Dtype::BF16);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.bias", b);
        w.copy_to_device(nm, B.q_b, Dtype::BF16);

        B.k_w = take((size_t)kv_dim * H); B.k_b = take(kv_dim);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", b);
        w.copy_to_device(nm, B.k_w, Dtype::BF16);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.bias", b);
        w.copy_to_device(nm, B.k_b, Dtype::BF16);

        B.v_w = take((size_t)kv_dim * H); B.v_b = take(kv_dim);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", b);
        w.copy_to_device(nm, B.v_w, Dtype::BF16);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.bias", b);
        w.copy_to_device(nm, B.v_b, Dtype::BF16);

        B.o_w = take((size_t)H * H);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", b);
        w.copy_to_device(nm, B.o_w, Dtype::BF16);

        B.post_ln_w = take(H);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", b);
        w.copy_to_device(nm, B.post_ln_w, Dtype::BF16);

        B.gate_w = take((size_t)I * H);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", b);
        w.copy_to_device(nm, B.gate_w, Dtype::BF16);
        B.up_w = take((size_t)I * H);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", b);
        w.copy_to_device(nm, B.up_w, Dtype::BF16);
        B.down_w = take((size_t)H * I);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", b);
        w.copy_to_device(nm, B.down_w, Dtype::BF16);
    }
    fprintf(stderr, "[llm] loaded %zu MiB weights, %d layers, V=%d\n",
            off * sizeof(bf16) / (1 << 20), L, V);
    return llm;
}

// ---- lm_head over a single row --------------------------------------------
// logits[V] = last_row @ lm_head^T. We copy the last hidden row to a [1,H]
// scratch and GEMM. logits_buf is bf16 [V].
static void compute_last_logits(const void* hidden, int S, int H, int V,
                                const bf16* lm_head_w,
                                DeviceTensor& last_row, DeviceTensor& logits) {
    copy_last_row(hidden, last_row.ptr(), S, H);
    tc_gemm(last_row.ptr(), false, lm_head_w, true,
            logits.ptr(), 1, V, H, Epilog::NONE);
}

// ============================================================================
// PREFILL
// ============================================================================
void LLM::prefill(const void* inputs_embeds, int S, KVCache& kv, void* logits_buf) {
    const LLMConfig& cfg = cfg_;
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int n_heads = cfg.num_attention_heads;
    const int n_kv = cfg.num_key_value_heads;
    const int hd = cfg.head_dim();
    const int L = cfg.num_hidden_layers;

    DOTS_CUDA_CHECK(cudaSetDevice(device_));

    // hidden activations (double-buffered across attention/mlp residuals).
    DeviceTensor hidden(Dtype::BF16, S, H);
    DOTS_CUDA_CHECK(cudaMemcpy(hidden.ptr(), inputs_embeds, hidden.nbytes(),
                               cudaMemcpyDeviceToDevice));

    // RoPE tables for positions [0, S).
    DeviceTensor rope_cos(Dtype::F32, S, hd / 2);
    DeviceTensor rope_sin(Dtype::F32, S, hd / 2);
    rope_table_llm((float*)rope_cos.ptr(), (float*)rope_sin.ptr(), S, hd, cfg.rope_theta);

    // Per-block scratch.
    DeviceTensor normed(Dtype::BF16, S, H);
    DeviceTensor q_full(Dtype::BF16, S, n_heads * hd);     // [S, 1536]
    DeviceTensor kv_proj(Dtype::BF16, S, n_kv * hd);       // [S, 256] reused for k and v
    DeviceTensor k_exp(Dtype::BF16, S, n_heads * hd);      // expanded for flash
    DeviceTensor v_exp(Dtype::BF16, S, n_heads * hd);
    DeviceTensor attn_out(Dtype::BF16, S, H);
    DeviceTensor proj_out(Dtype::BF16, S, H);
    DeviceTensor gate_buf(Dtype::BF16, S, I);
    DeviceTensor up_buf(Dtype::BF16, S, I);
    DeviceTensor mlp_hidden(Dtype::BF16, S, I);
    DeviceTensor mlp_out(Dtype::BF16, S, H);

    for (int b = 0; b < L; ++b) {
        const auto& B = w_->blocks[b];
        // ---- attention ----
        rms_norm(hidden.ptr(), B.in_ln_w, normed.ptr(), S, H, cfg.rms_norm_eps);
        // q/k/v with bias fused into the GEMM epilogue.
        tc_gemm(normed.ptr(), false, B.q_w, true, q_full.ptr(), S, H, H, Epilog::BIAS, B.q_b);
        // k -> cache
        tc_gemm(normed.ptr(), false, B.k_w, true, kv_proj.ptr(), S, n_kv * hd, H, Epilog::BIAS, B.k_b);
        write_kv_cache(kv.k[b], kv_proj.ptr(), 0, S, n_kv * hd, kv.max_seq);
        // v -> cache
        tc_gemm(normed.ptr(), false, B.v_w, true, kv_proj.ptr(), S, n_kv * hd, H, Epilog::BIAS, B.v_b);
        write_kv_cache(kv.v[b], kv_proj.ptr(), 0, S, n_kv * hd, kv.max_seq);
        // rope on q and the cached K (positions 0..S-1)
        apply_rope_llm(q_full.ptr(), kv.k[b], S, n_heads, n_kv, hd,
                       (const float*)rope_cos.ptr(), (const float*)rope_sin.ptr());
        // expand KV cache GQA -> full n_heads heads for the flash kernel
        expand_kv_gqa(kv.k[b], k_exp.ptr(), S, n_kv, hd, n_heads);
        expand_kv_gqa(kv.v[b], v_exp.ptr(), S, n_kv, hd, n_heads);
        // q layout is [S, n_heads, hd]? q_proj gave [S, H=n_heads*hd] which IS
        // [S, n_heads, hd] row-major (head axis inside). Good.
        flash_attention(q_full.ptr(), k_exp.ptr(), v_exp.ptr(), attn_out.ptr(),
                        S, n_heads, hd, /*is_causal=*/true);
        tc_gemm(attn_out.ptr(), false, B.o_w, true, proj_out.ptr(), S, H, H, Epilog::NONE);
        add(hidden.ptr(), proj_out.ptr(), hidden.ptr(), (size_t)S * H);

        // ---- mlp ----
        rms_norm(hidden.ptr(), B.post_ln_w, normed.ptr(), S, H, cfg.rms_norm_eps);
        tc_gemm(normed.ptr(), false, B.gate_w, true, gate_buf.ptr(), S, I, H, Epilog::NONE);
        tc_gemm(normed.ptr(), false, B.up_w, true, up_buf.ptr(), S, I, H, Epilog::NONE);
        silu_inplace(gate_buf.ptr(), (size_t)S * I);
        swiglu(gate_buf.ptr(), up_buf.ptr(), mlp_hidden.ptr(), (size_t)S * I);
        tc_gemm(mlp_hidden.ptr(), false, B.down_w, true, mlp_out.ptr(), S, H, I, Epilog::NONE);
        add(hidden.ptr(), mlp_out.ptr(), hidden.ptr(), (size_t)S * H);
    }

    // final norm + lm_head on the last row.
    rms_norm(hidden.ptr(), w_->norm_w, normed.ptr(), S, H, cfg.rms_norm_eps);
    DeviceTensor last_row(Dtype::BF16, 1, H);
    DeviceTensor logits(Dtype::BF16, 1, V);
    compute_last_logits(normed.ptr(), S, H, V, w_->lm_head, last_row, logits);
    DOTS_CUDA_CHECK(cudaMemcpy(logits_buf, logits.ptr(),
                               (size_t)V * sizeof(bf16), cudaMemcpyDeviceToDevice));
    kv.current_len = S;
}

// ============================================================================
// DECODE (one token)
// ============================================================================
void LLM::decode_step(const void* embed_1h, KVCache& kv, void* logits_buf) {
    const LLMConfig& cfg = cfg_;
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int n_heads = cfg.num_attention_heads;
    const int n_kv = cfg.num_key_value_heads;
    const int hd = cfg.head_dim();
    const int L = cfg.num_hidden_layers;
    const int pos = kv.current_len;   // position of the new token

    DOTS_CUDA_CHECK(cudaSetDevice(device_));

    // hidden[1, H]
    DeviceTensor hidden(Dtype::BF16, 1, H);
    DOTS_CUDA_CHECK(cudaMemcpy(hidden.ptr(), embed_1h, (size_t)H * sizeof(bf16),
                               cudaMemcpyDeviceToDevice));

    // rope tables for positions [0, pos+1): apply_rope_llm indexes by absolute
    // position, so the table must cover up to and including the current `pos`.
    // Allocating per-step is cheap (a few KB) and avoids the prefill-sized table.
    DeviceTensor rope_cos(Dtype::F32, pos + 1, hd / 2);
    DeviceTensor rope_sin(Dtype::F32, pos + 1, hd / 2);
    rope_table_llm((float*)rope_cos.ptr(), (float*)rope_sin.ptr(), pos + 1, hd, cfg.rope_theta);

    DeviceTensor normed(Dtype::BF16, 1, H);
    DeviceTensor q_full(Dtype::BF16, 1, n_heads * hd);
    DeviceTensor kv_proj(Dtype::BF16, 1, n_kv * hd);
    DeviceTensor attn_out(Dtype::BF16, 1, H);
    DeviceTensor proj_out(Dtype::BF16, 1, H);
    DeviceTensor gate_buf(Dtype::BF16, 1, I);
    DeviceTensor up_buf(Dtype::BF16, 1, I);
    DeviceTensor mlp_hidden(Dtype::BF16, 1, I);
    DeviceTensor mlp_out(Dtype::BF16, 1, H);

    for (int b = 0; b < L; ++b) {
        const auto& B = w_->blocks[b];
        rms_norm(hidden.ptr(), B.in_ln_w, normed.ptr(), 1, H, cfg.rms_norm_eps);
        // q/k/v with bias fused into the GEMM epilogue.
        tc_gemm(normed.ptr(), false, B.q_w, true, q_full.ptr(), 1, H, H, Epilog::BIAS, B.q_b);
        // k,v -> cache at row `pos`
        tc_gemm(normed.ptr(), false, B.k_w, true, kv_proj.ptr(), 1, n_kv * hd, H, Epilog::BIAS, B.k_b);
        write_kv_cache(kv.k[b], kv_proj.ptr(), pos, 1, n_kv * hd, kv.max_seq);
        tc_gemm(normed.ptr(), false, B.v_w, true, kv_proj.ptr(), 1, n_kv * hd, H, Epilog::BIAS, B.v_b);
        write_kv_cache(kv.v[b], kv_proj.ptr(), pos, 1, n_kv * hd, kv.max_seq);
        // rope ONLY the new q and the new k row (positions 0..pos-1 were roped
        // in their own decode step; re-rope would be wrong).
        apply_rope_llm(q_full.ptr(), kv.k[b] + (size_t)pos * n_kv * hd,
                       1, n_heads, n_kv, hd,
                       (const float*)rope_cos.ptr() + pos * (hd / 2),
                       (const float*)rope_sin.ptr() + pos * (hd / 2));
        // single-query attention over [0, pos+1) cache.
        decode_attention(q_full.ptr(), kv.k[b], kv.v[b],
                         pos + 1, n_heads, n_kv, hd, attn_out.ptr());
        tc_gemm(attn_out.ptr(), false, B.o_w, true, proj_out.ptr(), 1, H, H, Epilog::NONE);
        add(hidden.ptr(), proj_out.ptr(), hidden.ptr(), (size_t)H);

        // mlp
        rms_norm(hidden.ptr(), B.post_ln_w, normed.ptr(), 1, H, cfg.rms_norm_eps);
        tc_gemm(normed.ptr(), false, B.gate_w, true, gate_buf.ptr(), 1, I, H, Epilog::NONE);
        tc_gemm(normed.ptr(), false, B.up_w, true, up_buf.ptr(), 1, I, H, Epilog::NONE);
        silu_inplace(gate_buf.ptr(), (size_t)I);
        swiglu(gate_buf.ptr(), up_buf.ptr(), mlp_hidden.ptr(), (size_t)I);
        tc_gemm(mlp_hidden.ptr(), false, B.down_w, true, mlp_out.ptr(), 1, H, I, Epilog::NONE);
        add(hidden.ptr(), mlp_out.ptr(), hidden.ptr(), (size_t)H);
    }

    rms_norm(hidden.ptr(), w_->norm_w, normed.ptr(), 1, H, cfg.rms_norm_eps);
    DeviceTensor last_row(Dtype::BF16, 1, H);
    DeviceTensor logits(Dtype::BF16, 1, V);
    compute_last_logits(normed.ptr(), 1, H, V, w_->lm_head, last_row, logits);
    DOTS_CUDA_CHECK(cudaMemcpy(logits_buf, logits.ptr(),
                               (size_t)V * sizeof(bf16), cudaMemcpyDeviceToDevice));
    kv.current_len = pos + 1;
}

}  // namespace dots
