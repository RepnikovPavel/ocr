// CUDA kernel declarations for the dots.mocr inference engine.
//
// Conventions:
//   * weights and activations are bf16 (the checkpoint dtype) unless noted;
//     reductions that need range (RMSNorm variance, softmax, rope cos/sin)
//     run in fp32 and round back to bf16 at the end.
//   * every kernel takes explicit shapes/strides — no tensor objects in the
//     hot path, so the compiler sees raw pointers and we control the launch.
//   * GEMM (qkv, o_proj, mlp up/down/gate, lm_head, patch conv-as-gemm,
//     merger) goes through cuBLAS bf16 (cublasGemmEx with CUBLAS_COMPUTE_32F
//     and bf16 I/O), which selects tensor-core kernels on Ada/Blackwell.
#pragma once

#include "tensor.h"

#include <cstdint>

namespace dots {

// ---- cuBLAS handle ----------------------------------------------------------
struct cublasHandle;
cublasHandle* cublas_handle_get();
void          cublas_handle_destroy();

// cuBLAS bf16 GEMM:  C = alpha * op(A) * op(B) + beta * C
//   A: [M, K] if transA=false else [K, M]   (bf16)
//   B: [K, N] if transB=false else [N, K]   (bf16)
//   C: [M, N]                                (bf16)
// cuBLAS is column-major; we keep row-major tensors and flip the call so the
// math is identical (C^T = B^T A^T). transA/transB refer to the row-major view.
void cublas_bf16_gemm(const void* A, bool transA,
                      const void* B, bool transB,
                      void* C,
                      int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f,
                      int lda = 0, int ldb = 0, int ldc = 0);

// Batched GEMM for grouped-qkv-style calls (used by nothing currently, kept
// for the attention-block fused path if we add it).
void cublas_bf16_gemm_batched(const void* const* A, bool transA,
                              const void* const* B, bool transB,
                              void* const* C,
                              int M, int N, int K, int batch);

// ---- elementwise / fused kernels -------------------------------------------

// RMSNorm over the last `dim` features, one row at a time. Matches HF:
//   out = (x * rsqrt(mean(x^2) + eps)) * weight   [all in fp32, out in bf16]
void rms_norm(const void* x, const void* weight, void* out,
              int rows, int dim, float eps);

// LayerNorm (used by the vision PatchMerger's ln_q, with eps=1e-6 and bias).
void layer_norm(const void* x, const void* weight, const void* bias,
                void* out, int rows, int dim, float eps);

// SiLU(x) = x * sigmoid(x). In-place on bf16.
void silu_inplace(void* x, int n);

// SwiGLU activation: out = silu(gate) * up. Both inputs bf16 [rows, dim].
void swiglu(const void* gate, const void* up, void* out, int n);

// GELU tanh-approx (vision merger MLP uses nn.GELU() default='none' i.e. exact
// erf form). Qwen2.5-VL merger uses GELU; we implement erf-based GELU.
void gelu(void* x, int n);

// Embedding lookup: gather rows of the embedding table by id, then scatter
// vision embeddings into the img-pad positions. This is the `prepare_inputs_
// embeds` step: text tokens -> embed table rows; <|imgpad|> slots -> vision vecs.
//   ids:     [seq] int32
//   table:   [vocab, hidden] bf16
//   out:     [seq, hidden] bf16
//   img_mask:[seq] bool/int8 (1 => this row is an image slot)
//   ve:      [num_img, hidden] bf16 vision embeddings
void embed_and_scatter(const int* ids, const void* table, void* out,
                       int seq, int hidden,
                       const int8_t* img_mask, const void* ve, int num_img);

// Embedding lookup only (decode step uses a single id).
void embed_lookup(const int* ids, const void* table, void* out,
                  int seq, int hidden);

// Argmax over the last dim -> one int32 id (the greedy next token).
int argmax_last(const void* logits, int vocab);

// Copy the last row of a [seq, hidden] tensor into out[hidden].
void copy_last_row(const void* x, void* out, int seq, int hidden);

// Add residual: y = a + b (bf16), elementwise, n elements.
void add(const void* a, const void* b, void* y, int n);

// Add a per-row bias vector to a [rows, dim] bf16 tensor (Linear bias).
void add_row_bias(void* x, const void* bias, int rows, int dim);

// Cast f32 -> bf16 on device.
void cast_f32_to_bf16(const void* src_f32, void* dst_bf16, int n);
// Cast bf16 -> f32 on device.
void cast_bf16_to_f32(const void* src_bf16, void* dst_f32, int n);

// Split a fused qkv tensor [N_v, 3*E] (where E = n_heads*head_dim) into three
// [N_v, n_heads, head_dim] buffers. The fused layout from qkv_weight*normed is
// [..., 3, n_heads, head_dim] interleaved per-token (PyTorch reshape(3,H,HD) on
// the trailing 3*E). Output q,k,v are contiguous [N_v, H, HD].
void split_qkv(const void* qkv, void* q, void* k, void* v, int N, int n_heads, int head_dim);

// Vision qkv split variant: identical layout, kept as alias.
inline void split_qkv_vision(const void* qkv, void* q, void* k, void* v,
                             int N, int n_heads, int head_dim) {
    split_qkv(qkv, q, k, v, N, n_heads, head_dim);
}

// Copy a freshly-projected KV slice [T, kv_dim] into the per-layer KV-cache at
// rows [pos_start, pos_start+T). kv_dim = n_kv_heads*head_dim.
void write_kv_cache(void* cache, const void* src, int pos_start, int T,
                    int kv_dim, int cache_max_seq);

// Expand the GQA KV cache [seq, n_kv, hd] -> [seq, n_heads, hd] by repeating
// each kv head g=n_heads/n_kv times. Used before flash_attention on the prefill
// path (flash kernel takes n_heads heads for q,k,v uniformly).
void expand_kv_gqa(const void* kv_src, void* dst, int seq, int n_kv_heads,
                   int head_dim, int n_heads);

// ---- RoPE -------------------------------------------------------------------

// LLM RoPE (NEOX rotate-half, full head_dim applied, theta=1e6).
// Applies rotary embedding to q and k of one forward pass. The position ids
// come from the standard 0..S-1 sequence (Qwen2 with no rope_scaling).
//   q,k:      [seq, n_heads, head_dim]   (the Q and K after the q/k projection)
//   cos/sin:  precomputed [seq, head_dim]
void apply_rope_llm(void* q, void* k, int seq, int n_heads, int n_kv_heads,
                    int head_dim, const float* cos, const float* sin);

// Precompute cos/sin tables for the LLM for positions [0, seq).
//   table: [seq, head_dim] fp32, layout [pos, d] with the NEOX pairing.
void rope_table_llm(float* cos, float* sin, int seq, int head_dim, float theta,
                    cudaStream_t stream = 0);

// Vision 2D-RoPE. The rotary table is [N_v, 64] (half of head_dim=128) built
// from 2D grid positions; the apply_rotary_pos_emb_vision kernel expands it to
// the full head_dim by repeating the cos/sin block twice (see modeling code:
// cos = cos.unsqueeze(1).repeat(1,1,2)).
//   q,k:    [N_v, n_heads, head_dim]
//   freqs:  [N_v, head_dim/2] precomputed (cos+sin fused as the freq table)
void apply_rotary_vision(void* q, void* k, int N_v, int n_heads, int head_dim,
                         const float* freqs);

// ---- attention (flash-style) ------------------------------------------------
// Declared in attention.cu; see attention.h for the full surface.

}  // namespace dots
