// Flash-style attention kernels.
//
// Two shapes dominate this engine:
//
//  1. Vision tower (prefill): full (non-causal) attention over N_v patches,
//     N_v up to ~11k, head_dim=128, 12 heads. Naive materialises an N_v×N_v
//     score matrix per head (~3 GiB at N_v=11k) which is the OOM driver. We
//     tile Q in blocks of Br rows and stream K,V in blocks of Bc, keeping
//     partial softmax statistics on-chip (the online-softmax trick from
//     "FlashAttention"). Memory O(N_v), and the inner GEMV/GEMM goes through
//     shared memory with explicit accumulators.
//
//  2. LLM decode: T=1 query token attending to a growing KV-cache of length
//     `past`. This is a pure GEMV + softmax, memory-bound; we give it its own
//     fused kernel so there is one launch per layer instead of three.
//
//  3. LLM prefill: causal attention over S tokens, S up to ~3k. Same tiled
//     flash kernel as vision but with the causal mask baked into the score
//     block (skip the upper-triangle, scale the running max accordingly).
//
// The kernel is bf16-in, bf16-out, with the score accumulation and softmax
// in fp32 — exactly what HF's sdpa/flash backends do, and the parity target.
#pragma once

#include "tensor.h"

namespace dots {

// Flash attention for one attention segment, single batch (vision tower).
// q,k,v: [seq, n_heads, head_dim] bf16, contiguous in (seq, head, dim) order.
// out:   [seq, n_heads, head_dim] bf16.
// is_causal: true masks the upper triangle (LLM prefill); false is full
// attention (vision). cu_seqlens is for the multi-segment varlen case — pass
// {0, seq} for a single segment.
void flash_attention(const void* q, const void* k, const void* v, void* out,
                     int seq, int n_heads, int head_dim,
                     bool is_causal,
                     cudaStream_t stream = 0);

// LLM decode attention: a single query (T=1) attends to a KV-cache of length
// `past`. q: [n_heads, head_dim], k_cache/v_cache: [max_past, n_kv, head_dim]
// with the first `past` entries valid. GQA expansion (n_heads > n_kv) is done
// inside the kernel. out: [n_heads, head_dim].
void decode_attention(const void* q,
                      const void* k_cache, const void* v_cache,
                      int past, int n_heads, int n_kv_heads, int head_dim,
                      void* out, cudaStream_t stream = 0);

}  // namespace dots
