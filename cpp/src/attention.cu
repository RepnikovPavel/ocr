// Flash-style attention kernels. See attention.h for the design notes.
//
// This version is optimised for *latency of the first end-to-end run* rather
// than peak throughput: each (query row, head) is handled by one warp that
// streams over all keys, computing scores and the weighted V sum in registers
// with a warp-shuffle dot product. The online-softmax keeps memory O(seq)
// instead of O(seq^2) — the same property that lets vision attention fit at
// N_v ~ 11k. The prefill (causal) and vision (full) paths share the kernel;
// only the per-pair validity test differs.
#include "attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace dots {

// One warp per (q_row, head). Each lane owns head_dim/32 = 4 elements of the
// dot product. The warp loops over keys, accumulates an online softmax and a
// running weighted-V vector (in registers, 4 floats/lane over head_dim=128).
template <int HEAD_DIM>
__global__ void flash_attn_kernel(const bf16* __restrict__ q,
                                  const bf16* __restrict__ k,
                                  const bf16* __restrict__ v,
                                  bf16* __restrict__ out,
                                  int seq, int n_heads, int head_dim,
                                  bool is_causal) {
    (void)head_dim;
    constexpr int LANES = 32;
    constexpr int PER = HEAD_DIM / LANES;     // 4 for hd=128
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

    int q_row = blockIdx.x;
    int head  = blockIdx.y;
    if (q_row >= seq || head >= n_heads) return;

    int lane = threadIdx.x;   // 0..31 within the single warp (blockDim.x == 32)
    const bf16* q_row_ptr = q + ((size_t)q_row * n_heads + head) * HEAD_DIM;
    // Load this lane's slice of q into registers.
    float qv[PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p) qv[p] = __bfloat162float(q_row_ptr[lane * PER + p]);

    // Running softmax + weighted V accumulator (this lane's 4-dim slice).
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float acc[PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;

    for (int kv = 0; kv < seq; ++kv) {
        // Causal: query at q_row can only attend to keys at kv <= q_row.
        if (is_causal && kv > q_row) break;

        // Score = dot(q, k_row) * scale, reduced across the warp.
        const bf16* k_row_ptr = k + ((size_t)kv * n_heads + head) * HEAD_DIM;
        float s = 0.0f;
        #pragma unroll
        for (int p = 0; p < PER; ++p) s += qv[p] * __bfloat162float(k_row_ptr[lane * PER + p]);
        // warp reduce
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        s *= scale;

        // Online softmax: rescale previous accumulator and add this key's V.
        float m_new = fmaxf(m_prev, s);
        float exp_prev = expf(m_prev - m_new);   // rescale factor for prior terms
        float exp_cur  = expf(s - m_new);
        float l_new = l_prev * exp_prev + exp_cur;

        const bf16* v_row_ptr = v + ((size_t)kv * n_heads + head) * HEAD_DIM;
        #pragma unroll
        for (int p = 0; p < PER; ++p)
            acc[p] = acc[p] * exp_prev + exp_cur * __bfloat162float(v_row_ptr[lane * PER + p]);

        m_prev = m_new;
        l_prev = l_new;
    }

    // Normalise and write this lane's slice of the output.
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    bf16* out_row_ptr = out + ((size_t)q_row * n_heads + head) * HEAD_DIM;
    #pragma unroll
    for (int p = 0; p < PER; ++p)
        out_row_ptr[lane * PER + p] = __float2bfloat16(acc[p] * inv_l);
}

void flash_attention(const void* q, const void* k, const void* v, void* out,
                     int seq, int n_heads, int head_dim,
                     bool is_causal, cudaStream_t stream) {
    if (head_dim != 128) {
        fprintf(stderr, "[flash_attention] only head_dim=128 supported, got %d\n", head_dim);
        return;
    }
    // One warp (32 threads) per (query row, head). The grid covers seq*n_heads
    // work units; for the vision prefill at N_v~11k, n_heads=12 that is ~132k
    // warps = ~4.2M threads, saturating the GPU.
    dim3 grid(seq, n_heads);
    dim3 block(32);
    flash_attn_kernel<128><<<grid, block, 0, stream>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out,
        seq, n_heads, 128, is_causal);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ---- Decode attention (T=1 over a KV-cache) --------------------------------
// One warp per query head. The warp streams over `past` keys, computing scores
// with a warp-shuffle dot product, an online softmax, and a weighted V sum.
// GQA: query head h reads kv head h/g (g = n_heads / n_kv_heads).
template <int HEAD_DIM>
__global__ void decode_attn_kernel(const bf16* __restrict__ q,
                                   const bf16* __restrict__ k_cache,
                                   const bf16* __restrict__ v_cache,
                                   int past, int n_heads, int n_kv_heads,
                                   bf16* __restrict__ out) {
    constexpr int LANES = 32;
    constexpr int PER = HEAD_DIM / LANES;
    int head = blockIdx.x;
    if (head >= n_heads) return;
    int g = n_heads / n_kv_heads;
    int kv_head = head / g;
    int lane = threadIdx.x;

    const bf16* qh = q + (size_t)head * HEAD_DIM;
    float qv[PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p) qv[p] = __bfloat162float(qh[lane * PER + p]);

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Pass 1: find the max score (for numerical stability).
    float m_prev = -INFINITY;
    for (int r = 0; r < past; ++r) {
        const bf16* kr = k_cache + ((size_t)r * n_kv_heads + kv_head) * HEAD_DIM;
        float s = 0.0f;
        #pragma unroll
        for (int p = 0; p < PER; ++p) s += qv[p] * __bfloat162float(kr[lane * PER + p]);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        s *= scale;
        if (s > m_prev) m_prev = s;
    }

    // Pass 2: exp, sum, and weighted V (online over rows; max is already final).
    float l_prev = 0.0f;
    float acc[PER];
    #pragma unroll
    for (int p = 0; p < PER; ++p) acc[p] = 0.0f;
    for (int r = 0; r < past; ++r) {
        const bf16* kr = k_cache + ((size_t)r * n_kv_heads + kv_head) * HEAD_DIM;
        float s = 0.0f;
        #pragma unroll
        for (int p = 0; p < PER; ++p) s += qv[p] * __bfloat162float(kr[lane * PER + p]);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        s *= scale;
        float e = expf(s - m_prev);
        l_prev += e;
        const bf16* vr = v_cache + ((size_t)r * n_kv_heads + kv_head) * HEAD_DIM;
        #pragma unroll
        for (int p = 0; p < PER; ++p) acc[p] += e * __bfloat162float(vr[lane * PER + p]);
    }

    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    bf16* oh = out + (size_t)head * HEAD_DIM;
    #pragma unroll
    for (int p = 0; p < PER; ++p) oh[lane * PER + p] = __float2bfloat16(acc[p] * inv_l);
}

void decode_attention(const void* q,
                      const void* k_cache, const void* v_cache,
                      int past, int n_heads, int n_kv_heads, int head_dim,
                      void* out, cudaStream_t stream) {
    if (head_dim != 128) {
        fprintf(stderr, "[decode_attention] only head_dim=128 supported, got %d\n", head_dim);
        return;
    }
    decode_attn_kernel<128><<<n_heads, 32, 0, stream>>>(
        (const bf16*)q, (const bf16*)k_cache, (const bf16*)v_cache,
        past, n_heads, n_kv_heads, (bf16*)out);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dots
