// Flash-attention on tensor cores (nvcuda::wmma), with the online softmax kept
// scalar in the score tile. See attention.h.
//
// One CTA computes a BR-row tile of the output for a single head. It streams
// over K/V in BC-row chunks. Per chunk:
//   1) S = Q_tile · K_tile^T           -- WMMA GEMM, [BR,BC] fp32 scores
//   2) online softmax over S (row max, exp, running sum)  -- scalar
//   3) rescale O accumulator, then O += P_tile · V_tile   -- WMMA GEMM
// The two contractions (QK^T and PV) are the compute-heavy parts and run on
// tensor cores; the softmax exp/max/sum is scalar (transcendentals are not
// expressible as MMA). Memory stays O(seq) — the seq×seq score matrix is never
// materialised in HBM, only the BR×BC tile in shared/registers.
//
// head_dim is fixed at 128; BR=BC=16 keep shared memory under the 48 KB
// default cap. is_causal masks the upper triangle on the LLM prefill path.
//
// q,k,v layout: [seq, n_heads, head_dim] row-major (head axis inside seq), so
// head h of token t is contiguous [head_dim] at offset (t*n_heads+h)*head_dim.
#include "attention.h"
#include "kernels.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>

namespace dots {
using namespace nvcuda;

constexpr int HD = 128;
constexpr int BR = 16;     // query tile rows
constexpr int BC = 16;     // key/value tile rows
constexpr int WARPS = 4;   // warps per CTA

template <int HEAD_DIM>
__global__ void __launch_bounds__(WARPS * 32)
flash_attn_kernel(const bf16* __restrict__ q,
                  const bf16* __restrict__ k,
                  const bf16* __restrict__ v,
                  bf16* __restrict__ out,
                  int seq, int n_heads, int head_dim,
                  bool is_causal) {
    (void)head_dim;
    int q_base = blockIdx.x * BR;
    int head   = blockIdx.y;
    if (q_base >= seq || head >= n_heads) return;
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);

    extern __shared__ bf16 smem[];
    bf16* sQ   = smem;                       // [BR, HD]
    bf16* sK   = sQ + BR * HEAD_DIM;         // [BC, HD]
    bf16* sV   = sK + BC * HEAD_DIM;         // [BC, HD]
    bf16* sP   = sV + BC * HEAD_DIM;         // [BR, BC]
    float* sS  = (float*)(sP + BR * BC);     // [BR, BC]
    float* sO   = sS + BR * BC;              // [BR, HD]
    float* m_row = sO + BR * HEAD_DIM;       // [BR]
    float* l_row = m_row + BR;               // [BR]

    int tid = threadIdx.x;
    int warp = tid / 32;

    // Load Q tile.
    for (int i = tid; i < BR * HEAD_DIM; i += WARPS * 32) {
        int r = i / HEAD_DIM, d = i % HEAD_DIM;
        int qrow = q_base + r;
        sQ[i] = (qrow < seq) ? q[((size_t)qrow * n_heads + head) * HEAD_DIM + d] : (bf16)0;
    }
    for (int i = tid; i < BR * HEAD_DIM; i += WARPS * 32) sO[i] = 0.0f;
    for (int r = tid; r < BR; r += WARPS * 32) { m_row[r] = -INFINITY; l_row[r] = 0.0f; }
    __syncthreads();

    int n_kv_tiles = (seq + BC - 1) / BC;
    int causal_last = is_causal ? (q_base + BR - 1) : (seq - 1);

    // Output accumulator fragments: BR×HD tiled into BR/16 × HD/16 16×16 tiles.
    // BR=16 so one row-tile; HD/16=8 col-tiles. Distribute col-tiles over warps.
    constexpr int NT = HEAD_DIM / 16;     // 8
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[NT];
    for (int t = 0; t < NT; ++t) wmma::fill_fragment(o_frag[t], 0.0f);

    for (int kb = 0; kb < n_kv_tiles; ++kb) {
        int kv_base = kb * BC;
        if (is_causal && kv_base > causal_last) break;

        // Load K, V tiles.
        for (int i = tid; i < BC * HEAD_DIM; i += WARPS * 32) {
            int r = i / HEAD_DIM, d = i % HEAD_DIM;
            int kvrow = kv_base + r;
            bool ok = (kvrow < seq);
            sK[i] = ok ? k[((size_t)kvrow * n_heads + head) * HEAD_DIM + d] : (bf16)0;
            sV[i] = ok ? v[((size_t)kvrow * n_heads + head) * HEAD_DIM + d] : (bf16)0;
        }
        __syncthreads();

        // ---- QK^T via WMMA: S[BR,BC] = Q[BR,HD]·K[BC,HD]^T ----
        // BR=BC=16 -> one 16x16 output tile, computed by warp 0.
        if (warp == 0) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
            wmma::fill_fragment(s_frag, 0.0f);
            for (int kt = 0; kt < HEAD_DIM / 16; ++kt) {
                // matrix_a = Q slice [BR, 16] row_major, ld=HEAD_DIM, base sQ + kt*16
                wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, sQ + kt * 16, HEAD_DIM);
                // matrix_b = K^T slice [16, BC]: K is [BC,HD] row_major; its
                // col_major view with ld=HEAD_DIM gives [HD,BC]=K^T. Load the
                // kt-th 16-row block of that = K^T[kt*16.., :], base sK+kt*16.
                wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, sK + kt * 16, HEAD_DIM);
                wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
            }
            wmma::store_matrix_sync(sS, s_frag, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // Scale + causal/seq mask on the score tile.
        for (int i = tid; i < BR * BC; i += WARPS * 32) {
            int r = i / BC, c = i % BC;
            int qrow = q_base + r, kvrow = kv_base + c;
            float s = sS[i] * scale;
            if (qrow >= seq || kvrow >= seq || (is_causal && kvrow > qrow)) s = -INFINITY;
            sS[i] = s;
        }
        __syncthreads();

        // Online softmax per Q row: block max, rescale O + running stats, P=exp.
        for (int r = tid; r < BR; r += WARPS * 32) {
            int qrow = q_base + r;
            if (qrow >= seq) continue;
            float m_block = -INFINITY;
            for (int c = 0; c < BC; ++c) {
                int kvrow = kv_base + c;
                if (kvrow >= seq) continue;
                if (is_causal && kvrow > qrow) continue;
                if (sS[r * BC + c] > m_block) m_block = sS[r * BC + c];
            }
            float m_old = m_row[r];
            float m_new = fmaxf(m_old, m_block);
            float scale_o = expf(m_old - m_new);
            float l_new = l_row[r] * scale_o;
            for (int c = 0; c < BC; ++c) {
                int kvrow = kv_base + c;
                if (kvrow >= seq || (is_causal && kvrow > qrow)) { sP[r * BC + c] = (bf16)0; continue; }
                float p = expf(sS[r * BC + c] - m_new);
                l_new += p;
                sP[r * BC + c] = __float2bfloat16(p);
            }
            for (int d = 0; d < HEAD_DIM; ++d) sO[r * HEAD_DIM + d] *= scale_o;
            m_row[r] = m_new;
            l_row[r] = l_new;
        }
        __syncthreads();

        // ---- PV via WMMA: O[BR,HD] += P[BR,BC]·V[BC,HD] ----
        // Distribute the HD/16 col-tiles across warps.
        for (int t = warp; t < NT; t += WARPS) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> p_frag;
            wmma::load_matrix_sync(p_frag, sP, BC);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> v_frag;
            wmma::load_matrix_sync(v_frag, sV + t * 16, HEAD_DIM);
            wmma::mma_sync(o_frag[t], p_frag, v_frag, o_frag[t]);
        }
        __syncthreads();
    }

    // Dump o_frag[t] into the shared O tile (each warp owns its col-tiles).
    for (int t = warp; t < NT; t += WARPS) {
        wmma::store_matrix_sync(sO + t * 16, o_frag[t], HEAD_DIM, wmma::mem_row_major);
    }
    __syncthreads();

    // Normalise by l, write bf16.
    for (int i = tid; i < BR * HEAD_DIM; i += WARPS * 32) {
        int r = i / HEAD_DIM;
        int qrow = q_base + r;
        if (qrow >= seq) continue;
        float l = l_row[r];
        float o = (l > 0.0f) ? (sO[i] / l) : 0.0f;
        out[((size_t)qrow * n_heads + head) * HEAD_DIM + (i % HEAD_DIM)] = __float2bfloat16(o);
    }
}

void flash_attention(const void* q, const void* k, const void* v, void* out,
                     int seq, int n_heads, int head_dim,
                     bool is_causal, cudaStream_t stream) {
    if (head_dim != 128) {
        fprintf(stderr, "[flash_attention] only head_dim=128 supported, got %d\n", head_dim);
        return;
    }
    dim3 grid((seq + BR - 1) / BR, n_heads);
    dim3 block(WARPS * 32);
    size_t shared = (BR*HD + BC*HD + BC*HD + BR*BC) * sizeof(bf16)
                  + (BR*BC + BR*HD + BR + BR) * sizeof(float);
    static bool cap_set = false;
    if (!cap_set) {
        cudaFuncSetAttribute((const void*)flash_attn_kernel<HD>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cap_set = true;
    }
    flash_attn_kernel<HD><<<grid, block, shared, stream>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out,
        seq, n_heads, HD, is_causal);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ---- Decode attention (single query over KV-cache) ------------------------
// T=1: the QK contraction is a GEMV [1,past] (M=1, memory-bound — not a natural
// tensor-core shape). Kept as the scalar warp-reduce GEMV from the first pass;
// moving this to TC would need padding past to a multiple of 16 and a batched
// MMA across heads, which is future work. Documented in README.
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
