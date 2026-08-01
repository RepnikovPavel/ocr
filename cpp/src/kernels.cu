// CUDA kernels: elementwise ops, normalisation, RoPE, embedding.
//
// Every kernel here is one of two shapes of load:
//   - row-wise reduction (RMSNorm, LayerNorm): one block per row, threads
//     cooperate to reduce across `dim` with a warp shuffle;
//   - element-wise (SiLU, GELU, SwiGLU, add, embed gather): grid-stride loop,
//     bf16 vectorised as bf16x2 where the access is aligned.
//
// These are the bandwidth-bound glue between the tensor-core GEMMs (tc_gemm.cu)
// and attention (attention.cu). cuBLAS is no longer linked — all contractions
// are our own WMMA kernels.
#include "kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cmath>

namespace dots {

// ============================ RMSNorm ========================================
// One block per row. dim <= 1024 (our dims are 1536 and 4224, head_dim 128),
// so we use a two-pass block reduction: each thread accumulates a partial
// sum of squares, then a warp+block reduction finishes it.
template <int BLOCK>
__global__ void rms_norm_kernel(const bf16* __restrict__ x,
                                const bf16* __restrict__ w,
                                bf16* __restrict__ out,
                                int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const bf16* xr = x + (size_t)row * dim;
    bf16* outr = out + (size_t)row * dim;

    extern __shared__ float smem[];
    // First pass: sum of squares.
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK) {
        float v = __bfloat162float(xr[i]);
        sum += v * v;
    }
    // Warp reduce.
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    // Block reduce across warps.
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = sum;
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < (BLOCK / 32)) ? smem[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (threadIdx.x == 0) smem[0] = sum;
    }
    __syncthreads();
    float total = smem[0];
    float rsqrt = rsqrtf(total / dim + eps);
    // Second pass: normalise + scale.
    for (int i = threadIdx.x; i < dim; i += BLOCK) {
        float v = __bfloat162float(xr[i]);
        float wv = __bfloat162float(w[i]);
        outr[i] = __float2bfloat16(v * rsqrt * wv);
    }
}

void rms_norm(const void* x, const void* weight, void* out,
              int rows, int dim, float eps) {
    constexpr int BLOCK = 256;
    int shared = (BLOCK / 32) * sizeof(float);
    rms_norm_kernel<BLOCK><<<rows, BLOCK, shared>>>(
        (const bf16*)x, (const bf16*)weight, (bf16*)out, rows, dim, eps);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ============================ LayerNorm ======================================
template <int BLOCK>
__global__ void layer_norm_kernel(const bf16* __restrict__ x,
                                  const bf16* __restrict__ w,
                                  const bf16* __restrict__ b,
                                  bf16* __restrict__ out,
                                  int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const bf16* xr = x + (size_t)row * dim;
    bf16* outr = out + (size_t)row * dim;

    extern __shared__ float smem[];
    float sum = 0.0f, sumsq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK) {
        float v = __bfloat162float(xr[i]);
        sum += v; sumsq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) { sum += __shfl_xor_sync(0xffffffff, sum, off); sumsq += __shfl_xor_sync(0xffffffff, sumsq, off); }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { smem[wid] = sum; smem[wid + BLOCK / 32] = sumsq; }
    __syncthreads();
    if (wid == 0) {
        sum = (threadIdx.x < (BLOCK / 32)) ? smem[threadIdx.x] : 0.0f;
        sumsq = (threadIdx.x < (BLOCK / 32)) ? smem[threadIdx.x + BLOCK / 32] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) { sum += __shfl_xor_sync(0xffffffff, sum, off); sumsq += __shfl_xor_sync(0xffffffff, sumsq, off); }
        if (threadIdx.x == 0) { smem[0] = sum; smem[1] = sumsq; }
    }
    __syncthreads();
    float mean = smem[0] / dim;
    float var  = smem[1] / dim - mean * mean;
    float rsqrt = rsqrtf(var + eps);
    for (int i = threadIdx.x; i < dim; i += BLOCK) {
        float v = __bfloat162float(xr[i]);
        float wv = __bfloat162float(w[i]);
        float bv = b ? __bfloat162float(b[i]) : 0.0f;
        outr[i] = __float2bfloat16((v - mean) * rsqrt * wv + bv);
    }
}

void layer_norm(const void* x, const void* weight, const void* bias,
                void* out, int rows, int dim, float eps) {
    constexpr int BLOCK = 256;
    int shared = 2 * (BLOCK / 32) * sizeof(float);
    layer_norm_kernel<BLOCK><<<rows, BLOCK, shared>>>(
        (const bf16*)x, (const bf16*)weight, (const bf16*)bias, (bf16*)out, rows, dim, eps);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ===================== elementwise activations ==============================
__global__ void silu_inplace_kernel(bf16* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    float s = 1.0f / (1.0f + expf(-v));
    x[i] = __float2bfloat16(v * s);
}
void silu_inplace(void* x, int n) {
    int block = 256, grid = (n + block - 1) / block;
    silu_inplace_kernel<<<grid, block>>>((bf16*)x, n);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

__global__ void swiglu_kernel(const bf16* gate, const bf16* up, bf16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float s = 1.0f / (1.0f + expf(-g));
    out[i] = __float2bfloat16(g * s * u);
}
void swiglu(const void* gate, const void* up, void* out, int n) {
    int block = 256, grid = (n + block - 1) / block;
    swiglu_kernel<<<grid, block>>>((const bf16*)gate, (const bf16*)up, (bf16*)out, n);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// Exact-erf GELU (nn.GELU() default). The merger MLP uses nn.GELU(), whose
// forward is 0.5 * x * (1 + erf(x / sqrt(2))). CUDA has an erf intrinsic.
__global__ void gelu_kernel(bf16* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __bfloat162float(x[i]);
    float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    x[i] = __float2bfloat16(g);
}
void gelu(void* x, int n) {
    int block = 256, grid = (n + block - 1) / block;
    gelu_kernel<<<grid, block>>>((bf16*)x, n);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

__global__ void add_kernel(const bf16* a, const bf16* b, bf16* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}
void add(const void* a, const void* b, void* y, int n) {
    int block = 256, grid = (n + block - 1) / block;
    add_kernel<<<grid, block>>>((const bf16*)a, (const bf16*)b, (bf16*)y, n);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ---- per-row bias add, dtype cast, qkv split --------------------------------
__global__ void add_row_bias_kernel(bf16* x, const bf16* bias, int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    bf16* xr = x + (size_t)row * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        xr[i] = __float2bfloat16(__bfloat162float(xr[i]) + __bfloat162float(bias[i]));
}
void add_row_bias(void* x, const void* bias, int rows, int dim) {
    add_row_bias_kernel<<<rows, 256>>>((bf16*)x, (const bf16*)bias, rows, dim);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

__global__ void cast_f32_to_bf16_kernel(const float* src, bf16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}
void cast_f32_to_bf16(const void* src_f32, void* dst_bf16, int n) {
    int block = 256, grid = (n + block - 1) / block;
    cast_f32_to_bf16_kernel<<<grid, block>>>((const float*)src_f32, (bf16*)dst_bf16, n);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

__global__ void cast_bf16_to_f32_kernel(const bf16* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}
void cast_bf16_to_f32(const void* src_bf16, void* dst_f32, int n) {
    int block = 256, grid = (n + block - 1) / block;
    cast_bf16_to_f32_kernel<<<grid, block>>>((const bf16*)src_bf16, (float*)dst_f32, n);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// qkv fused layout: per token n, [3, H, HD] -> [3*H*HD]. We split to q[n,H,HD],
// k[n,H,HD], v[n,H,HD] contiguous. The trailing layout is the same (H,HD) per
// (n, qkv), so a copy with a stride is enough.
__global__ void split_qkv_kernel(const bf16* qkv, bf16* q, bf16* k, bf16* v,
                                 int N, int HHD) {
    // one thread per (n, qkv_head_offset)
    int total = N * 3 * HHD;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        int off = idx % HHD;
        int qkv_idx = (idx / HHD) % 3;
        int n = idx / (3 * HHD);
        const bf16* src = qkv + ((size_t)n * 3 + qkv_idx) * HHD + off;
        bf16* dstbase = (qkv_idx == 0 ? q : (qkv_idx == 1 ? k : v));
        dstbase[(size_t)n * HHD + off] = *src;
    }
}
void split_qkv(const void* qkv, void* q, void* k, void* v, int N, int n_heads, int head_dim) {
    int HHD = n_heads * head_dim;
    int n = N * 3 * HHD;
    int block = 256, grid = (n + block - 1) / block;
    split_qkv_kernel<<<grid, block>>>((const bf16*)qkv, (bf16*)q, (bf16*)k, (bf16*)v, N, HHD);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ---- KV cache helpers -------------------------------------------------------
__global__ void write_kv_cache_kernel(bf16* cache, const bf16* src,
                                      int pos_start, int T, int kv_dim, int cache_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * kv_dim;
    for (; idx < total; idx += gridDim.x * blockDim.x) {
        int t = idx / kv_dim, d = idx % kv_dim;
        cache[(size_t)(pos_start + t) * kv_dim + d] = src[(size_t)t * kv_dim + d];
    }
    (void)cache_max;
}
void write_kv_cache(void* cache, const void* src, int pos_start, int T,
                    int kv_dim, int cache_max_seq) {
    int n = T * kv_dim;
    int block = 256, grid = (n + block - 1) / block;
    write_kv_cache_kernel<<<grid, block>>>((bf16*)cache, (const bf16*)src,
                                           pos_start, T, kv_dim, cache_max_seq);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// Expand GQA KV [seq, n_kv, hd] -> [seq, n_heads, hd] by repeating each kv head
// g times contiguously (head h reads kv head h/g).
__global__ void expand_kv_gqa_kernel(const bf16* src, bf16* dst,
                                     int seq, int n_kv, int hd, int n_heads) {
    int g = n_heads / n_kv;
    int total = seq * n_heads * hd;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        int d = idx % hd;
        int h = (idx / hd) % n_heads;
        int s = idx / (n_heads * hd);
        const bf16* row = src + ((size_t)s * n_kv + (h / g)) * hd;
        dst[(size_t)s * n_heads * hd + h * hd + d] = row[d];
    }
}
void expand_kv_gqa(const void* kv_src, void* dst, int seq, int n_kv_heads,
                   int head_dim, int n_heads) {
    int n = seq * n_heads * head_dim;
    int block = 256, grid = (n + block - 1) / block;
    expand_kv_gqa_kernel<<<grid, block>>>((const bf16*)kv_src, (bf16*)dst,
                                          seq, n_kv_heads, head_dim, n_heads);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ===================== embedding gather + scatter ============================
// Text rows: copy the embedding row verbatim. Image slots: copy from the
// vision embedding buffer in order of appearance. One thread per (token, 8 bf16).
__global__ void embed_scatter_kernel(const int* ids, const bf16* table,
                                     bf16* out, int seq, int hidden,
                                     const int8_t* img_mask, const bf16* ve,
                                     int num_img) {
    int token = blockIdx.x;
    if (token >= seq) return;
    bool is_img = img_mask && img_mask[token];
    const bf16* src;
    if (is_img) {
        // find this token's ordinal among image slots
        // (done with a scan; cheap because image slots are contiguous in practice)
        int ord = 0;
        for (int t = 0; t < token; ++t) if (img_mask[t]) ++ord;
        src = ve + (size_t)ord * hidden;
    } else {
        src = table + (size_t)ids[token] * hidden;
    }
    bf16* dst = out + (size_t)token * hidden;
    // Vectorised copy 8 bf16 = 4 bf16x2 = 16 bytes at a time when aligned.
    for (int i = threadIdx.x * 8; i < hidden; i += blockDim.x * 8) {
        int4 v = *((const int4*)(src + i));
        *((int4*)(dst + i)) = v;
    }
}

void embed_and_scatter(const int* ids, const void* table, void* out,
                       int seq, int hidden,
                       const int8_t* img_mask, const void* ve, int num_img) {
    embed_scatter_kernel<<<seq, (hidden / 8 + 31) / 32 * 32>>>(
        ids, (const bf16*)table, (bf16*)out, seq, hidden,
        (const int8_t*)img_mask, (const bf16*)ve, num_img);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

void embed_lookup(const int* ids, const void* table, void* out,
                  int seq, int hidden) {
    embed_and_scatter(ids, table, out, seq, hidden, nullptr, nullptr, 0);
}

// ===================== argmax + last-row copy ===============================
__global__ void argmax_last_kernel(const bf16* logits, int vocab, int* out_id) {
    extern __shared__ float smem[];
    float bestv = -INFINITY;
    int besti = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > bestv) { bestv = v; besti = i; }
    }
    // Block-wide argmax reduction.
    __shared__ float sval[32]; __shared__ int sidx[32];
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bestv, off);
        int oi = __shfl_xor_sync(0xffffffff, besti, off);
        if (ov > bestv) { bestv = ov; besti = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sval[wid] = bestv; sidx[wid] = besti; }
    __syncthreads();
    if (wid == 0 && lane < (blockDim.x / 32)) {
        bestv = sval[lane]; besti = sidx[lane];
        for (int off = (blockDim.x / 32) / 2; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bestv, off);
            int oi = __shfl_xor_sync(0xffffffff, besti, off);
            if (ov > bestv) { bestv = ov; besti = oi; }
        }
        if (lane == 0) *out_id = besti;
    }
}

int argmax_last(const void* logits, int vocab) {
    int block = 1024;
    int* d_id; DOTS_CUDA_CHECK(cudaMalloc(&d_id, sizeof(int)));
    argmax_last_kernel<<<1, block, 32 * (sizeof(float) + sizeof(int))>>>(
        (const bf16*)logits, vocab, d_id);
    DOTS_CUDA_CHECK(cudaGetLastError());
    int id = -1;
    DOTS_CUDA_CHECK(cudaMemcpy(&id, d_id, sizeof(int), cudaMemcpyDeviceToHost));
    DOTS_CUDA_CHECK(cudaFree(d_id));
    return id;
}

__global__ void copy_last_row_kernel(const bf16* x, bf16* out, int seq, int hidden) {
    const bf16* src = x + (size_t)(seq - 1) * hidden;
    for (int i = threadIdx.x * 8; i < hidden; i += blockDim.x * 8) {
        *((int4*)(out + i)) = *((const int4*)(src + i));
    }
}
void copy_last_row(const void* x, void* out, int seq, int hidden) {
    copy_last_row_kernel<<<1, (hidden / 8 + 31) / 32 * 32>>>(
        (const bf16*)x, (bf16*)out, seq, hidden);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ============================ RoPE (LLM) =====================================
// NEOX rotate-half. For head_dim=128, the layout per head is:
//   [x0..x63 | x64..x127], and rotate_half produces [-x64..x127 | x0..x63].
// The cos/sin table is [seq, head_dim/2] duplicated to [seq, head_dim]:
//   cos_full[pos, d] = cos_table[pos, d % 64], same for sin.
//
// rope_table_llm fills cos/sin with: inv_freq[i] = 1/theta^(2i/head_dim),
//   freqs[pos,i] = pos * inv_freq[i],  cos=cos(freqs), sin=sin(freqs),
// then the apply kernel broadcasts the half-table to the full head_dim.

__global__ void rope_table_llm_kernel(float* cos, float* sin,
                                       int seq, int head_dim, float theta) {
    // grid-stride over (pos, half_dim)
    int half = head_dim / 2;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < seq * half;
         idx += gridDim.x * blockDim.x) {
        int pos = idx / half;
        int i   = idx % half;
        float inv_freq = 1.0f / powf(theta, (2.0f * i) / float(head_dim));
        float freq = pos * inv_freq;
        cos[idx] = cosf(freq);
        sin[idx] = sinf(freq);
    }
}

void rope_table_llm(float* cos, float* sin, int seq, int head_dim, float theta,
                    cudaStream_t stream) {
    int n = seq * (head_dim / 2);
    int block = 256, grid = (n + block - 1) / block;
    rope_table_llm_kernel<<<grid, block, 0, stream>>>(cos, sin, seq, head_dim, theta);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// Apply: q,k shape [seq, n_heads, head_dim]. cos/sin are [seq, head_dim/2],
// expanded to full by indexing d % half.
// rotate_half(x)[d] = d < half ? x[d+half] : x[d-half]  (with sign flip on first half)
//   out = x * cos_full + rotate_half(x) * sin_full
__global__ void apply_rope_llm_kernel(bf16* q, bf16* k,
                                      int seq, int n_heads, int n_kv_heads,
                                      int head_dim, const float* cos, const float* sin) {
    int half = head_dim / 2;
    // one thread per (pos, head, half-pair index)
    int total = seq * n_heads * half;
    for (int idx = blockIdx.x * blockDim.x; idx < total; idx += gridDim.x * blockDim.x) {
        int ph = idx % half;
        int rest = idx / half;
        int head = rest % n_heads;
        int pos  = rest / n_heads;
        float c = cos[pos * half + ph];
        float s = sin[pos * half + ph];
        // Q
        size_t base_q = ((size_t)pos * n_heads + head) * head_dim;
        float q0 = __bfloat162float(q[base_q + ph]);       // first half
        float q1 = __bfloat162float(q[base_q + half + ph]); // second half
        // rotate_half: [-q1, q0]
        q[base_q + ph]        = __float2bfloat16(q0 * c - q1 * s);
        q[base_q + half + ph] = __float2bfloat16(q1 * c + q0 * s);
        // K: same op but only n_kv_heads rows per position.
        if (head < n_kv_heads) {
            size_t base_k = ((size_t)pos * n_kv_heads + head) * head_dim;
            float k0 = __bfloat162float(k[base_k + ph]);
            float k1 = __bfloat162float(k[base_k + half + ph]);
            k[base_k + ph]        = __float2bfloat16(k0 * c - k1 * s);
            k[base_k + half + ph] = __float2bfloat16(k1 * c + k0 * s);
        }
    }
}

void apply_rope_llm(void* q, void* k, int seq, int n_heads, int n_kv_heads,
                    int head_dim, const float* cos, const float* sin) {
    int n = seq * n_heads * (head_dim / 2);
    int block = 256, grid = (n + block - 1) / block;
    apply_rope_llm_kernel<<<grid, block>>>(
        (bf16*)q, (bf16*)k, seq, n_heads, n_kv_heads, head_dim, cos, sin);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

// ============================ RoPE (vision 2D) ==============================
// apply_rotary_pos_emb_vision builds cos/sin by:
//   cos = freqs.cos().unsqueeze(1).repeat(1,1,2)   # [N_v, 1, head_dim/2] -> [N_v,1,head_dim]
// i.e. the half-table is duplicated to fill head_dim. The input tensor layout
// is [1, N_v, n_heads, head_dim] before the squeeze; we operate on
// [N_v, n_heads, head_dim] flat.
//
// freqs is [N_v, head_dim/2] holding the *angle* (pos*inv_freq). We compute
// cos/sin inside the kernel to avoid an extra pass.
__global__ void apply_rotary_vision_kernel(bf16* q, bf16* k, int N_v,
                                           int n_heads, int head_dim,
                                           const float* freqs) {
    int half = head_dim / 2;
    int total = N_v * n_heads * half;
    for (int idx = blockIdx.x * blockDim.x; idx < total; idx += gridDim.x * blockDim.x) {
        int ph = idx % half;
        int rest = idx / half;
        int head = rest % n_heads;
        int pos  = rest / n_heads;
        float ang = freqs[pos * half + ph];
        float c = cosf(ang), s = sinf(ang);
        size_t base_q = ((size_t)pos * n_heads + head) * head_dim;
        float q0 = __bfloat162float(q[base_q + ph]);
        float q1 = __bfloat162float(q[base_q + half + ph]);
        // rotate_half: [-q1, q0]
        q[base_q + ph]        = __float2bfloat16(q0 * c - q1 * s);
        q[base_q + half + ph] = __float2bfloat16(q1 * c + q0 * s);
        size_t base_k = ((size_t)pos * n_heads + head) * head_dim;
        float k0 = __bfloat162float(k[base_k + ph]);
        float k1 = __bfloat162float(k[base_k + half + ph]);
        k[base_k + ph]        = __float2bfloat16(k0 * c - k1 * s);
        k[base_k + half + ph] = __float2bfloat16(k1 * c + k0 * s);
    }
}

void apply_rotary_vision(void* q, void* k, int N_v, int n_heads, int head_dim,
                         const float* freqs) {
    int n = N_v * n_heads * (head_dim / 2);
    int block = 256, grid = (n + block - 1) / block;
    apply_rotary_vision_kernel<<<grid, block>>>(
        (bf16*)q, (bf16*)k, N_v, n_heads, head_dim, freqs);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dots
