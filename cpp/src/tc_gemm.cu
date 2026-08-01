// Tensor-core bf16 GEMM via nvcuda::wmma (m16n16k16), with a register-resident
// fused epilogue (bias / SiLU / GELU). See tc_gemm.h.
//
// Blocked GEMM: one CTA computes a BM×BN output tile using WMMA 16×16 frags.
// Each of the (BM/16)×(BN/16) warps owns one 16×16 output sub-tile. K is
// streamed in BK=16 chunks through shared memory (one A-tile [BM,BK] and one
// B-tile [BK,BN] per chunk), so each shared load feeds BM/16 × BN/16 WMMA ops —
// high arithmetic intensity per HBM byte. Direct row-major fragments + store.
//
// Transposed operands (transA/transB) are handled at the shared-memory load by
// indexing the source with the right stride, so the WMMA loads are always
// row_major from contiguous shared tiles.
#include "tc_gemm.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>

namespace dots {
using namespace nvcuda;

constexpr int WM = 16, WN = 16, WK = 16;
// Block tile. Each warp owns a 16x16 output; the block has WARPS_M*WARPS_N warps.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int WARPS_M = BM / WM;   // 4
constexpr int WARPS_N = BN / WN;   // 4
constexpr int NWARPS = WARPS_M * WARPS_N;  // 16

__device__ __forceinline__ float epilog_apply(float v, int col, Epilog ep,
                                              const bf16* bias) {
    switch (ep) {
        case Epilog::BIAS: v += __bfloat162float(bias[col]); break;
        case Epilog::SILU: { v += __bfloat162float(bias[col]);
                             float s = 1.0f / (1.0f + expf(-v)); v *= s; break; }
        case Epilog::GELU: { v += __bfloat162float(bias[col]);
                             v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)); break; }
        default: break;
    }
    return v;
}

__global__ void __launch_bounds__(NWARPS * 32)
tc_gemm_kernel(const bf16* __restrict__ A, bool transA,
               const bf16* __restrict__ B, bool transB,
               bf16* __restrict__ C,
               int M, int N, int K, Epilog ep,
               const bf16* __restrict__ bias, float alpha, float beta) {
    int bm0 = blockIdx.x * BM;
    int bn0 = blockIdx.y * BN;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int wm = warp / WARPS_N;     // 0..WARPS_M-1
    int wn = warp % WARPS_N;     // 0..WARPS_N-1

    int ldA = transA ? M : K;
    int ldB = transB ? K : N;
    auto A_at = [&](int row, int k) -> const bf16* {
        return transA ? (A + (size_t)k * ldA + row) : (A + (size_t)row * ldA + k);
    };
    auto B_at = [&](int k, int col) -> const bf16* {
        return transB ? (B + (size_t)col * ldB + k) : (B + (size_t)k * ldB + col);
    };

    extern __shared__ bf16 smem[];
    bf16* sA = smem;                // [BM, BK] row-major
    bf16* sB = sA + BM * BK;        // [BK, BN] row-major

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int nK = (K + BK - 1) / BK;
    for (int kb = 0; kb < nK; ++kb) {
        int k0 = kb * BK;
        // Cooperatively load A tile [BM,BK]: sA[i*BK+j] = op(A)[bm0+i, k0+j].
        for (int i = tid; i < BM * BK; i += NWARPS * 32) {
            int r = i / BK, c = i % BK;
            int aRow = bm0 + r, ak = k0 + c;
            sA[i] = (aRow < M && ak < K) ? *A_at(aRow, ak) : (bf16)0;
        }
        // Load B tile [BK,BN]: sB[i*BN+j] = op(B)[k0+i, bn0+j].
        for (int i = tid; i < BK * BN; i += NWARPS * 32) {
            int r = i / BN, c = i % BN;
            int bk = k0 + r, bCol = bn0 + c;
            sB[i] = (bk < K && bCol < N) ? *B_at(bk, bCol) : (bf16)0;
        }
        __syncthreads();

        // Each warp computes its 16x16 output: rows [wm*16..], cols [wn*16..].
        // matrix_a = op(A) sub-tile [16, BK=16] at sA[wm*16, 0], ld=BK.
        wmma::fragment<wmma::matrix_a, WM, WN, WK, bf16, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, sA + wm * 16 * BK, BK);
        // matrix_b = op(B) sub-tile [BK=16, 16] at sB[0, wn*16], ld=BN.
        wmma::fragment<wmma::matrix_b, WM, WN, WK, bf16, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, sB + wn * 16, BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store accumulator to a shared fp32 tile [BM,BN] row-major, apply epilog,
    // write bf16 to global. Each warp stores its 16x16 fragment.
    float* c_tile = (float*)(smem);   // reuse smem region as fp32 [BM,BN]
    wmma::store_matrix_sync(c_tile + (wm * 16) * BN + wn * 16, c_frag, BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < BM * BN; i += NWARPS * 32) {
        int r = i / BN, c = i % BN;
        int gRow = bm0 + r, gCol = bn0 + c;
        if (gRow >= M || gCol >= N) continue;
        float v = c_tile[i] * alpha;
        v = epilog_apply(v, gCol, ep, bias);
        if (beta != 0.0f) v += beta * __bfloat162float(C[(size_t)gRow * N + gCol]);
        C[(size_t)gRow * N + gCol] = __float2bfloat16(v);
    }
}

// ---- GEMV path for M==1 (decode linear layers) -----------------------------
// M==1 is a matrix-vector product: tensor cores are the wrong tool (a 16×16
// MMA tile would waste 15/16 of its rows). cuBLAS switches to a CUDA-core
// GEMV internally for this shape; we do the same. One warp per output column
// tile of 32 elements, looping over K. Output still gets the fused epilogue.
__global__ void __launch_bounds__(32)
tc_gemv_kernel(const bf16* __restrict__ A, bool transA,
               const bf16* __restrict__ B, bool transB,
               bf16* __restrict__ C,
               int N, int K, Epilog ep,
               const bf16* __restrict__ bias, float alpha, float beta) {
    // A is the single query row [1,K]. Each lane owns ONE output column `col`
    // and computes dot(A, op(B)[:,col]). Vectorised: read 2 A's and 2 B's per
    // step as bf16x2 (a single 32-bit load). This is the memory-bound GEMV
    // roofline — K=1536, so ~1.5 KB of B per output, broadcast across lanes.
    int n0 = blockIdx.x * 32;
    int lane = threadIdx.x;
    int col = n0 + lane;
    if (col >= N) return;
    float acc = 0.0f;
    const bf16x2* Av2 = (const bf16x2*)A;
    if (!transB) {
        const bf16x2* Bk = (const bf16x2*)(B + (size_t)0 * N + col);
        // B stored [K,N]; for row k the element is at B[k*N + col]. bf16x2 load
        // of consecutive k's is at &B[k*N+col] which is strided by N — NOT
        // contiguous in k, so can't vectorise the k-loop. Fall back to scalar.
        for (int k = 0; k < K; ++k)
            acc += __bfloat162float(A[k]) * __bfloat162float(B[(size_t)k * N + col]);
    } else {
        // B stored [N,K]: B[col*K + k]. Consecutive k's ARE contiguous, so
        // vectorise the dot product with bf16x2 (4 bytes/2 elems per load).
        const bf16x2* Bcol = (const bf16x2*)(B + (size_t)col * K);
        int pairs = K / 2;
        for (int k = 0; k < pairs; ++k) {
            bf16x2 a = Av2[k];
            bf16x2 b = Bcol[k];
            acc += __bfloat162float(__low2bfloat16(a)) * __bfloat162float(__low2bfloat16(b));
            acc += __bfloat162float(__high2bfloat16(a)) * __bfloat162float(__high2bfloat16(b));
        }
        if (K & 1) acc += __bfloat162float(A[K - 1]) * __bfloat162float(B[(size_t)col * K + K - 1]);
    }
    float v = acc * alpha;
    v = epilog_apply(v, col, ep, bias);
    if (beta != 0.0f) v += beta * __bfloat162float(C[col]);
    C[col] = __float2bfloat16(v);
}

// Small-M variant: one warp per 16×16 output tile, BM=BN=16. Used when M is
// small (decode: M=1) so the big BM=64 block doesn't waste 63/64 of its tiles.
template <int /*tag*/>
__global__ void __launch_bounds__(32)
tc_gemm_small_kernel(const bf16* __restrict__ A, bool transA,
                     const bf16* __restrict__ B, bool transB,
                     bf16* __restrict__ C,
                     int M, int N, int K, Epilog ep,
                     const bf16* __restrict__ bias, float alpha, float beta) {
    int bm = blockIdx.x * 16;
    int bn = blockIdx.y * 16;
    if (bm >= M || bn >= N) return;
    int ldA = transA ? M : K;
    int ldB = transB ? K : N;
    auto A_at = [&](int row, int k) -> const bf16* {
        return transA ? (A + (size_t)k * ldA + row) : (A + (size_t)row * ldA + k);
    };
    auto B_at = [&](int k, int col) -> const bf16* {
        return transB ? (B + (size_t)col * ldB + k) : (B + (size_t)k * ldB + col);
    };
    extern __shared__ bf16 smem[];
    bf16* sA = smem;             // [16,16]
    bf16* sB = sA + 256;         // [16,16]

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    int nK = (K + BK - 1) / BK;
    for (int kb = 0; kb < nK; ++kb) {
        int k0 = kb * BK;
        for (int idx = threadIdx.x; idx < 256; idx += 32) {
            int i = idx >> 4, j = idx & 15;
            int aRow = bm + i, ak = k0 + j;
            sA[idx] = (aRow < M && ak < K) ? *A_at(aRow, ak) : (bf16)0;
            int bk = k0 + i, bCol = bn + j;
            sB[idx] = (bk < K && bCol < N) ? *B_at(bk, bCol) : (bf16)0;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, WM, WN, WK, bf16, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, sA, 16);
        wmma::fragment<wmma::matrix_b, WM, WN, WK, bf16, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, sB, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    float* c_tile = (float*)smem;
    wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int i = idx >> 4, j = idx & 15;
        int gRow = bm + i, gCol = bn + j;
        if (gRow >= M || gCol >= N) continue;
        float v = c_tile[idx] * alpha;
        v = epilog_apply(v, gCol, ep, bias);
        if (beta != 0.0f) v += beta * __bfloat162float(C[(size_t)gRow * N + gCol]);
        C[(size_t)gRow * N + gCol] = __float2bfloat16(v);
    }
}

void tc_gemm(const void* A, bool transA,
             const void* B, bool transB,
             void* C,
             int M, int N, int K,
             Epilog ep, const void* bias,
             float alpha, float beta) {
    if (M <= 0 || N <= 0) return;
    // M==1 is a GEMV: tensor cores waste 15/16 of a 16x16 tile's rows, so use
    // a CUDA-core GEMV (one warp per 32-element output tile) — this is what
    // cuBLAS does internally for this shape. Every M>1 contraction stays on TC.
    if (M == 1) {
        tc_gemv_kernel<<<(N + 31) / 32, 32>>>(
            (const bf16*)A, transA, (const bf16*)B, transB, (bf16*)C,
            N, K, ep, (const bf16*)bias, alpha, beta);
        DOTS_CUDA_CHECK(cudaGetLastError());
        return;
    }
    // Small-M (decode, M=1): the BM=64 blocked kernel would compute 64x64
    // tiles and throw away 63/64 of the rows. Use the 16x16 single-warp kernel
    // whose tile matches the small M — many CTAs along N give parallelism.
    if (M <= 16) {
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        int shared = 2 * 16 * 16 * sizeof(bf16) + 16 * 16 * sizeof(float);
        tc_gemm_small_kernel<0><<<grid, 32, shared>>>(
            (const bf16*)A, transA, (const bf16*)B, transB, (bf16*)C,
            M, N, K, ep, (const bf16*)bias, alpha, beta);
        DOTS_CUDA_CHECK(cudaGetLastError());
        return;
    }
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(NWARPS * 32);
    size_t shared_ab = (BM * BK + BK * BN) * sizeof(bf16);
    size_t shared_c  = BM * BN * sizeof(float);
    size_t shared = shared_ab > shared_c ? shared_ab : shared_c;
    static bool cap_set = false;
    if (!cap_set) {
        cudaFuncSetAttribute((const void*)tc_gemm_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cap_set = true;
    }
    tc_gemm_kernel<<<grid, block, shared>>>(
        (const bf16*)A, transA, (const bf16*)B, transB, (bf16*)C,
        M, N, K, ep, (const bf16*)bias, alpha, beta);
    DOTS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dots
