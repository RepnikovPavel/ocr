// Hand-written tensor-core GEMM (nvcuda::wmma, bf16, m16n16k16).
//
// This replaces cuBLAS/cuBLASLt for every linear layer in the engine. Because
// the GEMM is ours, the epilogue runs in registers on the fp32 accumulator
// before the single bf16 store to HBM — so bias-add and the activation
// (SiLU / GELU) are fused with zero extra memory traffic. cuBLASLt has no
// SiLU epilogue at all, so this is the only way to fuse it; for GELU/bias it
// matches what cuBLASLt would do.
//
// Row-major operands throughout (matching how the safetensors weights and the
// activations are laid out). The WMMA API is column-major, which we account
// for with the standard transpose identity (see tc_gemm.cu).
#pragma once

#include "tensor.h"

namespace dots {

enum class Epilog : int {
    NONE = 0,   // C = alpha * op(A) * op(B) + beta * C
    BIAS,       // ... + bias[n]                  (Linear with bias)
    SILU,       // ... + bias[n], then x*sigmoid(x)
    GELU,       // ... + bias[n], then exact-erf GELU
};

// Row-major bf16 GEMM on tensor cores.
//   op_A(A) is [M,K] (transA=false) or [K,M] (transA=true)
//   op_B(B) is [K,N] / [N,K]
//   C is [M,N] bf16 (row-major). alpha/beta as usual.
// `bias` is a [N] bf16 vector broadcast across the M rows; required for the
// non-NONE epilogues.
void tc_gemm(const void* A, bool transA,
             const void* B, bool transB,
             void* C,
             int M, int N, int K,
             Epilog ep = Epilog::NONE,
             const void* bias = nullptr,
             float alpha = 1.0f, float beta = 0.0f);

}  // namespace dots
