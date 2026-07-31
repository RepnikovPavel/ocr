// cuBLAS bf16 GEMM correctness test against a naive CPU reference, for all four
// transA/transB combinations. This is the one place a row<->col-major mixup
// hides, so we check it directly.
#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace dots;

// CPU reference: C[M,N] = alpha * op(A)*op(B) + beta*C, row-major, float.
static void cpu_gemm(const std::vector<float>& A, bool tA,
                     const std::vector<float>& B, bool tB,
                     std::vector<float>& C, int M,int N,int K,
                     float alpha, float beta) {
    auto Aat = [&](int m, int k){ return tA ? A[k*M+m] : A[m*K+k]; };  // A stored [M,K] if !tA else [K,M]
    auto Bat = [&](int k, int n){ return tB ? B[n*K+k] : B[k*N+n]; };
    for (int m=0;m<M;++m) for (int n=0;n<N;++n){
        float s=0; for(int k=0;k<K;++k) s += Aat(m,k)*Bat(k,n);
        C[m*N+n] = alpha*s + beta*C[m*N+n];
    }
}

static bool run_one(bool tA, bool tB, int M, int N, int K) {
    std::vector<float> hA(M*K), hB(K*N), hC(M*N);
    for (auto& v: hA) v = ((float)(rand()%200)-100)/100.0f;
    for (auto& v: hB) v = ((float)(rand()%200)-100)/100.0f;
    std::vector<float> ref(M*N, 0.0f);
    cpu_gemm(hA, tA, hB, tB, ref, M, N, K, 1.0f, 0.0f);

    // upload as bf16
    bf16 *dA,*dB; void* dC; cudaMalloc(&dA, hA.size()*sizeof(bf16)); cudaMalloc(&dB, hB.size()*sizeof(bf16));
    cudaMalloc(&dC, hC.size()*sizeof(bf16));
    float* sA; cudaMalloc(&sA, hA.size()*4); cudaMemcpy(sA, hA.data(), hA.size()*4, cudaMemcpyHostToDevice);
    float* sB; cudaMalloc(&sB, hB.size()*4); cudaMemcpy(sB, hB.data(), hB.size()*4, cudaMemcpyHostToDevice);
    cast_f32_to_bf16(sA, dA, hA.size()); cast_f32_to_bf16(sB, dB, hB.size());
    cublas_bf16_gemm(dA, tA, dB, tB, dC, M, N, K);
    float* sC; cudaMalloc(&sC, hC.size()*4);
    cast_bf16_to_f32(dC, sC, hC.size());
    std::vector<float> got(M*N); cudaMemcpy(got.data(), sC, got.size()*4, cudaMemcpyDeviceToHost);

    // compare
    bool ok=true; float maxerr=0; float maxref=0;
    for (size_t i=0;i<got.size();++i){
        float e=std::abs(got[i]-ref[i]); if(e>maxerr)maxerr=e;
        if (std::abs(ref[i])>maxref) maxref=std::abs(ref[i]);
    }
    // bf16 inputs round to ~3 sig digits; the absolute error of a K-wide dot
    // scales like max|ref| * sqrt(K) * 1e-2. Use that as the bound.
    float tol = maxref * std::sqrt((float)K) * 1.5e-2f + 1e-3f;
    if (maxerr > tol) ok=false;
    printf("GEMM tA=%d tB=%d M=%d N=%d K=%d: %s (maxerr=%.4f max|ref|=%.3f tol=%.4f)\n",
           tA,tB,M,N,K, ok?"OK":"FAIL", maxerr, maxref, tol);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(sA);cudaFree(sB);cudaFree(sC);
    return ok;
}

int main() {
    int nd=0; cudaGetDeviceCount(&nd); if(!nd){ printf("skip: no CUDA\n"); return 77; }
    bool ok = true;
    ok &= run_one(false, false, 16, 8, 12);
    ok &= run_one(false, true,  16, 8, 12);   // the Linear pattern: C = x @ W^T
    ok &= run_one(true,  false, 16, 8, 12);
    ok &= run_one(true,  true,  16, 8, 12);
    // a realistic block shape: [N_v=256, hidden=1536] x qkv_w^T[4608,1536]
    ok &= run_one(false, true,  256, 4608, 1536);
    printf("%s\n", ok?"OK gemm":"FAIL gemm");
    return ok?0:1;
}
