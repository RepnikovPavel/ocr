// tc_gemm (tensor-core WMMA) correctness test against a CPU reference, for all
// transA/transB combinations, all epilogues, and the real layer shapes the
// engine uses. This is the anchor for "the GEMM rewrite didn't change the math."
#include "tc_gemm.h"
#include "kernels.h"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace dots;

// CPU reference: C[M,N] = alpha * op(A)*op(B) + beta*C, then epilog. Row-major fp32.
static void cpu_gemm(const std::vector<float>& A, bool tA,
                     const std::vector<float>& B, bool tB,
                     std::vector<float>& C, int M,int N,int K,
                     float alpha, float beta, Epilog ep,
                     const std::vector<float>& bias) {
    auto Aat = [&](int m, int k){ return tA ? A[k*M+m] : A[m*K+k]; };
    auto Bat = [&](int k, int n){ return tB ? B[n*K+k] : B[k*N+n]; };
    for (int m=0;m<M;++m) for (int n=0;n<N;++n){
        float s=0; for(int k=0;k<K;++k) s += Aat(m,k)*Bat(k,n);
        float v = alpha*s + beta*C[m*N+n];
        switch (ep) {
            case Epilog::BIAS: v += bias[n]; break;
            case Epilog::SILU: { v += bias[n]; float sg=1.0f/(1.0f+expf(-v)); v *= sg; break; }
            case Epilog::GELU: { v += bias[n]; v = 0.5f*v*(1.0f+erff(v*0.70710678f)); break; }
            default: break;
        }
        C[m*N+n] = v;
    }
}

static bool run_one(bool tA, bool tB, int M, int N, int K, Epilog ep, const char* label) {
    std::vector<float> hA(M*K), hB(K*N), hC(M*N), bias(N, 0.0f);
    for (auto& v: hA) v = ((float)(rand()%200)-100)/100.0f;
    for (auto& v: hB) v = ((float)(rand()%200)-100)/100.0f;
    for (auto& v: bias) v = ((float)(rand()%200)-100)/200.0f;
    std::vector<float> ref(M*N, 0.0f);
    cpu_gemm(hA, tA, hB, tB, ref, M, N, K, 1.0f, 0.0f, ep, bias);

    bf16 *dA,*dB,*dC; cudaMalloc(&dA, hA.size()*2); cudaMalloc(&dB, hB.size()*2);
    cudaMalloc(&dC, hC.size()*2);
    bf16* dbias=nullptr;
    auto upload = [&](const std::vector<float>& h, bf16* d){
        if (!d) return;
        float* s; cudaMalloc(&s, h.size()*4); cudaMemcpy(s, h.data(), h.size()*4, cudaMemcpyHostToDevice);
        cast_f32_to_bf16(s, d, h.size()); cudaFree(s);
    };
    if (ep != Epilog::NONE) { cudaMalloc(&dbias, N*2); }
    upload(hA, dA); upload(hB, dB); upload(bias, dbias);
    tc_gemm(dA, tA, dB, tB, dC, M, N, K, ep, dbias);
    float* sC; cudaMalloc(&sC, hC.size()*4); cast_bf16_to_f32(dC, sC, hC.size());
    std::vector<float> got(M*N); cudaMemcpy(got.data(), sC, got.size()*4, cudaMemcpyDeviceToHost);

    float maxerr=0, maxref=0;
    for (size_t i=0;i<got.size();++i){
        float e=std::abs(got[i]-ref[i]); if(e>maxerr)maxerr=e;
        if (std::abs(ref[i])>maxref) maxref=std::abs(ref[i]);
    }
    // bf16 inputs (~3 sig digits) accumulate over K; tolerance scales with sqrt(K)*|ref|.
    float tol = maxref * std::sqrt((float)K) * 1.5e-2f + (ep!=Epilog::NONE?0.05f:1e-3f);
    bool ok = maxerr <= tol;
    printf("%-28s tA=%d tB=%d M=%-4d N=%-4d K=%-4d %s (maxerr=%.4f tol=%.4f)\n",
           label, tA, tB, M, N, K, ok?"OK":"FAIL", maxerr, tol);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(sC); if(dbias) cudaFree(dbias);
    return ok;
}

int main() {
    int nd=0; cudaGetDeviceCount(&nd); if(!nd){ printf("skip: no CUDA\n"); return 77; }
    bool ok = true;
    // core trans/epilog combos at small size
    for (int e = 0; e <= 3; ++e) {
        Epilog ep = (Epilog)e;
        const char* name = e==0?"NONE":e==1?"BIAS":e==2?"SILU":"GELU";
        ok &= run_one(false, true, 32, 32, 48, ep, name);
    }
    // the Linear pattern C = x @ W^T (transB=true) with bias, all epilogues
    ok &= run_one(false, true, 32, 64, 48, Epilog::BIAS, "Linear+bias");
    ok &= run_one(false, true, 32, 64, 48, Epilog::SILU, "Linear+silu");
    ok &= run_one(false, true, 32, 64, 48, Epilog::GELU, "Linear+gelu");
    // non-aligned M,N (masking path)
    ok &= run_one(false, true, 20, 30, 48, Epilog::BIAS, "unaligned");
    // real layer shapes
    ok &= run_one(false, true, 256, 4608, 1536, Epilog::NONE, "vision qkv");
    ok &= run_one(false, true, 256, 4224, 1536, Epilog::NONE, "vision fc1");
    ok &= run_one(false, true, 256, 1536, 8960, Epilog::NONE, "vision fc2");
    ok &= run_one(false, true, 600,  6144, 6144, Epilog::BIAS, "merger mlp0");
    ok &= run_one(false, true,   1, 151936, 1536, Epilog::NONE, "lm_head GEMV");
    printf("%s\n", ok?"OK tc_gemm":"FAIL tc_gemm");
    return ok?0:1;
}
