// CUDA kernel tests: small numerical checks against CPU references for the
// fused elementwise ops, RMSNorm, RoPE, and the flash/decode attention.
// These run on the GPU and need a CUDA-capable device.
#include "kernels.h"
#include "attention.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace dots;

static bool close(float a, float b, float tol = 2e-3) { return std::abs(a - b) <= tol + 2e-3 * std::abs(b); }

// Cast a device bf16 tensor to a host float vector (via a device f32 scratch).
// The engine's cast kernels read/write device memory; writing straight into a
// host pointer from a kernel is undefined, so we stage through a device buffer.
static std::vector<float> bf16_dev_to_host_f32(const void* dev, int n) {
    float* scratch; cudaMalloc(&scratch, n * sizeof(float));
    cast_bf16_to_f32(dev, scratch, n);
    std::vector<float> out(n);
    cudaMemcpy(out.data(), scratch, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(scratch);
    return out;
}
static void f32_host_to_bf16_dev(const std::vector<float>& host, void* dev) {
    float* scratch; cudaMalloc(&scratch, host.size() * sizeof(float));
    cudaMemcpy(scratch, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    cast_f32_to_bf16(scratch, dev, host.size());
    cudaFree(scratch);
}

int main() {
    int nDev = 0;
    cudaGetDeviceCount(&nDev);
    if (nDev == 0) { fprintf(stderr, "skip: no CUDA device\n"); return 77; }

    // --- RMSNorm vs a trivial CPU reference ---
    {
        int rows = 4, dim = 8;
        std::vector<float> hx(rows * dim), hw(dim, 1.0f), hout(rows * dim, 0);
        for (auto& v : hx) v = ((float)(rand() % 200) - 100) / 50.0f;
        bf16 *dx, *dw, *dout;
        cudaMalloc(&dx, hx.size() * sizeof(bf16));
        cudaMalloc(&dw, dim * sizeof(bf16));
        cudaMalloc(&dout, hx.size() * sizeof(bf16));
        f32_host_to_bf16_dev(hx, dx);
        f32_host_to_bf16_dev(hw, dw);
        rms_norm(dx, dw, dout, rows, dim, 1e-6f);
        hout = bf16_dev_to_host_f32(dout, hx.size());
        // CPU ref
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            float mean_sq = 0;
            for (int i = 0; i < dim; ++i) mean_sq += hx[r*dim+i] * hx[r*dim+i];
            mean_sq /= dim;
            float rs = 1.0f / std::sqrt(mean_sq + 1e-6f);
            for (int i = 0; i < dim; ++i) {
                float ref = hx[r*dim+i] * rs * hw[i];
                if (!close(ref, hout[r*dim+i], 5e-3)) ok = false;
            }
        }
        printf("rms_norm: %s\n", ok ? "OK" : "FAIL");
        cudaFree(dx); cudaFree(dw); cudaFree(dout);
        if (!ok) return 1;
    }

    // --- flash attention (non-causal) matches a naive CPU softmax(QK^T/sqrt(d))V ---
    {
        int seq = 32, n_heads = 2, hd = 128;
        std::vector<float> hq(seq*n_heads*hd), hk(seq*n_heads*hd), hv(seq*n_heads*hd);
        for (auto& v : hq) v = ((float)(rand()%100))/100.0f - 0.5f;
        for (auto& v : hk) v = ((float)(rand()%100))/100.0f - 0.5f;
        for (auto& v : hv) v = ((float)(rand()%100))/100.0f - 0.5f;
        bf16 *dq,*dk,*dv,*dout;
        cudaMalloc(&dq, hq.size()*sizeof(bf16)); cudaMalloc(&dk, hk.size()*sizeof(bf16));
        cudaMalloc(&dv, hv.size()*sizeof(bf16)); cudaMalloc(&dout, hq.size()*sizeof(bf16));
        f32_host_to_bf16_dev(hq, dq); f32_host_to_bf16_dev(hk, dk);
        f32_host_to_bf16_dev(hv, dv);
        flash_attention(dq,dk,dv,dout,seq,n_heads,hd,false);
        std::vector<float> hout = bf16_dev_to_host_f32(dout, hq.size());
        // CPU reference for head 0
        bool ok = true;
        float scale = 1.0f/std::sqrt((float)hd);
        for (int q = 0; q < seq && ok; ++q) for (int h = 0; h < n_heads && ok; ++h) {
            // attention weights over all k
            std::vector<float> sc(seq), ex(seq); float mx=-1e30f;
            for (int k = 0; k < seq; ++k) {
                float s=0; for (int d=0;d<hd;++d) s += hq[((q*n_heads+h)*hd)+d]*hk[((k*n_heads+h)*hd)+d];
                s*=scale; sc[k]=s; if (s>mx) mx=s;
            }
            float sum=0; for (int k=0;k<seq;++k){ex[k]=expf(sc[k]-mx);sum+=ex[k];}
            for (int d=0;d<hd && ok;++d) {
                float o=0; for (int k=0;k<seq;++k) o += ex[k]*hv[((k*n_heads+h)*hd)+d];
                o/=sum;
                if (!close(o, hout[((q*n_heads+h)*hd)+d], 3e-2)) ok=false;
            }
        }
        printf("flash_attention(non-causal): %s\n", ok?"OK":"FAIL");
        cudaFree(dq);cudaFree(dk);cudaFree(dv);cudaFree(dout);
        if (!ok) return 1;
    }

    printf("OK kernels\n");
    return 0;
}
