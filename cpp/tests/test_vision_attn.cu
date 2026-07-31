// Test apply_rotary_vision + flash_attention together against a CPU reference,
// with the exact shapes the vision tower uses (n_heads=12, hd=128). This is the
// combo that diverged from HF; a standalone check isolates whether the bug is
// in the rope->flash data flow or elsewhere.
#include "kernels.h"
#include "attention.h"
#include <cmath>
#include <cstdio>
#include <vector>
using namespace dots;

int main() {
    const int Nv = 8, H = 12, HD = 128, half = HD/2;
    // local cast helpers (host<->device, via device scratch)
    auto h2d = [](const std::vector<float>& host, void* dev){
        float* s; cudaMalloc(&s, host.size()*4);
        cudaMemcpy(s, host.data(), host.size()*4, cudaMemcpyHostToDevice);
        cast_f32_to_bf16(s, dev, host.size()); cudaFree(s);
    };
    auto d2h = [](void* dev, int n){
        float* s; cudaMalloc(&s, n*4); cast_bf16_to_f32(dev, s, n);
        std::vector<float> out(n); cudaMemcpy(out.data(), s, n*4, cudaMemcpyDeviceToHost); cudaFree(s);
        return out;
    };
    std::vector<float> hq(Nv*H*HD), hk(Nv*H*HD), hv(Nv*H*HD);
    for (auto& v: hq) v = ((float)(rand()%200)-100)/200.0f;
    for (auto& v: hk) v = ((float)(rand()%200)-100)/200.0f;
    for (auto& v: hv) v = ((float)(rand()%200)-100)/200.0f;

    // build rope freqs [Nv, half] the way build_vision_rope would (all-zero patch0,
    // identity-ish for others). Use a simple per-position angle.
    std::vector<float> freqs(Nv*half);
    for (int n=0;n<Nv;++n) for(int i=0;i<half;++i) freqs[n*half+i] = n * (1.0f/(powf(10000.0f, (2.0f*i)/half)));

    // CPU reference: rope(q), rope(k), then softmax(QK^T/sqrt(HD)) V per head.
    auto rope_apply = [&](std::vector<float>& x){
        std::vector<float> out(x.size());
        for (int n=0;n<Nv;++n) for(int h=0;h<H;++h){
            for(int ph=0;ph<half;++ph){
                float c=cosf(freqs[n*half+ph]), s=sinf(freqs[n*half+ph]);
                float q0=x[((n*H+h)*HD)+ph], q1=x[((n*H+h)*HD)+half+ph];
                out[((n*H+h)*HD)+ph]=q0*c-q1*s;
                out[((n*H+h)*HD)+half+ph]=q1*c+q0*s;
            }
        }
        return out;
    };
    auto qr = rope_apply(hq), kr = rope_apply(hk);
    std::vector<float> ref(Nv*H*HD, 0.0f);
    float scale = 1.0f/sqrtf((float)HD);
    for (int n=0;n<Nv;++n) for(int h=0;h<H;++h){
        float sc[16]; float mx=-1e30f;
        for (int k=0;k<Nv;++k){
            float s=0; for(int d=0;d<HD;++d) s+=qr[((n*H+h)*HD)+d]*kr[((k*H+h)*HD)+d];
            s*=scale; sc[k]=s; if(s>mx)mx=s;
        }
        float sum=0; float ex[16];
        for(int k=0;k<Nv;++k){ex[k]=expf(sc[k]-mx); sum+=ex[k];}
        for(int d=0;d<HD;++d){float o=0; for(int k=0;k<Nv;++k) o+=ex[k]*hv[((k*H+h)*HD)+d]; ref[((n*H+h)*HD)+d]=o/sum;}
    }

    // GPU: upload, rope, flash.
    bf16 *dq,*dk,*dv,*dout; float* dfreq;
    cudaMalloc(&dq, hq.size()*2); cudaMalloc(&dk, hk.size()*2); cudaMalloc(&dv, hv.size()*2);
    cudaMalloc(&dout, hq.size()*2); cudaMalloc(&dfreq, freqs.size()*4);
    h2d(hq, dq); h2d(hk, dk); h2d(hv, dv);
    cudaMemcpy(dfreq, freqs.data(), freqs.size()*4, cudaMemcpyHostToDevice);
    apply_rotary_vision(dq, dk, Nv, H, HD, dfreq);
    flash_attention(dq, dk, dv, dout, Nv, H, HD, false);
    auto got = d2h(dout, hq.size());

    // compare
    float maxerr=0; for(size_t i=0;i<got.size();++i){float e=fabsf(got[i]-ref[i]); if(e>maxerr)maxerr=e;}
    bool ok = maxerr < 0.05f;
    printf("rope+flash vs CPU: %s (maxerr=%.5f)\n", ok?"OK":"FAIL", maxerr);
    if(!ok){ printf("ref[0,:6]:"); for(int i=0;i<6;++i) printf(" %.4f", ref[i]); printf("\n");
             printf("got[0,:6]:"); for(int i=0;i<6;++i) printf(" %.4f", got[i]); printf("\n"); }
    return ok?0:1;
}
