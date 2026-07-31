// Load the HF-captured q,k,v (after rope) for vision block 0, run our flash
// attention on them, and compare to HF's saved sdpa attn_out. This isolates
// whether the flash kernel diverges on real (non-random) data.
#include "kernels.h"
#include "attention.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
using namespace dots;

static std::vector<float> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    size_t n = raw.size()/sizeof(float);
    std::vector<float> out(n);
    memcpy(out.data(), raw.data(), n*sizeof(float));
    return out;
}

int main(int argc, char** argv) {
    std::string dir = argc>1 ? argv[1] : "/work/cpp/ref";
    int Nv=64, H=12, HD=128;
    auto q = load_bin(dir+"/q2.bin");
    auto k = load_bin(dir+"/k2.bin");
    auto v = load_bin(dir+"/v.bin");
    auto ref = load_bin(dir+"/attn_out.bin");
    int nq=q.size(), nk=k.size(), nv=v.size(), nref=ref.size();
    fprintf(stderr, "loaded q=%d k=%d v=%d ref=%d\n", nq,nk,nv,nref);
    bf16 *dq,*dk,*dv,*dout;
    cudaMalloc(&dq, nq*2); cudaMalloc(&dk, nk*2); cudaMalloc(&dv, nv*2); cudaMalloc(&dout, nq*2);
    float* sq; cudaMalloc(&sq, nq*4); cudaMemcpy(sq, q.data(), nq*4, cudaMemcpyHostToDevice); cast_f32_to_bf16(sq, dq, nq); cudaFree(sq);
    float* sk; cudaMalloc(&sk, nk*4); cudaMemcpy(sk, k.data(), nk*4, cudaMemcpyHostToDevice); cast_f32_to_bf16(sk, dk, nk); cudaFree(sk);
    float* sv; cudaMalloc(&sv, nv*4); cudaMemcpy(sv, v.data(), nv*4, cudaMemcpyHostToDevice); cast_f32_to_bf16(sv, dv, nv); cudaFree(sv);
    flash_attention(dq, dk, dv, dout, Nv, H, HD, false);
    float* sout; cudaMalloc(&sout, nq*4); cast_bf16_to_f32(dout, sout, nq);
    std::vector<float> got(nq); cudaMemcpy(got.data(), sout, nq*4, cudaMemcpyDeviceToHost); cudaFree(sout);

    float maxerr=0; for(int i=0;i<nq;++i){float e=fabsf(got[i]-ref[i]); if(e>maxerr)maxerr=e;}
    bool ok = maxerr < 0.05f;
    printf("flash on real qkv vs HF sdpa: %s (maxerr=%.5f)\n", ok?"OK":"FAIL", maxerr);
    printf("got[0,:6]:"); for(int i=0;i<6;++i) printf(" %.5f", got[i]); printf("\n");
    printf("ref[0,:6]:"); for(int i=0;i<6;++i) printf(" %.5f", ref[i]); printf("\n");
    printf("got[0,128:132] (head1):"); for(int i=128;i<132;++i) printf(" %.5f", got[i]); printf("\n");
    printf("ref[0,128:132] (head1):"); for(int i=128;i<132;++i) printf(" %.5f", ref[i]); printf("\n");
    return ok?0:1;
}
