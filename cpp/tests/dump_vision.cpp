// Dump the C++ engine's vision embeddings (E_img) for a captured
// pixel_values.bin, so they can be compared against the HF reference e_img.npy.
//   dump_vision <ckpt> <ref_dir>
#include "vision_tower.h"
#include "config.h"
#include "safetensors_loader.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace dots;

int main(int argc, char** argv) {
    std::string ckpt = argv[1];
    std::string refdir = argv[2];

    int t=1,h=0,w=0;
    { std::ifstream f(refdir + "/grid_thw.txt"); f >> t >> h >> w; }
    int N_v = t*h*w;
    std::vector<float> pv((size_t)N_v*588);
    { std::ifstream f(refdir + "/pixel_values.bin", std::ios::binary);
      f.read((char*)pv.data(), pv.size()*4); }

    auto cfg = ModelConfig::load(ckpt);
    auto weights = ModelWeights::load(ckpt, 0);
    auto vt = VisionTower::load(weights, cfg.vision, 0);
    int n_img = 0;
    auto e = vt->forward(pv.data(), N_v, h, w, n_img);
    cudaDeviceSynchronize();

    // dump to host
    std::vector<float> host((size_t)n_img * cfg.vision.embed_dim);
    bf16* scratch; cudaMalloc(&scratch, host.size()*2);
    cast_bf16_to_f32(e.ptr(), scratch, host.size());
    // stage via device f32
    float* df32; cudaMalloc(&df32, host.size()*4);
    cast_bf16_to_f32(e.ptr(), df32, host.size());
    cudaMemcpy(host.data(), df32, host.size()*4, cudaMemcpyDeviceToHost);
    cudaFree(scratch); cudaFree(df32);

    printf("E_img shape: %d x %d\n", n_img, cfg.vision.embed_dim);
    printf("E_img[0,:8]:"); for (int i=0;i<8;++i) printf(" %.5f", host[i]); printf("\n");
    double mean=0; for (float v: host) mean += v; mean /= host.size();
    double sq=0; for (float v: host) sq += (v-mean)*(v-mean); double std=sqrt(sq/host.size());
    printf("E_img mean/std: %.5f %.5f\n", mean, std);
    // write raw bin
    FILE* f = fopen("/work/cpp/my_e_img.bin", "wb");
    fwrite(host.data(), sizeof(float), host.size(), f); fclose(f);
    printf("wrote /work/cpp/my_e_img.bin\n");
    return 0;
}
