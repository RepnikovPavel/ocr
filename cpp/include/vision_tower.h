// Vision tower (DotsVisionTransformer): patch embed -> 42 transformer blocks
// -> post-trunk norm -> patch merger. Produces N_img image-token embeddings
// in the LLM hidden size (1536).
//
// Weight layout (from the safetensors checkpoint, all bf16, NO bias on
// qkv/proj — vision use_bias=false):
//   vision_tower.patch_embed.patchifier.proj.weight : [1536, 3, 14, 14]  (Conv2d)
//   vision_tower.patch_embed.patchifier.proj.bias   : [1536]             (Conv2d bias)
//   vision_tower.patch_embed.patchifier.norm.weight : [1536]             (RMSNorm)
//   vision_tower.blocks.{0..41}.attn.qkv.weight     : [4608, 1536]
//   vision_tower.blocks.{0..41}.attn.proj.weight    : [1536, 1536]
//   vision_tower.blocks.{0..41}.mlp.fc1.weight      : [4224, 1536]
//   vision_tower.blocks.{0..41}.mlp.fc2.weight      : [1536, 4224]
//   vision_tower.blocks.{0..41}.mlp.fc3.weight      : [4224, 1536]
//   vision_tower.blocks.{0..41}.norm1.weight        : [1536]
//   vision_tower.blocks.{0..41}.norm2.weight        : [1536]
//   vision_tower.post_trunk_norm.weight             : [1536]
//   vision_tower.merger.ln_q.{weight,bias}          : [1536]   (LayerNorm)
//   vision_tower.merger.mlp.0.{weight,bias}         : [6144, 6144]
//   vision_tower.merger.mlp.2.{weight,bias}         : [1536, 6144]
//
// The patch embed Conv2d(k=14,s=14) over a [N_v,3,14,14] "patch image" is a
// per-patch linear: y[n,o] = sum_{c,i,j} W[o,c,i,j] * x[n,c,i,j]. Since our
// pixel_values are already sliced into 588-d patch vectors laid out as
// [C, P, P] (channel-first), the conv reduces to a single GEMM
// W_conv[1536, 588] @ pixel[N_v, 588]^T -> [N_v, 1536], with the Conv2d bias
// added and RMSNorm applied. We build that 588-row view of W on load.
#pragma once

#include "config.h"
#include "safetensors_loader.h"
#include "tensor.h"

#include <memory>

namespace dots {

struct VisionTowerWeights {
    // Patch embed: conv weight reshaped to [out=1536, in=588].
    bf16* proj_w = nullptr;   // [1536, 588]
    bf16* proj_b = nullptr;   // [1536]
    bf16* patch_norm_w = nullptr; // [1536]

    static constexpr int NUM_BLOCKS = 42;
    struct Block {
        bf16* qkv_w;          // [4608, 1536]
        bf16* proj_w;         // [1536, 1536]
        bf16* fc1_w;          // [4224, 1536]
        bf16* fc2_w;          // [1536, 4224]
        bf16* fc3_w;          // [4224, 1536]
        bf16* norm1_w;        // [1536]
        bf16* norm2_w;        // [1536]
    };
    Block blocks[NUM_BLOCKS];

    bf16* post_trunk_norm_w = nullptr;  // [1536]
    bf16* merger_ln_q_w = nullptr;      // [1536]
    bf16* merger_ln_q_b = nullptr;      // [1536]
    bf16* merger_mlp0_w = nullptr;      // [6144, 6144]
    bf16* merger_mlp0_b = nullptr;      // [6144]
    bf16* merger_mlp2_w = nullptr;      // [1536, 6144]
    bf16* merger_mlp2_b = nullptr;      // [1536]

    DeviceBuffer storage;  // owns all device memory above
};

class VisionTower {
public:
    static std::unique_ptr<VisionTower> load(const ModelWeights& w,
                                             const VisionConfig& cfg,
                                             int device = 0);
    // Run the encoder. pixel_values: host float32 [N_v, 588]. Returns the
    // image-token embeddings [N_img, hidden] on device, where N_img = N_v/4.
    // Writes N_img into out_n_img.
    DeviceTensor forward(const float* pixel_values_host, int N_v,
                         int h_grid, int w_grid, int& out_n_img);
    const VisionConfig& config() const { return cfg_; }

private:
    VisionConfig cfg_;
    std::unique_ptr<VisionTowerWeights> w_;
    int device_ = 0;

    // persistent device scratch (reused across calls of similar size)
    DeviceBuffer rope_freqs_;
};

}  // namespace dots
