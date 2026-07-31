// Image preprocessing for dots.mocr — byte-exact port of
// Qwen2VLImageProcessor + smart_resize.
//
// Pipeline (matches the HF processor the checkpoint ships with):
//   raw RGB → smart_resize(H,W, factor=28) → normalize(/255, mean, std)
//           → split into 14×14 patches → pack as [N_v, 588] float32
//
// The 588-d patch vector layout is channel-first inside the patch:
//   [R(14×14 row-major) | G(14×14) | B(14×14)]
// i.e. index = c*196 + py*14 + px. This is what HF's Qwen2VLImageProcessor
// emits (verified against the running demo: pv[:,0:196] all equal to the R
// normalised value, etc.), and the vision Conv2d consumes it directly.
//
// grid_thw = [t=1, h_patches, w_patches] with h = H/14, w = W/14.
#pragma once

#include <string>
#include <vector>

namespace dots {

struct ImageInput {
    std::vector<float> pixel_values;   // [N_v, 588], row-major
    int t = 1, h = 0, w = 0;           // grid_thw
    int N_v() const { return t * h * w; }
    int num_image_tokens() const { return N_v() / (2 * 2); }  // /m^2, m=2
};

struct PreprocParams {
    int   patch_size  = 14;
    int   merge_size  = 2;
    int   min_pixels  = 3136;
    int   max_pixels  = 11289600;
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
};

// Decode an image file (PNG/JPEG) with stb_image, RGB, then run the full
// pipeline. Returns false on decode failure.
bool preprocess_image_file(const std::string& path,
                           const PreprocParams& pp,
                           ImageInput& out);

// Same, from an already-decoded RGB buffer [H*W*3], row-major.
void preprocess_rgb(const unsigned char* rgb, int H, int W,
                    const PreprocParams& pp, ImageInput& out);

// smart_resize — reproduces utils/image_utils.smart_resize exactly:
// both dims divisible by factor(=28), pixels within [min,max], aspect kept.
void smart_resize(int height, int width, int factor, int min_pixels,
                  int max_pixels, int& out_h, int& out_w);

}  // namespace dots
