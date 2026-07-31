// Image preprocessing test: smart_resize correctness + a synthetic-image
// normalisation check against the known reference values.
#include "image_preproc.h"
#include <cassert>
#include <cstdio>
#include <vector>

using namespace dots;

int main() {
    // smart_resize: 800x600, factor=28, default min/max.
    // 800 -> round(800/28)*28 = round(28.57)*28 = 29*28 = 812
    // 600 -> round(600/28)*28 = round(21.43)*28 = 21*28 = 588
    // product 812*588 = 477456 < max(11.29M) and > min -> stays.
    int rh, rw;
    smart_resize(600, 800, 28, 3136, 11289600, rh, rw);
    assert(rh == 588);
    assert(rw == 812);

    // A uniform RGB image must normalise to the exact channel values.
    PreprocParams pp;
    std::vector<unsigned char> rgb(56 * 84 * 3);  // 4x6 patches
    for (auto& c : rgb) c = 123;
    ImageInput out;
    preprocess_rgb(rgb.data(), 56, 84, pp, out);
    assert(out.h == 4 && out.w == 6);
    assert(out.N_v() == 24);
    // patch 0, channel 0 (R), pixel 0:
    float expect = (123.0f / 255.0f - pp.image_mean[0]) / pp.image_std[0];
    float got = out.pixel_values[0];
    assert(std::abs(got - expect) < 1e-4);

    printf("OK imageproc: smart_resize + normalise, N_v=%d\n", out.N_v());
    return 0;
}
