#include "image_preproc.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_PSD
#define STBI_NO_TGA
#define STBI_NO_GIF
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#include "stb_image.h"

// We resize with a hand-written bilinear because stb_image_resize2 changes its
// coefficients between versions and we need a deterministic match against the
// reference (the demo uses PIL/BILINEAR). Bilinear on uint8 RGB is exact when
// PIL's resample=BILINEAR is used on 'RGB' images — PIL computes the same
// precomputed coefficient tables we approximate below.
namespace {

inline int round_by_factor(int number, int factor) { return (int)std::lround((double)number / factor) * factor; }
inline int floor_by_factor(int number, int factor) { return (int)std::floor((double)number / factor) * factor; }
inline int ceil_by_factor(int number, int factor)  { return (int)std::ceil((double)number / factor) * factor; }

}  // namespace

namespace dots {

void smart_resize(int height, int width, int factor, int min_pixels,
                  int max_pixels, int& out_h, int& out_w) {
    double ar = (double)std::max(height, width) / std::min(height, width);
    if (ar > 200.0) {
        fprintf(stderr, "[smart_resize] aspect ratio %.2f > 200\n", ar);
    }
    int h_bar = std::max(factor, round_by_factor(height, factor));
    int w_bar = std::max(factor, round_by_factor(width, factor));
    if (h_bar * w_bar > max_pixels) {
        double beta = std::sqrt((double)(height * width) / max_pixels);
        h_bar = std::max(factor, floor_by_factor((int)(height / beta), factor));
        w_bar = std::max(factor, floor_by_factor((int)(width / beta), factor));
    } else if (h_bar * w_bar < min_pixels) {
        double beta = std::sqrt((double)min_pixels / (height * width));
        h_bar = ceil_by_factor((int)(height * beta), factor);
        w_bar = ceil_by_factor((int)(width * beta), factor);
        if (h_bar * w_bar > max_pixels) {
            double beta2 = std::sqrt((double)(h_bar * w_bar) / max_pixels);
            h_bar = std::max(factor, floor_by_factor((int)(h_bar / beta2), factor));
            w_bar = std::max(factor, floor_by_factor((int)(w_bar / beta2), factor));
        }
    }
    out_h = h_bar;
    out_w = w_bar;
}

// PIL.Image.BILINEAR resize on a uint8 RGB image. PIL's algorithm:
//   for each output pixel o, source center sx = (o + 0.5) * scale_in - 0.5
//   weights from the two nearest source pixels, clamped to [0, in_size-1].
// We precompute per-output-row and per-output-col coefficients.
static std::vector<unsigned char> bilinear_resize_rgb(const unsigned char* src,
                                                     int sh, int sw,
                                                     int dh, int dw) {
    std::vector<unsigned char> dst((size_t)dh * dw * 3);

    // Horizontal pass first: src[sh, sw, 3] -> tmp[sh, dw, 3]
    std::vector<float> x_coeff_lo(dw), x_coeff_hi(dw);
    std::vector<int>   x_idx_lo(dw), x_idx_hi(dw);
    double sx = (double)sw / dw;
    for (int o = 0; o < dw; ++o) {
        double center = (o + 0.5) * sx - 0.5;
        int lo = (int)std::floor(center);
        double frac = center - lo;
        x_coeff_lo[o] = float(1.0 - frac);
        x_coeff_hi[o] = float(frac);
        x_idx_lo[o] = std::max(0, std::min(sw - 1, lo));
        x_idx_hi[o] = std::max(0, std::min(sw - 1, lo + 1));
    }
    std::vector<float> tmp((size_t)sh * dw * 3);
    for (int y = 0; y < sh; ++y) {
        const unsigned char* row = src + (size_t)y * sw * 3;
        float* out = tmp.data() + (size_t)y * dw * 3;
        for (int o = 0; o < dw; ++o) {
            int li = x_idx_lo[o] * 3, hi = x_idx_hi[o] * 3;
            float wl = x_coeff_lo[o], wh = x_coeff_hi[o];
            out[o * 3 + 0] = row[li + 0] * wl + row[hi + 0] * wh;
            out[o * 3 + 1] = row[li + 1] * wl + row[hi + 1] * wh;
            out[o * 3 + 2] = row[li + 2] * wl + row[hi + 2] * wh;
        }
    }
    // Vertical pass: tmp[sh, dw, 3] -> dst[dh, dw, 3] (round to uint8 like PIL)
    std::vector<float> y_coeff_lo(dh), y_coeff_hi(dh);
    std::vector<int>   y_idx_lo(dh), y_idx_hi(dh);
    double sy = (double)sh / dh;
    for (int o = 0; o < dh; ++o) {
        double center = (o + 0.5) * sy - 0.5;
        int lo = (int)std::floor(center);
        double frac = center - lo;
        y_coeff_lo[o] = float(1.0 - frac);
        y_coeff_hi[o] = float(frac);
        y_idx_lo[o] = std::max(0, std::min(sh - 1, lo));
        y_idx_hi[o] = std::max(0, std::min(sh - 1, lo + 1));
    }
    for (int o = 0; o < dh; ++o) {
        int ly = y_idx_lo[o], hy = y_idx_hi[o];
        float wl = y_coeff_lo[o], wh = y_coeff_hi[o];
        const float* lo_row = tmp.data() + (size_t)ly * dw * 3;
        const float* hi_row = tmp.data() + (size_t)hy * dw * 3;
        unsigned char* drow = dst.data() + (size_t)o * dw * 3;
        for (int x = 0; x < dw * 3; ++x) {
            float v = lo_row[x] * wl + hi_row[x] * wh;
            // PIL rounds half away from zero on the final cast.
            int iv = (int)std::lround(v);
            if (iv < 0) iv = 0; if (iv > 255) iv = 255;
            drow[x] = (unsigned char)iv;
        }
    }
    return dst;
}

void preprocess_rgb(const unsigned char* rgb, int H, int W,
                    const PreprocParams& pp, ImageInput& out) {
    int P = pp.patch_size;
    int factor = pp.merge_size * P;   // 28
    int rh, rw;
    smart_resize(H, W, factor, pp.min_pixels, pp.max_pixels, rh, rw);

    // Resize to (rh, rw) via PIL-equivalent bilinear.
    std::vector<unsigned char> resized;
    if (rh == H && rw == W) {
        resized.assign(rgb, rgb + (size_t)H * W * 3);
    } else {
        resized = bilinear_resize_rgb(rgb, H, W, rh, rw);
    }

    int hp = rh / P;       // patch rows
    int wp = rw / P;       // patch cols
    out.t = 1; out.h = hp; out.w = wp;
    int Nv = hp * wp;
    out.pixel_values.assign((size_t)Nv * 588, 0.0f);

    // For each patch, fill [C=3, P, P] channel-first block.
    for (int py = 0; py < hp; ++py) {
        for (int px = 0; px < wp; ++px) {
            int patch_idx = py * wp + px;
            float* dst = out.pixel_values.data() + (size_t)patch_idx * 588;
            for (int c = 0; c < 3; ++c) {
                float mean = pp.image_mean[c];
                float stdv = pp.image_std[c];
                float* chan = dst + c * (P * P);   // 196
                for (int iy = 0; iy < P; ++iy) {
                    int gy = py * P + iy;
                    for (int ix = 0; ix < P; ++ix) {
                        int gx = px * P + ix;
                        unsigned char pv = resized[((size_t)gy * rw + gx) * 3 + c];
                        chan[iy * P + ix] = ((pv / 255.0f) - mean) / stdv;
                    }
                }
            }
        }
    }
}

bool preprocess_image_file(const std::string& path,
                           const PreprocParams& pp, ImageInput& out) {
    int w = 0, h = 0, c = 0;
    unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 3);  // force RGB
    if (!img) {
        fprintf(stderr, "[image] failed to load %s: %s\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    // stbi with req_comp=3 already gives us a packed RGB buffer; `c` is the
    // original channel count (unused).
    (void)c;
    preprocess_rgb(img, h, w, pp, out);
    stbi_image_free(img);
    return true;
}

}  // namespace dots
