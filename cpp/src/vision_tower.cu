// Vision tower forward: patch embed -> 42 blocks (norm1->attn->residual,
// norm2->swiglu->residual) -> post-trunk norm -> merger.
//
// Memory plan (for N_v ~ 11k, the max-pixels case):
//   hidden x :        [N_v, 1536] bf16        ~ 34 MiB
//   qkv :            [N_v, 4608] bf16        ~ 102 MiB (q,k,v interleaved)
//   attn scratch q,k,v,out: [N_v,12,128] bf16 x4  ~ 136 MiB
//   mlp scratch up,gate,out: [N_v,4224] bf16 x3   ~ 283 MiB
// All reused across the 42 blocks; peak ~ 600 MiB plus weights ~1.2 GiB.
#include "vision_tower.h"
#include "kernels.h"
#include "tc_gemm.h"
#include "attention.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>

namespace dots {

namespace {
// Build the vision 2D-RoPE frequency table [N_v, head_dim/2] for one image
// with grid (h, w). Reproduces DotsVisionTransformer.rot_pos_emb:
//   pos_ids: for each patch (in merge-reorder), (hpos, wpos) coords
//   inv_freq[i] = 1/theta^(2i/(head_dim/2 * 2))? -> careful: VisionRotaryEmbedding
//     dim = head_dim//2 = 64; inv_freq[i] = 1/theta^(2i/64), i in [0,32)
//   freqs = outer(maxpos, inv_freq) shape [max_grid, 32]
//   rotary_pos_emb = freqs[pos_ids].flatten(1) -> [N_v, 64]
std::vector<float> build_vision_rope(const VisionConfig& cfg, int h_grid, int w_grid) {
    int m = cfg.spatial_merge_size;     // 2
    int half = cfg.head_dim() / 2;      // 64
    int P = cfg.patch_size;
    (void)P;
    // pos_ids via get_pos_ids_by_grid (hpos_ids, wpos_ids) with the
    // merge-order permutation.
    int h = h_grid, w = w_grid;
    // hpos: arange(h).expand(h,w) reshaped to [h/m, m, w/m, m].permute(0,2,1,3).flatten()
    std::vector<int> hpos(h * w), wpos(h * w);
    // build arange grids
    for (int hi = 0; hi < h; ++hi)
        for (int wi = 0; wi < w; ++wi) {
            hpos[hi * w + wi] = hi;     // arange(h).unsqueeze(1).expand(-1,w)
            wpos[hi * w + wi] = wi;
        }
    // reshape [h/m, m, w/m, m] and permute [0,2,1,3]
    auto merge_permute = [m, h, w](std::vector<int>& a) {
        std::vector<int> out(a.size());
        int hm = h / m, wm = w / m;
        for (int a0 = 0; a0 < hm; ++a0)
            for (int a2 = 0; a2 < wm; ++a2)
                for (int a1 = 0; a1 < m; ++a1)
                    for (int a3 = 0; a3 < m; ++a3) {
                        // input index in [h/m, m, w/m, m] layout:
                        // the original expand gives row-major [h, w]; reshape to
                        // [hm, m, wm, m] is contiguous in (hm,m,wm,m).
                        int in_idx = ((a0 * m + a1) * wm + a2) * m + a3;
                        int out_idx = ((a0 * wm + a2) * m + a1) * m + a3;
                        out[out_idx] = a[in_idx];
                    }
        a = out;
    };
    merge_permute(hpos);
    merge_permute(wpos);

    int max_grid = std::max(h, w);
    // inv_freq[i] = 1/theta^(2i/half), i in [0, half/2)
    std::vector<float> inv_freq(half / 2);
    for (int i = 0; i < half / 2; ++i)
        inv_freq[i] = 1.0f / std::pow(cfg.rope_theta, (2.0f * i) / float(half));
    // freqs table [max_grid, half/2]
    std::vector<std::vector<float>> freqs(max_grid, std::vector<float>(half / 2));
    for (int p = 0; p < max_grid; ++p)
        for (int i = 0; i < half / 2; ++i)
            freqs[p][i] = p * inv_freq[i];

    // rotary_pos_emb = freqs[pos_ids].flatten(1) -> [N_v, half]
    int Nv = h * w;
    std::vector<float> rope((size_t)Nv * half, 0.0f);
    for (int n = 0; n < Nv; ++n) {
        int hp = hpos[n], wp = wpos[n];
        for (int i = 0; i < half / 2; ++i) {
            rope[(size_t)n * half + i]              = freqs[hp][i];
            rope[(size_t)n * half + i + half / 2]   = freqs[wp][i];
        }
    }
    return rope;
}
}  // namespace

std::unique_ptr<VisionTower> VisionTower::load(const ModelWeights& w,
                                                const VisionConfig& cfg,
                                                int device) {
    auto tower = std::make_unique<VisionTower>();
    tower->cfg_ = cfg;
    tower->device_ = device;
    tower->w_ = std::make_unique<VisionTowerWeights>();

    // Compute total bytes, allocate one device buffer, carve offsets.
    auto bytes = [](const std::vector<int>& shape) -> size_t {
        size_t n = 1; for (int s : shape) n *= s; return n * sizeof(bf16);
    };
    // We reshape the conv weight [1536,3,14,14] to [1536,588] on host first.
    const int E = cfg.embed_dim;            // 1536
    const int I = cfg.intermediate_size;    // 4224
    const int patch_vec = 3 * cfg.patch_size * cfg.patch_size; // 588
    const int merge_hidden = E * cfg.spatial_merge_size * cfg.spatial_merge_size; // 6144

    size_t total = 0;
    total += E * patch_vec;        // proj_w [1536,588]
    total += E;                    // proj_b
    total += E;                    // patch_norm_w
    for (int b = 0; b < cfg.num_hidden_layers; ++b) {
        total += (3*E)*E;          // qkv
        total += E*E;              // proj
        total += I*E;              // fc1
        total += E*I;              // fc2
        total += I*E;              // fc3
        total += E;                // norm1
        total += E;                // norm2
    }
    total += E;                    // post_trunk_norm
    total += E; total += E;        // merger ln_q w, b
    total += merge_hidden*merge_hidden; total += merge_hidden; // mlp0 w,b
    total += E*merge_hidden; total += E;                          // mlp2 w,b

    DeviceBuffer& S = tower->w_->storage;
    S = DeviceBuffer(total * sizeof(bf16), device);
    bf16* base = (bf16*)S.ptr;
    size_t off = 0;
    auto take = [&](int n) -> bf16* { bf16* p = base + off; off += n; return p; };

    // Load conv weight and reshape to [1536, 588].
    {
        auto conv_shape = w.shape("vision_tower.patch_embed.patchifier.proj.weight");
        // expected [1536, 3, 14, 14]. Flatten the last three dims.
        std::vector<bf16> host(E * patch_vec);
        // Read host bf16 directly from the mmap'd shard.
        auto it = w.by_name.find("vision_tower.patch_embed.patchifier.proj.weight");
        if (it == w.by_name.end()) { fprintf(stderr, "[vision] missing conv weight\n"); return nullptr; }
        const TensorDescriptor& td = *it->second;
        if (td.nbytes != sizeof(bf16) * E * patch_vec) {
            fprintf(stderr, "[vision] conv weight size mismatch: %zu vs %zu\n",
                    td.nbytes, sizeof(bf16)*E*patch_vec);
        }
        std::memcpy(host.data(), td.host_ptr, td.nbytes);
        // PyTorch Conv2d.weight is [out, in, kH, kW] = [1536,3,14,14] row-major.
        // Our patch vector layout is [C, P, P] = [3,14,14] (channel-first), which
        // is exactly [in, kH, kW] flattened — so no reorder needed: flatten the
        // last three axes and the 588 index lines up.
        bf16* dst = take(E * patch_vec);
        DOTS_CUDA_CHECK(cudaMemcpy(dst, host.data(), sizeof(bf16)*E*patch_vec,
                                   cudaMemcpyHostToDevice));
        tower->w_->proj_w = dst;
    }
    tower->w_->proj_b = take(E);
    w.copy_to_device("vision_tower.patch_embed.patchifier.proj.bias", tower->w_->proj_b, Dtype::BF16);
    tower->w_->patch_norm_w = take(E);
    w.copy_to_device("vision_tower.patch_embed.patchifier.norm.weight", tower->w_->patch_norm_w, Dtype::BF16);

    char nm[128];
    for (int b = 0; b < cfg.num_hidden_layers; ++b) {
        auto& B = tower->w_->blocks[b];
        B.qkv_w = take(3*E*E);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.attn.qkv.weight", b);
        w.copy_to_device(nm, B.qkv_w, Dtype::BF16);
        B.proj_w = take(E*E);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.attn.proj.weight", b);
        w.copy_to_device(nm, B.proj_w, Dtype::BF16);
        B.fc1_w = take(I*E);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.mlp.fc1.weight", b);
        w.copy_to_device(nm, B.fc1_w, Dtype::BF16);
        B.fc2_w = take(E*I);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.mlp.fc2.weight", b);
        w.copy_to_device(nm, B.fc2_w, Dtype::BF16);
        B.fc3_w = take(I*E);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.mlp.fc3.weight", b);
        w.copy_to_device(nm, B.fc3_w, Dtype::BF16);
        B.norm1_w = take(E);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.norm1.weight", b);
        w.copy_to_device(nm, B.norm1_w, Dtype::BF16);
        B.norm2_w = take(E);
        snprintf(nm, sizeof(nm), "vision_tower.blocks.%d.norm2.weight", b);
        w.copy_to_device(nm, B.norm2_w, Dtype::BF16);
    }
    tower->w_->post_trunk_norm_w = take(E);
    w.copy_to_device("vision_tower.post_trunk_norm.weight", tower->w_->post_trunk_norm_w, Dtype::BF16);
    tower->w_->merger_ln_q_w = take(E);
    w.copy_to_device("vision_tower.merger.ln_q.weight", tower->w_->merger_ln_q_w, Dtype::BF16);
    tower->w_->merger_ln_q_b = take(E);
    w.copy_to_device("vision_tower.merger.ln_q.bias", tower->w_->merger_ln_q_b, Dtype::BF16);
    tower->w_->merger_mlp0_w = take(merge_hidden*merge_hidden);
    w.copy_to_device("vision_tower.merger.mlp.0.weight", tower->w_->merger_mlp0_w, Dtype::BF16);
    tower->w_->merger_mlp0_b = take(merge_hidden);
    w.copy_to_device("vision_tower.merger.mlp.0.bias", tower->w_->merger_mlp0_b, Dtype::BF16);
    tower->w_->merger_mlp2_w = take(E*merge_hidden);
    w.copy_to_device("vision_tower.merger.mlp.2.weight", tower->w_->merger_mlp2_w, Dtype::BF16);
    tower->w_->merger_mlp2_b = take(E);
    w.copy_to_device("vision_tower.merger.mlp.2.bias", tower->w_->merger_mlp2_b, Dtype::BF16);

    fprintf(stderr, "[vision] loaded %zu MiB weights, %d blocks\n",
            off * sizeof(bf16) / (1 << 20), cfg.num_hidden_layers);
    return tower;
}

DeviceTensor VisionTower::forward(const float* pixel_values_host, int N_v,
                                  int h_grid, int w_grid, int& out_n_img) {
    const VisionConfig& cfg = cfg_;
    const int E = cfg.embed_dim;            // 1536
    const int I = cfg.intermediate_size;    // 4224
    const int H = cfg.num_attention_heads;  // 12
    const int HD = cfg.head_dim();          // 128
    const int patch_vec = 3 * cfg.patch_size * cfg.patch_size; // 588

    DOTS_CUDA_CHECK(cudaSetDevice(device_));

    // 1) Patch embed: GEMM [N_v,588] x [588,1536]^T -> [N_v,1536] (+bias, RMSNorm)
    //    pixel_values come in as float32 host. Upload and cast to bf16 on device.
    DeviceTensor x_bf16(Dtype::BF16, N_v, patch_vec);
    {
        // cast host f32 -> device bf16 via a small kernel-free path: copy f32 to
        // device, then convert. We do the convert with a tiny kernel.
        DeviceBuffer f32buf(sizeof(float) * (size_t)N_v * patch_vec, device_);
        DOTS_CUDA_CHECK(cudaMemcpy(f32buf.ptr, pixel_values_host,
                                   f32buf.cap, cudaMemcpyHostToDevice));
        cast_f32_to_bf16(f32buf.ptr, x_bf16.ptr(), (size_t)N_v * patch_vec);
    }
    DeviceTensor hidden(Dtype::BF16, N_v, E);
    // GEMM: C = x_bf16 @ proj_w^T   -> [N_v, 1536]
    //   A = x_bf16 [N_v, 588], B = proj_w [1536, 588], we want B @ A^T? No:
    //   PyTorch Linear: out = x @ W^T, W is [out,in]=[1536,588]. So:
    //   out[n,o] = sum_i x[n,i] * W[o,i]  => C = x @ W^T.
    //   cublas_bf16_gemm(A, transA=false, B, transB=true, ...) with A=[N_v,588],
    //   B=[1536,588] transposed gives [N_v,1536]. But our helper takes (A,B,C)
    //   with row-major semantics: C = op(A)*op(B). We want A=[N_v,588] (no trans),
    //   B=[1536,588] (trans). 
    tc_gemm(x_bf16.ptr(), false,
            w_->proj_w, true,
            hidden.ptr(),
            N_v, E, patch_vec,
            Epilog::BIAS, w_->proj_b);
    rms_norm(hidden.ptr(), w_->patch_norm_w, hidden.ptr(), N_v, E, cfg.rms_norm_eps);

    // 2) Build the RoPE freq table [N_v, 64] and upload.
    std::vector<float> rope_host = build_vision_rope(cfg, h_grid, w_grid);
    DeviceTensor rope_freqs(Dtype::F32, N_v, HD / 2);
    DOTS_CUDA_CHECK(cudaMemcpy(rope_freqs.ptr(), rope_host.data(),
                               rope_freqs.nbytes(), cudaMemcpyHostToDevice));

    // 3) Scratch buffers for the 42 blocks (allocated once).
    DeviceTensor qkv(Dtype::BF16, N_v, 3 * E);                 // [N_v, 4608]
    DeviceTensor attn_out(Dtype::BF16, N_v, E);
    // q,k,v are views into unpacked layouts [N_v, H, HD]; allocate separately.
    DeviceTensor qbuf(Dtype::BF16, N_v, H * HD);
    DeviceTensor kbuf(Dtype::BF16, N_v, H * HD);
    DeviceTensor vbuf(Dtype::BF16, N_v, H * HD);
    DeviceTensor normed(Dtype::BF16, N_v, E);
    DeviceTensor proj_out(Dtype::BF16, N_v, E);

    DeviceTensor mlp_gate(Dtype::BF16, N_v, I);  // fc1 with silu
    DeviceTensor mlp_up(Dtype::BF16, N_v, I);    // fc3
    DeviceTensor mlp_hidden(Dtype::BF16, N_v, I); // silu(gate)*up
    DeviceTensor mlp_out(Dtype::BF16, N_v, E);

    for (int b = 0; b < cfg.num_hidden_layers; ++b) {
        const auto& B = w_->blocks[b];
        // --- attention sublayer ---
        rms_norm(hidden.ptr(), B.norm1_w, normed.ptr(), N_v, E, cfg.rms_norm_eps);
        // qkv = normed @ qkv_w^T  -> [N_v, 4608]
        tc_gemm(normed.ptr(), false, B.qkv_w, true,
                qkv.ptr(), N_v, 3 * E, E, Epilog::NONE);
        // split qkv (interleaved [3, N_v, H, HD] -> [N_v, H, HD] each) and RoPE.
        split_qkv_vision(qkv.ptr(), qbuf.ptr(), kbuf.ptr(), vbuf.ptr(), N_v, H, HD);
        apply_rotary_vision(qbuf.ptr(), kbuf.ptr(), N_v, H, HD, (const float*)rope_freqs.ptr());
        // attention (full, non-causal)
        flash_attention(qbuf.ptr(), kbuf.ptr(), vbuf.ptr(), attn_out.ptr(),
                        N_v, H, HD, /*is_causal=*/false);
        // proj
        tc_gemm(attn_out.ptr(), false, B.proj_w, true,
                proj_out.ptr(), N_v, E, E, Epilog::NONE);
        // residual
        add(hidden.ptr(), proj_out.ptr(), hidden.ptr(), (size_t)N_v * E);

        // --- MLP sublayer ---
        rms_norm(hidden.ptr(), B.norm2_w, normed.ptr(), N_v, E, cfg.rms_norm_eps);
        // fc1 -> gate (silu applied later), fc3 -> up
        tc_gemm(normed.ptr(), false, B.fc1_w, true,
                mlp_gate.ptr(), N_v, I, E, Epilog::NONE);
        tc_gemm(normed.ptr(), false, B.fc3_w, true,
                mlp_up.ptr(), N_v, I, E, Epilog::NONE);
        silu_inplace(mlp_gate.ptr(), (size_t)N_v * I);
        swiglu(mlp_gate.ptr(), mlp_up.ptr(), mlp_hidden.ptr(), (size_t)N_v * I);
        tc_gemm(mlp_hidden.ptr(), false, B.fc2_w, true,
                mlp_out.ptr(), N_v, E, I, Epilog::NONE);
        add(hidden.ptr(), mlp_out.ptr(), hidden.ptr(), (size_t)N_v * E);
    }

    // 4) post-trunk norm + merger.
    rms_norm(hidden.ptr(), w_->post_trunk_norm_w, hidden.ptr(), N_v, E, cfg.rms_norm_eps);

    int m = cfg.spatial_merge_size;       // 2
    int N_img = N_v / (m * m);
    out_n_img = N_img;
    // Merger (PatchMerger.forward): ln_q is LayerNorm(context_dim=1536) applied
    // to the full N_v rows FIRST, then .view(-1, 4*1536) groups consecutive-4
    // rows. pixel_values already arrive in merge-permuted order (patch index
    // 0..3 = the (0,0),(0,1),(1,0),(1,1) 2x2 cell — verified against the demo),
    // so consecutive-4 IS the spatial cell: no gather needed.
    layer_norm(hidden.ptr(), w_->merger_ln_q_w, w_->merger_ln_q_b,
               hidden.ptr(), N_v, E, 1e-6f);
    // Reshape view: [N_v, 1536] -> [N_img, 6144] is just a reinterpretation of
    // the same memory (4 consecutive rows of 1536 concatenate to one 6144 row).
    Tensor merged_view(hidden.ptr(), Dtype::BF16, N_img, E * m * m);
    // mlp: Linear(6144,6144)+bias -> GELU -> Linear(6144,1536)+bias
    // Both bias+GELU/bias fused into the tc_gemm epilogue (no extra HBM round-trip).
    DeviceTensor mh(Dtype::BF16, N_img, E * m * m);
    tc_gemm(merged_view.data, false, w_->merger_mlp0_w, true,
            mh.ptr(), N_img, E * m * m, E * m * m, Epilog::GELU, w_->merger_mlp0_b);
    DeviceTensor img(Dtype::BF16, N_img, E);
    tc_gemm(mh.ptr(), false, w_->merger_mlp2_w, true,
            img.ptr(), N_img, E, E * m * m, Epilog::BIAS, w_->merger_mlp2_b);

    return img;
}

}  // namespace dots
