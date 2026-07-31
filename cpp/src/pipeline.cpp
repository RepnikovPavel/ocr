// Pipeline implementation. See pipeline.h.
#include "pipeline.h"
#include "image_preproc.h"
#include "kernels.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace dots {

using clk = std::chrono::steady_clock;
template <typename T> static double ms_since(T start) {
    return std::chrono::duration<double, std::milli>(clk::now() - start).count();
}

std::unique_ptr<Pipeline> Pipeline::create(const std::string& ckpt_dir,
                                           int device, int max_seq) {
    auto p = std::make_unique<Pipeline>();
    p->device_ = device;
    p->max_seq_ = max_seq;

    DOTS_CUDA_CHECK(cudaSetDevice(device));
    p->cfg_ = ModelConfig::load(ckpt_dir);

    fprintf(stderr, "[pipeline] loading weights from %s\n", ckpt_dir.c_str());
    ModelWeights weights = ModelWeights::load(ckpt_dir, device);
    if (weights.tensors.empty()) { fprintf(stderr, "[pipeline] weight load failed\n"); return nullptr; }

    p->tokenizer_ = std::make_unique<Tokenizer>(Tokenizer::load(ckpt_dir + "/tokenizer.json"));
    p->vision_ = VisionTower::load(weights, p->cfg_.vision, device);
    p->llm_ = LLM::load(weights, p->cfg_.llm, device);
    if (!p->vision_ || !p->llm_) { fprintf(stderr, "[pipeline] model load failed\n"); return nullptr; }
    fprintf(stderr, "[pipeline] ready\n");
    return p;
}

InferenceResult Pipeline::parse_image(const std::string& image_path,
                                      const std::string& user_prompt,
                                      int max_new_tokens) {
    InferenceResult res;
    DOTS_CUDA_CHECK(cudaSetDevice(device_));

    // 1) image preprocessing
    PreprocParams pp;
    pp.patch_size = cfg_.preprocessor.patch_size;
    pp.merge_size = cfg_.preprocessor.merge_size;
    pp.min_pixels = cfg_.preprocessor.min_pixels;
    pp.max_pixels = cfg_.preprocessor.max_pixels;
    std::memcpy(pp.image_mean, cfg_.preprocessor.image_mean, sizeof(pp.image_mean));
    std::memcpy(pp.image_std,  cfg_.preprocessor.image_std,  sizeof(pp.image_std));

    ImageInput img;
    if (!preprocess_image_file(image_path, pp, img)) return res;
    int N_v = img.N_v();
    int N_img = img.num_image_tokens();
    res.num_image_tokens = N_img;
    fprintf(stderr, "[pipeline] image %dx%d patches -> N_v=%d N_img=%d\n",
            img.h, img.w, N_v, N_img);

    // 2) vision tower
    auto t0 = clk::now();
    DeviceTensor img_embeds = vision_->forward(img.pixel_values.data(), N_v,
                                               img.h, img.w, N_img);
    cudaDeviceSynchronize();
    res.prefill_ms += ms_since(t0);  // vision is part of prefill cost

    // 3) build prompt ids with the image placeholders.
    std::vector<int> prompt_ids = tokenizer_->build_image_prompt(user_prompt, N_img);
    return run_from_embeds(std::move(prompt_ids), img_embeds, N_img, max_new_tokens, res);
}

InferenceResult Pipeline::run_captured(const std::vector<int>& prompt_ids,
                                       const float* pixel_values, int N_v,
                                       int h_grid, int w_grid, int max_new_tokens) {
    InferenceResult res;
    DOTS_CUDA_CHECK(cudaSetDevice(device_));
    int N_img = N_v / (cfg_.vision.spatial_merge_size * cfg_.vision.spatial_merge_size);
    res.num_image_tokens = N_img;
    auto t0 = clk::now();
    DeviceTensor img_embeds = vision_->forward(pixel_values, N_v, h_grid, w_grid, N_img);
    cudaDeviceSynchronize();
    res.prefill_ms += ms_since(t0);
    return run_from_embeds(prompt_ids, img_embeds, N_img, max_new_tokens, res);
}

// Shared post-vision path: scatter image embeddings into the prompt's <|imgpad|>
// slots, prefill the LLM, then greedy-decode with the KV-cache.
InferenceResult Pipeline::run_from_embeds(const std::vector<int>& prompt_ids,
                                          const DeviceTensor& img_embeds,
                                          int N_img, int max_new_tokens,
                                          InferenceResult res) {
    res.prompt_len = (int)prompt_ids.size();
    int S = res.prompt_len;
    if (S > max_seq_) {
        fprintf(stderr, "[pipeline] prompt %d > max_seq %d\n", S, max_seq_);
        return res;
    }

    // img_mask: which prompt positions are <|imgpad|> (id == image_token_id).
    std::vector<int8_t> img_mask(S, 0);
    int imgpad_id = cfg_.llm.image_token_id;
    int img_count = 0;
    for (int i = 0; i < S; ++i)
        if (prompt_ids[i] == imgpad_id) { img_mask[i] = 1; ++img_count; }
    if (img_count != N_img) {
        fprintf(stderr, "[pipeline] WARN img placeholders %d != N_img %d (truncating)\n",
                img_count, N_img);
    }

    int* d_ids;     DOTS_CUDA_CHECK(cudaMalloc(&d_ids, sizeof(int) * S));
    int8_t* d_mask; DOTS_CUDA_CHECK(cudaMalloc(&d_mask, sizeof(int8_t) * S));
    DOTS_CUDA_CHECK(cudaMemcpy(d_ids, prompt_ids.data(), sizeof(int) * S, cudaMemcpyHostToDevice));
    DOTS_CUDA_CHECK(cudaMemcpy(d_mask, img_mask.data(), sizeof(int8_t) * S, cudaMemcpyHostToDevice));

    DeviceTensor inputs_embeds(Dtype::BF16, S, cfg_.llm.hidden_size);
    embed_and_scatter(d_ids, llm_->weights().embed, inputs_embeds.ptr(),
                      S, cfg_.llm.hidden_size, d_mask, img_embeds.ptr(), N_img);

    KVCache kv;
    kv.alloc(cfg_.llm.num_hidden_layers, max_seq_,
             cfg_.llm.num_key_value_heads, cfg_.llm.head_dim(), device_);

    DeviceTensor logits(Dtype::BF16, 1, cfg_.llm.vocab_size);
    auto t0 = clk::now();
    llm_->prefill(inputs_embeds.ptr(), S, kv, logits.ptr());
    cudaDeviceSynchronize();
    res.prefill_ms += ms_since(t0);

    auto is_eos = [&](int id) {
        for (int e : cfg_.generation.eos_token_ids) if (id == e) return true;
        return false;
    };
    int next_id = argmax_last(logits.ptr(), cfg_.llm.vocab_size);
    std::vector<int>& gen = res.token_ids;
    if (!(is_eos(next_id) || (int)gen.size() >= max_new_tokens)) gen.push_back(next_id);

    auto td = clk::now();
    while (!gen.empty()) {
        int cur_id = gen.back();
        if (is_eos(cur_id)) break;
        if ((int)gen.size() >= max_new_tokens) break;
        if (kv.current_len >= max_seq_ - 1) { fprintf(stderr, "[pipeline] hit max_seq\n"); break; }

        DeviceTensor one_embed(Dtype::BF16, 1, cfg_.llm.hidden_size);
        embed_lookup(&cur_id, llm_->weights().embed, one_embed.ptr(), 1, cfg_.llm.hidden_size);
        llm_->decode_step(one_embed.ptr(), kv, logits.ptr());
        next_id = argmax_last(logits.ptr(), cfg_.llm.vocab_size);
        if (is_eos(next_id)) break;
        gen.push_back(next_id);
    }
    cudaDeviceSynchronize();
    res.decode_ms = ms_since(td);
    res.tokens_generated = (int)gen.size();
    res.text = tokenizer_->decode(gen, /*skip_special=*/true);

    dfree(d_ids); dfree(d_mask);
    return res;
}

}  // namespace dots
