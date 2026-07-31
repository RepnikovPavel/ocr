// End-to-end inference pipeline: image + prompt -> markdown text.
//
// Orchestrates: image preprocess -> vision tower -> embed+scatter -> LLM
// prefill -> greedy decode loop (KV-cache) until EOS or max_new_tokens.
#pragma once

#include "config.h"
#include "llm.h"
#include "tokenizer.h"
#include "vision_tower.h"

#include <memory>
#include <string>
#include <vector>

namespace dots {

struct InferenceResult {
    std::vector<int> token_ids;     // generated ids (excluding prompt)
    std::string text;               // detokenised
    int num_image_tokens = 0;
    int prompt_len = 0;
    double prefill_ms = 0;
    double decode_ms = 0;
    int tokens_generated = 0;
};

class Pipeline {
public:
    static std::unique_ptr<Pipeline> create(const std::string& ckpt_dir,
                                            int device = 0,
                                            int max_seq = 16384);

    // Parse one image with the given user prompt; greedy decode.
    InferenceResult parse_image(const std::string& image_path,
                                const std::string& user_prompt,
                                int max_new_tokens = 16384);

    // Run on pre-captured inputs (prompt token ids + HF pixel_values) so the
    // engine's model math can be compared against a reference byte-for-byte,
    // independent of any preprocessing drift. Used by test_regression.
    InferenceResult run_captured(const std::vector<int>& prompt_ids,
                                 const float* pixel_values, int N_v,
                                 int h_grid, int w_grid, int max_new_tokens);

    const ModelConfig& config() const { return cfg_; }

private:
    // Shared post-vision path (scatter -> prefill -> greedy decode).
    InferenceResult run_from_embeds(const std::vector<int>& prompt_ids,
                                    const DeviceTensor& img_embeds,
                                    int N_img, int max_new_tokens,
                                    InferenceResult res);

private:
    ModelConfig cfg_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<VisionTower> vision_;
    std::unique_ptr<LLM> llm_;
    int device_ = 0;
    int max_seq_ = 16384;
};

}  // namespace dots
