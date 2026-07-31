// dots.mocr model config — mirrors config.json + generation_config.json.
//
// All the constants the forward pass needs live here so the kernels are
// parameterised by values, not magic numbers. Numbers are the checkpoint's
// own defaults; a user can override by pointing at a different config.json.
#pragma once

#include <string>
#include <vector>

namespace dots {

struct VisionConfig {
    int   embed_dim          = 1536;
    int   hidden_size        = 1536;   // after merger (= LLM hidden_size)
    int   intermediate_size  = 4224;
    int   num_hidden_layers  = 42;
    int   num_attention_heads= 12;
    int   num_channels       = 3;
    int   patch_size         = 14;
    int   spatial_merge_size = 2;
    int   temporal_patch_size= 1;
    float rms_norm_eps       = 1e-5f;
    bool  use_bias           = false;  // vision qkv/proj have NO bias
    bool  post_norm          = true;
    bool  is_causal          = false;  // full (non-causal) attention
    float rope_theta         = 10000.0f;

    int head_dim() const { return embed_dim / num_attention_heads; }   // 128
    int image_factor() const { return spatial_merge_size * patch_size; } // 28
};

struct LLMConfig {
    int   vocab_size        = 151936;
    int   hidden_size       = 1536;
    int   intermediate_size = 8960;
    int   num_hidden_layers = 28;
    int   num_attention_heads       = 12;
    int   num_key_value_heads       = 2;
    int   max_position_embeddings   = 131072;
    float rms_norm_eps      = 1e-6f;
    float rope_theta        = 1e6f;
    bool  attention_bias    = true;    // q/k/v HAVE bias; o_proj does NOT
    bool  tie_word_embeddings = false;
    int   image_token_id    = 151665;  // <|imgpad|>
    int   video_token_id    = 151656;  // <|video_pad|>

    int head_dim() const { return hidden_size / num_attention_heads; }   // 128
    int num_kv_heads() const { return num_key_value_heads; }
    int q_heads_per_kv() const { return num_attention_heads / num_key_value_heads; } // g = 6
};

struct PreprocessorConfig {
    int   patch_size         = 14;
    int   temporal_patch_size= 1;
    int   merge_size         = 2;
    int   min_pixels         = 3136;
    int   max_pixels         = 11289600;
    float image_mean[3]      = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3]       = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct GenerationConfig {
    int  max_length         = 32768;
    std::vector<int> eos_token_ids = {151643, 151672, 151673};
};

struct ModelConfig {
    VisionConfig       vision;
    LLMConfig          llm;
    PreprocessorConfig preprocessor;
    GenerationConfig   generation;

    // Load from a checkpoint directory (reads config.json +
    // generation_config.json + preprocessor_config.json).
    static ModelConfig load(const std::string& ckpt_dir);
};

}  // namespace dots
