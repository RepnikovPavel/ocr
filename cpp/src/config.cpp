#include "config.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>

namespace dots {

using json = nlohmann::json;

static json load_json(const std::string& path, bool& ok) {
    ok = false;
    std::ifstream f(path);
    if (!f) return {};
    json j;
    try { f >> j; ok = true; } catch (...) {}
    return j;
}

ModelConfig ModelConfig::load(const std::string& ckpt_dir) {
    ModelConfig c;

    bool ok = false;
    json cfg = load_json(ckpt_dir + "/config.json", ok);
    if (!ok) { fprintf(stderr, "[config] cannot read %s/config.json\n", ckpt_dir.c_str()); return c; }

    auto geti = [&](const json& j, const char* k, int def) {
        return j.contains(k) && j[k].is_number_integer() ? j[k].get<int>() : def;
    };
    auto getf = [&](const json& j, const char* k, float def) {
        return j.contains(k) && j[k].is_number() ? j[k].get<float>() : def;
    };
    auto getb = [&](const json& j, const char* k, bool def) {
        return j.contains(k) && j[k].is_boolean() ? j[k].get<bool>() : def;
    };

    // LLM block.
    c.llm.vocab_size        = geti(cfg, "vocab_size", c.llm.vocab_size);
    c.llm.hidden_size       = geti(cfg, "hidden_size", c.llm.hidden_size);
    c.llm.intermediate_size = geti(cfg, "intermediate_size", c.llm.intermediate_size);
    c.llm.num_hidden_layers = geti(cfg, "num_hidden_layers", c.llm.num_hidden_layers);
    c.llm.num_attention_heads     = geti(cfg, "num_attention_heads", c.llm.num_attention_heads);
    c.llm.num_key_value_heads     = geti(cfg, "num_key_value_heads", c.llm.num_key_value_heads);
    c.llm.max_position_embeddings = geti(cfg, "max_position_embeddings", c.llm.max_position_embeddings);
    c.llm.rms_norm_eps      = getf(cfg, "rms_norm_eps", c.llm.rms_norm_eps);
    c.llm.rope_theta        = getf(cfg, "rope_theta", c.llm.rope_theta);
    c.llm.attention_bias    = getb(cfg, "attention_bias", c.llm.attention_bias);
    c.llm.tie_word_embeddings = getb(cfg, "tie_word_embeddings", c.llm.tie_word_embeddings);
    c.llm.image_token_id    = geti(cfg, "image_token_id", c.llm.image_token_id);
    c.llm.video_token_id    = geti(cfg, "video_token_id", c.llm.video_token_id);

    // Vision block (nested "vision_config").
    if (cfg.contains("vision_config") && cfg["vision_config"].is_object()) {
        const auto& v = cfg["vision_config"];
        VisionConfig& vc = c.vision;
        vc.embed_dim          = geti(v, "embed_dim", vc.embed_dim);
        vc.hidden_size        = geti(v, "hidden_size", vc.hidden_size);
        vc.intermediate_size  = geti(v, "intermediate_size", vc.intermediate_size);
        vc.num_hidden_layers  = geti(v, "num_hidden_layers", vc.num_hidden_layers);
        vc.num_attention_heads= geti(v, "num_attention_heads", vc.num_attention_heads);
        vc.num_channels       = geti(v, "num_channels", vc.num_channels);
        vc.patch_size         = geti(v, "patch_size", vc.patch_size);
        vc.spatial_merge_size = geti(v, "spatial_merge_size", vc.spatial_merge_size);
        vc.temporal_patch_size= geti(v, "temporal_patch_size", vc.temporal_patch_size);
        vc.rms_norm_eps       = getf(v, "rms_norm_eps", vc.rms_norm_eps);
        vc.use_bias           = getb(v, "use_bias", vc.use_bias);
        vc.post_norm          = getb(v, "post_norm", vc.post_norm);
        vc.is_causal          = getb(v, "is_causal", vc.is_causal);
    }

    // preprocessor_config.json.
    json pp = load_json(ckpt_dir + "/preprocessor_config.json", ok);
    if (ok) {
        c.preprocessor.min_pixels  = geti(pp, "min_pixels", c.preprocessor.min_pixels);
        c.preprocessor.max_pixels  = geti(pp, "max_pixels", c.preprocessor.max_pixels);
        c.preprocessor.patch_size  = geti(pp, "patch_size", c.preprocessor.patch_size);
        c.preprocessor.merge_size  = geti(pp, "merge_size", c.preprocessor.merge_size);
        if (pp.contains("image_mean") && pp["image_mean"].is_array() && pp["image_mean"].size() >= 3)
            for (int i = 0; i < 3; ++i) c.preprocessor.image_mean[i] = pp["image_mean"][i].get<float>();
        if (pp.contains("image_std") && pp["image_std"].is_array() && pp["image_std"].size() >= 3)
            for (int i = 0; i < 3; ++i) c.preprocessor.image_std[i] = pp["image_std"][i].get<float>();
    }

    // generation_config.json.
    json gc = load_json(ckpt_dir + "/generation_config.json", ok);
    if (ok) {
        c.generation.max_length = geti(gc, "max_length", c.generation.max_length);
        if (gc.contains("eos_token_id")) {
            c.generation.eos_token_ids.clear();
            if (gc["eos_token_id"].is_array()) {
                for (const auto& e : gc["eos_token_id"])
                    c.generation.eos_token_ids.push_back(e.get<int>());
            } else if (gc["eos_token_id"].is_number_integer()) {
                c.generation.eos_token_ids.push_back(gc["eos_token_id"].get<int>());
            }
        }
    }

    fprintf(stderr, "[config] llm: H=%d L=%d heads=%d kv=%d V=%d theta=%.0g eps=%.0g\n",
            c.llm.hidden_size, c.llm.num_hidden_layers, c.llm.num_attention_heads,
            c.llm.num_key_value_heads, c.llm.vocab_size, c.llm.rope_theta, c.llm.rms_norm_eps);
    fprintf(stderr, "[config] vision: H=%d L=%d heads=%d I=%d patch=%d merge=%d\n",
            c.vision.embed_dim, c.vision.num_hidden_layers, c.vision.num_attention_heads,
            c.vision.intermediate_size, c.vision.patch_size, c.vision.spatial_merge_size);
    return c;
}

}  // namespace dots
