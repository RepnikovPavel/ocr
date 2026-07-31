// Qwen2 BPE tokenizer, ported from the HF `tokenizer.json` (tokenizers lib).
//
// Why a from-scratch port instead of pulling in a library: the regression
// contract is *exact greedy token-id match* against HF, and the only way to
// be sure we reproduce the same token sequence is to walk the same pipeline
// — NFC normalise, GPT-4 regex split, byte-level encode, BPE merge — that
// tokenizers does. The added/special tokens (the <|img|>, <|imgpad|>, …
// markers the chat template emits) are matched as whole strings before BPE
// so they never get split.
//
// Decoder is byte-level too, so detokenisation reconstructs UTF-8 for the
// final markdown output.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace dots {

struct TokenizerConfig {
    // Special / added tokens, parsed from tokenizer.json "added_tokens".
    std::unordered_map<std::string, int> added_tokens;   // content -> id
    std::unordered_map<int, std::string> id_to_added;
    std::vector<int> special_ids;                        // sorted, for fast lookup
    int eos_id       = 151643;
    int pad_id       = 151680;
    int image_pad_id = 151665;   // <|imgpad|>
    int img_id       = 151666;   // <|img|>
    int endofimg_id  = 151667;   // <|endofimg|>
    int user_id      = 151670;   // <|user|>
    int endofuser_id = 151671;   // <|endofuser|>
    int assistant_id = 151672;   // <|assistant|>
};

class Tokenizer {
public:
    // Load vocab/merges/added_tokens from a HF tokenizer.json.
    static Tokenizer load(const std::string& tokenizer_json_path);

    const TokenizerConfig& cfg() const { return cfg_; }
    int vocab_size() const { return vocab_size_; }

    // Encode a single (non-special) string to BPE ids, applying the GPT-4
    // pre-tokenizer + byte-level + BPE merge. Does NOT insert special tokens.
    std::vector<int> encode(const std::string& text) const;

    // Encode with special-token matching: occurrences of any added token
    // string in `text` are emitted as their id verbatim and never BPE'd.
    std::vector<int> encode_with_special(const std::string& text) const;

    // Decode a span of ids back to a UTF-8 string (byte-level inverse).
    std::string decode(const int* ids, std::size_t n, bool skip_special) const;
    std::string decode(const std::vector<int>& ids, bool skip_special) const {
        return decode(ids.data(), ids.size(), skip_special);
    }

    // Build the chat-template prompt for a single user turn with one image.
    // Mirrors chat_template.json: <|user|><|img|><|imgpad|><|endofimg|>{prompt}<|endofuser|><|assistant|>
    // The <|imgpad|> placeholder appears `num_image_tokens` times so the
    // vision embeddings have exactly that many slots to scatter into.
    std::vector<int> build_image_prompt(const std::string& user_text,
                                        int num_image_tokens) const;

private:
    TokenizerConfig cfg_;
    // BPE state.
    std::unordered_map<std::string, int> vocab_;        // includes byte-level bytes
    std::unordered_map<int, std::string> id_to_token_;
    int vocab_size_ = 0;
    // rank of a pair (a,b) = position in merges list; lower rank merges first.
    std::unordered_map<uint64_t, int> merge_rank_;
    int unk_id_ = -1;

    // Apply BPE to a single pre-token word (already byte-level encoded).
    std::vector<int> bpe_word(const std::string& word) const;
    // Encode one pre-token chunk (already regex-matched) to ids.
    std::vector<int> encode_pretoken(const std::string& chunk) const;
    static uint64_t pair_key(int a, int b) {
        return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
    }

    // byte-level
    static std::string bytes_to_unicode(const std::string& in);
    static std::string unicode_to_bytes(const std::string& in);
};

}  // namespace dots
