#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace dots {

using json = nlohmann::json;

// ---- byte-level alphabet (the GPT-2/Qwen byte->printable mapping) -----------
//
// HF ByteLevel maps each of the 256 byte values to a printable unicode char so
// that spaces and control bytes survive as visible tokens. This is exactly the
// classic GPT-2 `bytes_to_unicode` table.
static const char* kByteLevelMap[256];
static bool        kByteLevelMapInit = false;

static void init_byte_level_map() {
    if (kByteLevelMapInit) return;
    // Build as code points first.
    static int codepoints[256];
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
            codepoints[b] = b;
        else { codepoints[b] = 256 + n; ++n; }
    }
    for (int b = 0; b < 256; ++b) {
        // Encode the code point as UTF-8.
        static thread_local char buf[8];
        int cp = codepoints[b];
        if (cp < 0x80) { buf[0] = char(cp); buf[1] = 0; }
        else if (cp < 0x800) {
            buf[0] = char(0xC0 | (cp >> 6));
            buf[1] = char(0x80 | (cp & 0x3F));
            buf[2] = 0;
        } else {
            buf[0] = char(0xE0 | (cp >> 12));
            buf[1] = char(0x80 | ((cp >> 6) & 0x3F));
            buf[2] = char(0x80 | (cp & 0x3F));
            buf[3] = 0;
        }
        kByteLevelMap[b] = strdup(buf);
    }
    kByteLevelMapInit = true;
}

std::string Tokenizer::bytes_to_unicode(const std::string& in) {
    init_byte_level_map();
    std::string out;
    for (unsigned char c : in) out += kByteLevelMap[c];
    return out;
}

// Inverse: decode a byte-level string back to raw bytes.
std::string Tokenizer::unicode_to_bytes(const std::string& in) {
    init_byte_level_map();
    std::string out;
    out.reserve(in.size());
    std::size_t i = 0;
    while (i < in.size()) {
        unsigned char c = (unsigned char)in[i];
        // Decode UTF-8 code point (1..3 bytes here, since the table is BMP).
        int cp = 0; int len = 0;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = (c & 0x1F); len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = (c & 0x0F); len = 3; }
        else { cp = (c & 0x07); len = 4; }
        for (int k = 1; k < len && i + k < in.size(); ++k)
            cp = (cp << 6) | (((unsigned char)in[i + k]) & 0x3F);
        // Find the byte whose map equals this code point.
        bool found = false;
        for (int b = 0; b < 256; ++b) {
            int mapcp = 0; const char* s = kByteLevelMap[b];
            unsigned char c0 = (unsigned char)s[0];
            if (c0 < 0x80) { mapcp = c0; }
            else if ((c0 & 0xE0) == 0xC0) { mapcp = (c0 & 0x1F); if (s[1]) mapcp = (mapcp << 6) | ((unsigned char)s[1] & 0x3F); }
            else if ((c0 & 0xF0) == 0xE0) {
                mapcp = (c0 & 0x0F);
                if (s[1]) mapcp = (mapcp << 6) | ((unsigned char)s[1] & 0x3F);
                if (s[2]) mapcp = (mapcp << 6) | ((unsigned char)s[2] & 0x3F);
            }
            if (mapcp == cp) { out.push_back((char)b); found = true; break; }
        }
        if (!found) out.push_back('?');
        i += len;
    }
    return out;
}

// ---- GPT-4 pre-tokenizer regex ----------------------------------------------
// The pattern (HuggingFace tokenizers form):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S)
//   | \s+
//
// We walk it as a hand-written state machine over UTF-8 bytes, classifying
// each code point into {letter, digit, space (\t\v\f and space), \r, \n, other}.
// The contractions clause is case-insensitive over ASCII.
//
// This is the same approach Rust's regex crate takes for these alternations,
// and reproduces HF's split exactly — which is what matters for parity.

namespace {

enum class CC { Letter, Digit, Space, CR, LF, Other };

// ASCII fast-path classification; non-ASCII counts as a letter (the model's
// vocab was trained on byte-level, so multibyte letters form words too).
static CC classify(unsigned char c) {
    if (c == '\n') return CC::LF;
    if (c == '\r') return CC::CR;
    if (c == ' ' || c == '\t' || c == '\v' || c == '\f') return CC::Space;
    if ((c >= '0' && c <= '9')) return CC::Digit;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return CC::Letter;
    if (c < 0x80) return CC::Other;
    return CC::Letter;  // continuation/lead byte of a multibyte rune = letter-ish
}

// Advance `i` over one UTF-8 code point, returning its byte length.
static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Case-insensitive ASCII match of `suffix` at position i, consuming the
// apostrophe + letters if it matches one of the contractions.
static bool match_contraction(const std::string& s, std::size_t i, std::size_t& end) {
    if (i + 2 > s.size()) return false;
    if (s[i] != '\'') return false;
    // The contraction body is one of t/s/re/ve/m/ll/d (case-insensitive).
    // Lengths: 1 ('s,'t,'m,'d), 2 ('re,'ve,'ll).
    char a = tolower((unsigned char)s[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        end = i + 2; return true;
    }
    if (i + 3 <= s.size()) {
        char b = tolower((unsigned char)s[i + 2]);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
            end = i + 3; return true;
        }
    }
    return false;
}

// Run the GPT-4 regex against `s`, emitting the pre-token chunks in order.
// `add_prefix_space` matches ByteLevel's first-chunk leading-space handling.
static std::vector<std::string> gpt4_pretokenize(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0, n = s.size();
    std::string cur;
    auto flush = [&]() { if (!cur.empty()) { out.push_back(cur); cur.clear(); } };

    while (i < n) {
        // 1) contraction
        std::size_t e;
        if (match_contraction(s, i, e)) {
            flush(); out.push_back(s.substr(i, e - i)); i = e; continue;
        }
        CC c = classify((unsigned char)s[i]);
        // 2) [^\r\n\p{L}\p{N}]?\p{L}+   — optional one non-letter/non-digit/newline, then letters
        {
            std::size_t j = i;
            // optional leading single char that is not \r \n letter digit
            if (j < n) {
                CC cj = classify((unsigned char)s[j]);
                if (cj != CC::Letter && cj != CC::Digit && cj != CC::CR && cj != CC::LF) {
                    j += utf8_len((unsigned char)s[j]);
                }
            }
            // require at least one letter at the new j
            if (j < n && classify((unsigned char)s[j]) == CC::Letter) {
                std::size_t start = j;
                while (j < n && classify((unsigned char)s[j]) == CC::Letter) j += utf8_len((unsigned char)s[j]);
                // But only consume the optional prefix if there is a letter run
                // after it; the match spans from i.
                flush();
                out.push_back(s.substr(i, j - i));
                i = j;
                continue;
            }
        }
        // 3) \p{N}
        if (c == CC::Digit) {
            std::size_t j = i;
            while (j < n && classify((unsigned char)s[j]) == CC::Digit) ++j;
            flush(); out.push_back(s.substr(i, j - i)); i = j; continue;
        }
        // 4) ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            std::size_t j = i;
            if (j < n) {
                CC cj = classify((unsigned char)s[j]);
                if (cj != CC::Space && cj != CC::Letter && cj != CC::Digit) {
                    // optional single leading space already accounted by [^ ]? —
                    // here the optional leading char is ' ?' i.e. a single space.
                }
            }
            // The clause is " ?[^\s\L\N]+[\r\n]*". Re-read: optional ONE space,
            // then one-or-more "other" (non-space, non-letter, non-digit), then
            // any \r\n run.
            std::size_t k = i;
            if (k < n && classify((unsigned char)s[k]) == CC::Space) k += 1; // optional single space
            std::size_t other_start = k;
            while (k < n) {
                CC ck = classify((unsigned char)s[k]);
                if (ck == CC::Space || ck == CC::Letter || ck == CC::Digit) break;
                ++k;
            }
            if (k > other_start) {  // at least one "other" char
                while (k < n && (classify((unsigned char)s[k]) == CC::CR || classify((unsigned char)s[k]) == CC::LF)) ++k;
                flush(); out.push_back(s.substr(i, k - i)); i = k; continue;
            }
            // else fall through
        }
        // 5) \s*[\r\n]+
        if (c == CC::Space || c == CC::CR || c == CC::LF) {
            std::size_t j = i;
            while (j < n && classify((unsigned char)s[j]) == CC::Space) ++j;     // \s*
            if (j < n && (classify((unsigned char)s[j]) == CC::CR || classify((unsigned char)s[j]) == CC::LF)) {
                while (j < n && (classify((unsigned char)s[j]) == CC::CR || classify((unsigned char)s[j]) == CC::LF)) ++j;
                flush(); out.push_back(s.substr(i, j - i)); i = j; continue;
            }
            // 6) \s+(?!\S)  — trailing whitespace (not followed by non-space)
            // 7) \s+        — any whitespace run
            // The regex engine tries \s*[\r\n]+ first (above); if no newline,
            // it falls to \s+(?!\S) then \s+. We combine: take the maximal
            // space run; if it ends the string or is followed by more space
            // at end, that's (?!\S)-satisfying.
            j = i;
            while (j < n && classify((unsigned char)s[j]) == CC::Space) ++j;
            // \s+(?!\S): a space run that is NOT immediately followed by a
            // non-space. Equivalently the run includes trailing spaces.
            if (j >= n) {
                flush(); out.push_back(s.substr(i, j - i)); i = j; continue;
            }
            // If after the space run there is a non-space char, the (?!\S)
            // alternative fails, but \s+ still matches the leading single
            // space (tokenizers' \s+ is greedy but the (?!\S) branch is tried
            // first; on failure it backtracks to one space). Reproduce: emit
            // exactly one space, leave the rest.
            flush(); out.push_back(s.substr(i, 1)); i += 1; continue;
        }
        // Fallback: a single "other" char.
        { int ul = utf8_len((unsigned char)s[i]); flush(); out.push_back(s.substr(i, ul)); i += ul; }
    }
    flush();
    return out;
}

}  // namespace

// ---- BPE ---------------------------------------------------------------------

std::vector<int> Tokenizer::bpe_word(const std::string& word) const {
    // word is the byte-level string of a single pre-token. Split into symbols
    // = one symbol per character (UTF-8 code point), then greedily merge the
    // lowest-rank adjacent pair until no merge applies. Classic BPE.
    if (word.empty()) return {};
    // Single char -> direct vocab lookup.
    // Build symbol list as token ids.
    std::vector<int> syms;
    std::vector<std::string> parts;
    std::size_t i = 0;
    while (i < word.size()) {
        int ul = utf8_len((unsigned char)word[i]);
        std::string ch = word.substr(i, ul);
        parts.push_back(ch);
        auto it = vocab_.find(ch);
        syms.push_back(it != vocab_.end() ? it->second : unk_id_);
        i += ul;
    }
    if (syms.size() < 2) return syms;

    // Greedy merge by minimum rank over all adjacent pairs.
    while (true) {
        int best_rank = INT32_MAX;
        std::size_t best_idx = SIZE_MAX;
        for (std::size_t k = 0; k + 1 < syms.size(); ++k) {
            auto it = merge_rank_.find(pair_key(syms[k], syms[k + 1]));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second; best_idx = k;
            }
        }
        if (best_idx == SIZE_MAX) break;
        // Merge syms[best_idx] + syms[best_idx+1] into the merged token.
        std::string merged = parts[best_idx] + parts[best_idx + 1];
        int merged_id = unk_id_;
        auto it = vocab_.find(merged);
        if (it != vocab_.end()) merged_id = it->second;
        syms[best_idx] = merged_id;
        parts[best_idx] = merged;
        syms.erase(syms.begin() + best_idx + 1);
        parts.erase(parts.begin() + best_idx + 1);
    }
    return syms;
}

std::vector<int> Tokenizer::encode_pretoken(const std::string& chunk) const {
    // Byte-level encode the chunk, then BPE.
    std::string bl = bytes_to_unicode(chunk);
    return bpe_word(bl);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    auto chunks = gpt4_pretokenize(text);
    for (const auto& c : chunks) {
        auto sub = encode_pretoken(c);
        ids.insert(ids.end(), sub.begin(), sub.end());
    }
    return ids;
}

std::vector<int> Tokenizer::encode_with_special(const std::string& text) const {
    // Scan for the longest added-token match at each position. HF tokenizers
    // matches added tokens left-to-right, longest-first within a position.
    std::vector<int> ids;
    std::size_t i = 0;
    // Precompute max added-token length for the scan window.
    std::size_t max_len = 0;
    for (const auto& kv : cfg_.added_tokens) max_len = std::max(max_len, kv.first.size());

    while (i < text.size()) {
        // Try to match an added token starting here (longest first).
        std::size_t win = std::min(max_len, text.size() - i);
        bool matched = false;
        for (std::size_t L = win; L >= 1; --L) {
            std::string cand = text.substr(i, L);
            auto it = cfg_.added_tokens.find(cand);
            if (it != cfg_.added_tokens.end()) {
                ids.push_back(it->second);
                i += L;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // Otherwise consume one pre-token: run the GPT-4 regex from i to get
        // the next chunk, BPE it, advance.
        // Simplest correct approach: pretokenize the remainder lazily.
        // Find the length of the next chunk by re-running the tokenizer from i.
        // To avoid re-scanning, pretokenize the whole tail once per non-match.
        std::string tail = text.substr(i);
        auto chunks = gpt4_pretokenize(tail);
        if (chunks.empty()) break;
        // Only consume the first chunk; the next iteration re-checks specials.
        const std::string& c0 = chunks[0];
        auto sub = encode_pretoken(c0);
        ids.insert(ids.end(), sub.begin(), sub.end());
        i += c0.size();
    }
    return ids;
}

std::string Tokenizer::decode(const int* ids, std::size_t n, bool skip_special) const {
    std::string bl;
    for (std::size_t k = 0; k < n; ++k) {
        int id = ids[k];
        // added/special tokens
        auto ai = cfg_.id_to_added.find(id);
        if (ai != cfg_.id_to_added.end()) {
            if (skip_special) {
                // skip special tokens; keep non-special added tokens as text
                bool is_special = std::binary_search(cfg_.special_ids.begin(),
                                                    cfg_.special_ids.end(), id);
                if (is_special) continue;
            }
            bl += ai->second;
            continue;
        }
        auto ti = id_to_token_.find(id);
        if (ti == id_to_token_.end()) continue;
        bl += ti->second;
    }
    return unicode_to_bytes(bl);
}

std::vector<int> Tokenizer::build_image_prompt(const std::string& user_text,
                                               int num_image_tokens) const {
    // <|user|><|img|><|imgpad>x N<|endofimg|>{prompt}<|endofuser|><|assistant|>
    std::vector<int> ids;
    ids.push_back(cfg_.user_id);
    ids.push_back(cfg_.img_id);
    for (int t = 0; t < num_image_tokens; ++t) ids.push_back(cfg_.image_pad_id);
    ids.push_back(cfg_.endofimg_id);
    auto tids = encode(user_text);
    ids.insert(ids.end(), tids.begin(), tids.end());
    ids.push_back(cfg_.endofuser_id);
    ids.push_back(cfg_.assistant_id);
    return ids;
}

// ---- loader ------------------------------------------------------------------

Tokenizer Tokenizer::load(const std::string& path) {
    Tokenizer tk;
    init_byte_level_map();
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "[tokenizer] cannot open %s\n", path.c_str()); return tk; }
    json j; try { f >> j; } catch (...) { fprintf(stderr, "[tokenizer] bad json\n"); return tk; }

    const auto& model = j["model"];
    // vocab
    for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
        int id = it.value().get<int>();
        tk.vocab_[it.key()] = id;
        tk.id_to_token_[id] = it.key();
        if (id >= tk.vocab_size_) tk.vocab_size_ = id + 1;
    }
    // merges -> ranks
    int rank = 0;
    for (const auto& m : model["merges"]) {
        std::string s = m.get<std::string>();
        auto sp = s.find(' ');
        if (sp == std::string::npos) { ++rank; continue; }
        std::string a = s.substr(0, sp), b = s.substr(sp + 1);
        auto ia = tk.vocab_.find(a), ib = tk.vocab_.find(b);
        if (ia != tk.vocab_.end() && ib != tk.vocab_.end())
            tk.merge_rank_[pair_key(ia->second, ib->second)] = rank;
        ++rank;
    }
    tk.unk_id_ = model.value("unk_id", -1);
    if (tk.unk_id_ < 0 && model.contains("unk_token")) {
        auto u = model["unk_token"];
        if (u.is_string()) { auto it = tk.vocab_.find(u.get<std::string>()); if (it != tk.vocab_.end()) tk.unk_id_ = it->second; }
    }
    // added_tokens
    if (j.contains("added_tokens")) {
        for (const auto& at : j["added_tokens"]) {
            std::string content = at["content"];
            int id = at["id"].get<int>();
            bool special = at.value("special", false);
            tk.cfg_.added_tokens[content] = id;
            tk.cfg_.id_to_added[id] = content;
            if (special) tk.cfg_.special_ids.push_back(id);
            // Also register in vocab so decode/BPE can see them (HF treats
            // added tokens as atomic and never BPE-splits them; encode_with_special
            // guarantees this by matching them first).
            if (!tk.vocab_.count(content)) tk.vocab_[content] = id;
            if (id >= tk.vocab_size_) tk.vocab_size_ = id + 1;
        }
        std::sort(tk.cfg_.special_ids.begin(), tk.cfg_.special_ids.end());
    }
    // Set the well-known ids from their content strings, so this works even if
    // a future checkpoint renames a token.
    auto resolve = [&](const char* name, int def) {
        auto it = tk.cfg_.added_tokens.find(name);
        return it != tk.cfg_.added_tokens.end() ? it->second : def;
    };
    tk.cfg_.image_pad_id = resolve("<|imgpad|>", tk.cfg_.image_pad_id);
    tk.cfg_.img_id       = resolve("<|img|>",     tk.cfg_.img_id);
    tk.cfg_.endofimg_id  = resolve("<|endofimg|>",tk.cfg_.endofimg_id);
    tk.cfg_.user_id      = resolve("<|user|>",    tk.cfg_.user_id);
    tk.cfg_.endofuser_id = resolve("<|endofuser|>",tk.cfg_.endofuser_id);
    tk.cfg_.assistant_id = resolve("<|assistant|>",tk.cfg_.assistant_id);
    tk.cfg_.eos_id       = resolve("<|endoftext|>",tk.cfg_.eos_id);
    tk.cfg_.pad_id       = resolve("[PAD]", tk.cfg_.pad_id);

    fprintf(stderr, "[tokenizer] vocab=%d merges=%d added=%zu unk=%d\n",
            tk.vocab_size_, (int)model["merges"].size(),
            tk.cfg_.added_tokens.size(), tk.unk_id_);
    return tk;
}

}  // namespace dots
