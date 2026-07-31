// Tokenizer unit test: encode/decode round-trip and known-id checks.
#include "tokenizer.h"
#include <cassert>
#include <cstdio>
#include <string>

using namespace dots;

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "tokenizer.json";
    Tokenizer tk = Tokenizer::load(path);
    if (tk.vocab_size() == 0) { fprintf(stderr, "skip: no tokenizer at %s\n", path.c_str()); return 77; }

    // Known special ids (from the checkpoint's added_tokens).
    assert(tk.cfg().image_pad_id == 151665);
    assert(tk.cfg().img_id       == 151666);
    assert(tk.cfg().user_id      == 151670);
    assert(tk.cfg().assistant_id == 151672);
    assert(tk.cfg().eos_id       == 151643);

    // "hello world" should encode to a handful of BPE pieces (Ġ = space prefix).
    auto ids = tk.encode("hello world");
    assert(!ids.empty());
    auto dec = tk.decode(ids, false);
    assert(dec == "hello world");

    // Special-token matching: <|imgpad|> must encode to its single id.
    auto sid = tk.encode_with_special("<|img|><|imgpad|><|endofimg|>hi");
    assert(sid.size() >= 4);
    assert(sid[0] == 151666);  // <|img|>
    assert(sid[1] == 151665);  // <|imgpad|>
    // the last id before "hi" tokens is <|endofimg|>
    assert(sid[2] == 151667);

    // build_image_prompt shape sanity: user + img + N*imgpad + endofimg + prompt + endofuser + assistant
    auto p = tk.build_image_prompt("hi", 5);
    assert(p.size() == 1 + 1 + 5 + 1 + (int)tk.encode("hi").size() + 1 + 1);

    printf("OK tokenizer: vocab=%d, 'hello world'->%zu ids\n", tk.vocab_size(), ids.size());
    return 0;
}
