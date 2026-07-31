// dots_mocr — CLI for the pure C++ + CUDA inference engine.
//
//   dots_mocr --ckpt <dir> --image <file> [--prompt "text"] [--max-new-tokens N]
//
// Loads the safetensors checkpoint, runs the full vision+LLM pipeline on the
// GPU, and prints the generated markdown text to stdout (timing to stderr).
#include "pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage() {
    fprintf(stderr,
        "usage: dots_mocr --ckpt <ckpt_dir> --image <image_file>\n"
        "                 [--prompt <text>] [--max-new-tokens N] [--device N]\n"
        "                 [--max-seq N]\n"
        "  --prompt default: the prompt_layout_all_en layout-extraction prompt\n");
}

int main(int argc, char** argv) {
    std::string ckpt, image, prompt;
    int max_new_tokens = 16384;
    int device = 0;
    int max_seq = 16384;
    bool have_prompt = false;
    std::string dump_ids_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
            return argv[++i];
        };
        if      (a == "--ckpt")    ckpt = next("--ckpt");
        else if (a == "--image")   image = next("--image");
        else if (a == "--prompt")  { prompt = next("--prompt"); have_prompt = true; }
        else if (a == "--max-new-tokens") max_new_tokens = atoi(next("--max-new-tokens"));
        else if (a == "--device")  device = atoi(next("--device"));
        else if (a == "--max-seq") max_seq = atoi(next("--max-seq"));
        else if (a == "--dump-ids") dump_ids_path = next("--dump-ids");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); usage(); return 2; }
    }
    if (ckpt.empty() || image.empty()) { usage(); return 2; }

    if (!have_prompt) {
        prompt =
            "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.\n\n"
            "1. Bbox format: [x1, y1, x2, y2]\n"
            "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n"
            "3. Text Extraction & Formatting Rules:\n"
            "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
            "    - Formula: Format its text as LaTeX.\n"
            "    - Table: Format its text as HTML.\n"
            "    - All Others (Text, Title, etc.): Format their text as Markdown.\n"
            "4. Constraints:\n"
            "    - The output text must be the original text from the image, with no translation.\n"
            "    - All layout elements must be sorted according to human reading order.\n"
            "5. Final Output: The entire output must be a single JSON object.\n";
    }

    auto pipe = dots::Pipeline::create(ckpt, device, max_seq);
    if (!pipe) { fprintf(stderr, "failed to load model\n"); return 1; }

    auto r = pipe->parse_image(image, prompt, max_new_tokens);
    fprintf(stderr, "[cli] prompt=%d img_tokens=%d generated=%d "
            "prefill=%.1fms decode=%.1fms (%.1f tok/s)\n",
            r.prompt_len, r.num_image_tokens, r.tokens_generated,
            r.prefill_ms, r.decode_ms,
            r.decode_ms > 0 ? r.tokens_generated / (r.decode_ms / 1000.0) : 0.0);
    // The generated text to stdout.
    fputs(r.text.c_str(), stdout);
    fputc('\n', stdout);

    // Optionally dump the generated token ids for regression comparison.
    if (!dump_ids_path.empty()) {
        FILE* f = fopen(dump_ids_path.c_str(), "w");
        if (f) {
            for (size_t i = 0; i < r.token_ids.size(); ++i)
                fprintf(f, "%d\n", r.token_ids[i]);
            fclose(f);
            fprintf(stderr, "[cli] dumped %zu ids to %s\n", r.token_ids.size(), dump_ids_path.c_str());
        }
    }
    return 0;
}
