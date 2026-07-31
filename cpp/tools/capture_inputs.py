#!/usr/bin/env python3
# Capture ONLY the HF processor inputs (pixel_values.bin, input_ids.txt,
# grid_thw.txt) — no model load. Used when the HF generate path is unavailable
# (version drift) but we still want byte-identical pixels for the C++ engine.
#   docker exec dots_mocr_demo python3 /opt/dots-mocr/cpp/tools/capture_inputs.py \
#       /opt/dots-mocr/cpp/tiny.png /state/ref
import os, sys
import numpy as np

def main():
    img_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained("/models", local_files_only=True, trust_remote_code=True)
    img = Image.open(img_path).convert("RGB")
    prompt = (
        "Please output the layout information from the PDF image, including "
        "each layout element's bbox, its category, and the corresponding text "
        "content within the bbox.\n\n"
        "1. Bbox format: [x1, y1, x2, y2]\n\n"
        "2. Layout Categories: The possible categories are ['Caption', "
        "'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', "
        "'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
        "3. Text Extraction & Formatting Rules:\n"
        "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
        "    - Formula: Format its text as LaTeX.\n"
        "    - Table: Format its text as HTML.\n"
        "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
        "4. Constraints:\n"
        "    - The output text must be the original text from the image, with no translation.\n"
        "    - All layout elements must be sorted according to human reading order.\n\n"
        "5. Final Output: The entire output must be a single JSON object.\n"
    )
    text = "<|img|><|imgpad|><|endofimg|>" + prompt
    inp = p(text=[text], images=[img], return_tensors="pt")
    ids = inp["input_ids"][0].cpu().tolist()
    pv  = inp["pixel_values"].cpu().numpy().astype(np.float32)
    gt  = inp["image_grid_thw"][0].cpu().tolist()
    with open(os.path.join(out_dir, "input_ids.txt"), "w") as f:
        f.write(" ".join(str(i) for i in ids))
    pv.tofile(os.path.join(out_dir, "pixel_values.bin"))
    with open(os.path.join(out_dir, "grid_thw.txt"), "w") as f:
        f.write("%d %d %d" % (int(gt[0]), int(gt[1]), int(gt[2])))
    print("captured: ids=%d pv=%s grid=%s" % (len(ids), pv.shape, gt), flush=True)

if __name__ == "__main__":
    main()
