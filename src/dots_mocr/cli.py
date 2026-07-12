import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import argparse
from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

from dots_mocr.transformers_patch import register_transformers

from dots_mocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_mocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_mocr.utils.doc_utils import fitz_doc_to_image, load_pdf_pages
from dots_mocr.utils.prompts import dict_promptmode_to_prompt
from dots_mocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes, parse_scene_text_output, post_process_scene_text, draw_scene_text_on_image, format_scene_text_to_markdown
from dots_mocr.utils.svg_utils import extract_svg_from_response, svg_to_png, create_comparison_image
from dots_mocr.utils.format_transformer import layoutjson2md


class DotsMOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self, 
            ckpt='./weights/DotsMOCR',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=32768,
            num_thread=1,
            dpi=200, 
            output_dir="./output", 
            min_pixels=None,
            max_pixels=None,
            attn_implementation="sdpa",
            device="auto",
            dtype="auto",
        ):
        self.dpi = dpi
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.attn_implementation = attn_implementation
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        if self.attn_implementation == "flash_attention_2" and self.device == "cpu":
            raise ValueError("flash_attention_2 requires CUDA")
        if self.attn_implementation == "flash_attention_2" and self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash_attention_2 requires float16 or bfloat16")

        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

        self._load_model(ckpt)
        print(f"Model loaded from {ckpt}, device={self.device}, dtype={self.dtype}, num_thread={self.num_thread}")

    def _resolve_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return device

    def _resolve_dtype(self, dtype):
        if dtype == "auto":
            return torch.bfloat16 if self.device == "cuda" else torch.float32
        return getattr(torch, dtype)

    def _load_model(self, ckpt):
        register_transformers()
        config = AutoConfig.from_pretrained(
            ckpt,
            local_files_only=True,
            trust_remote_code=False,
        )
        config.vision_config.attn_implementation = self.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            config=config,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="balanced",  # balanced layer split across GPUs for model parallel (half layers approx on each)
            max_memory={0: "14GiB", 1: "14GiB"},
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            ckpt,
            local_files_only=True,
            trust_remote_code=False,
        )

    def _inference(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("mm_token_type_ids", None)

        inputs = inputs.to(self.device)

        generation_kwargs = {"max_new_tokens": self.max_completion_tokens}
        if self.temperature > 0:
            generation_kwargs.update(
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None, custom_prompt=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        if prompt_mode == 'prompt_image_to_svg':
            prompt = prompt.replace("{width}", str(origin_image.width))
            prompt = prompt.replace("{height}", str(origin_image.height))
            print(prompt)
        if prompt_mode == 'prompt_general':
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = "Please describe the content of this image."
        return prompt

    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        custom_prompt=None,
        temperature=None,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels, custom_prompt=custom_prompt)
        
        saved_temperature = self.temperature
        if temperature is not None:
            self.temperature = temperature

        response = self._inference(image, prompt)

        self.temperature = saved_temperature

        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr', 'prompt_web_parsing']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        elif prompt_mode in ['prompt_scene_spotting']:
            instances, failed = post_process_scene_text(response, origin_image, image, min_pixels, max_pixels)
            
            vis_image = origin_image if failed else draw_scene_text_on_image(origin_image, instances) if instances else origin_image
            
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            vis_image.save(image_layout_path)
            
            json_file_path = os.path.join(save_dir, f"{save_name}.json")
            with open(json_file_path, 'w', encoding="utf-8") as f:
                json.dump(instances if not failed else {"raw": response}, f, ensure_ascii=False, indent=2)
            
            md_content = format_scene_text_to_markdown(instances) if not failed else response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            result.update({
                'layout_image_path': image_layout_path,
                'layout_info_path': json_file_path,
                'md_content_path': md_file_path,
                'text_instances': instances if not failed else None,
                'filtered': failed
            })

        elif prompt_mode in ['prompt_image_to_svg']:
            svg_content, has_svg = extract_svg_from_response(response)
            
            if has_svg:
                svg_path = os.path.join(save_dir, f"{save_name}.svg")
                with open(svg_path, "w", encoding="utf-8") as svg_file:
                    svg_file.write(svg_content)
                png_path = os.path.join(save_dir, f"{save_name}_rendered.png")
                w, h = origin_image.size
                success, error = svg_to_png(svg_content, png_path, width=w, height=h)
                                
                if success:
                    rendered_image = Image.open(png_path)
                    comparison_image = create_comparison_image(origin_image, rendered_image)
                    image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                    comparison_image.save(image_layout_path)
                else:
                    print(f"SVG to PNG failed: {error}")
                    image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                    origin_image.save(image_layout_path)        
            else:
                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
            
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            md_content = f"# Generated SVG Code\n\n```xml\n{response}\n```"
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            result.update({
                'layout_image_path': image_layout_path,
                'md_content_path': md_file_path,
            })
            if has_svg:
                result['svg_content_path'] = svg_path
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False, custom_prompt=None, temperature=None):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess, custom_prompt=custom_prompt, temperature=temperature)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir, pages=None):
        print(f"loading pdf: {input_path}")
        pdf_pages = load_pdf_pages(input_path, dpi=self.dpi, page_ids=pages)
        total_pages = len(pdf_pages)
        if total_pages == 0:
            raise ValueError("PDF contains no renderable selected pages")
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": page_idx,
            } for page_idx, image in pdf_pages
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_file(self, 
        input_path, 
        output_dir="", 
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
        custom_prompt=None,
        pages=None,
        ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir, pages=pages)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess, custom_prompt=custom_prompt)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")
        
        print(f"Parsing finished, results saving to {save_dir}")
        with open(os.path.join(output_dir, os.path.basename(filename)+'.jsonl'), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results


def parse_pages(value):
    pages = set()
    for item in value.split(","):
        bounds = item.strip().split("-", 1)
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 1 or end < start:
            raise argparse.ArgumentTypeError(f"invalid page range: {item}")
        pages.update(range(start - 1, end))
    return sorted(pages)



def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.mocr Multimodal OCR: Parse Anything from Documents",
    )
    
    parser.add_argument(
        "--input_path", type=str,
        help="Input PDF/image file path"
    )
    parser.add_argument(
        "--pages", type=parse_pages, default=None,
        help="1-based PDF pages, for example 14,17,28-35"
    )
    
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ckpt", type=str, default="./weights/DotsMOCR",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--attn_implementation", type=str, default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation (default: sdpa)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device (default: auto)"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Model dtype (default: auto)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=1,
        help=""
    )
    parser.add_argument(
        "--no_fitz_preprocess", action='store_true',
        help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--custom_prompt", type=str, default=None,
        help="Custom prompt for free QA mode"
    )
    args = parser.parse_args()

    dots_mocr_parser = DotsMOCRParser(
        ckpt=args.ckpt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        device=args.device,
        dtype=args.dtype,
    )

    fitz_preprocess = not args.no_fitz_preprocess
    if fitz_preprocess:
        print(f"Using fitz preprocess for image input, check the change of the image pixels")
    result = dots_mocr_parser.parse_file(
        args.input_path, 
        prompt_mode=args.prompt,
        bbox=args.bbox,
        fitz_preprocess=fitz_preprocess,
        pages=args.pages,
        )


if __name__ == "__main__":
    main()
