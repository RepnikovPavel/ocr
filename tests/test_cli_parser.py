import argparse
import json
import os

import pytest

from dots_mocr.cli import DotsMOCRParser, parse_pages


def make_parser(monkeypatch, tmp_path, fake_response, **kwargs):
    """DotsMOCRParser with the model replaced by a canned response."""
    monkeypatch.setattr(DotsMOCRParser, "_load_model", lambda self, ckpt: None)
    parser = DotsMOCRParser(
        ckpt="/nonexistent",
        device="cpu",
        dtype="float32",
        output_dir=str(tmp_path),
        **kwargs,
    )
    calls = []

    def fake_inference(image, prompt, temperature=None, stats=None):
        calls.append({"prompt": prompt, "temperature": temperature, "size": image.size})
        if stats is not None:  # mimic a real 3-token generation for the telemetry
            stats.start(prompt_tokens=7)
            for _ in range(3):
                stats.record_token()
            stats.finish(generated_tokens=3)
        return fake_response

    parser._inference = fake_inference
    parser._calls = calls
    return parser


LAYOUT_RESPONSE = json.dumps([
    {"bbox": [56, 56, 700, 112], "category": "Title", "text": "Synthetic Title"},
    {"bbox": [56, 200, 700, 400], "category": "Text", "text": "Body text"},
])


def test_parse_pages_ranges():
    assert parse_pages("14,17,28-30") == [13, 16, 27, 28, 29]
    assert parse_pages("1") == [0]
    with pytest.raises(argparse.ArgumentTypeError):
        parse_pages("0")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_pages("5-3")


def test_parse_image_layout_writes_artifacts(monkeypatch, tmp_path, synthetic_page_image):
    image_path = tmp_path / "page.png"
    synthetic_page_image.save(image_path)
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE)

    results = parser.parse_file(str(image_path), prompt_mode="prompt_layout_all_en")

    assert len(results) == 1
    result = results[0]
    save_dir = tmp_path / "page"
    layout = json.loads((save_dir / "page.json").read_text(encoding="utf-8"))
    assert layout[0]["category"] == "Title"
    md = (save_dir / "page.md").read_text(encoding="utf-8")
    assert "Synthetic Title" in md and "Body text" in md
    assert (save_dir / "page.jpg").exists()
    assert (save_dir / "page_nohf.md").exists()
    assert result["md_content_path"].endswith("page.md")
    # jsonl summary next to save dir
    jsonl = (tmp_path / "page.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl) == 1


def test_parse_image_malformed_response_marks_filtered(monkeypatch, tmp_path, synthetic_page_image):
    image_path = tmp_path / "page.png"
    synthetic_page_image.save(image_path)
    parser = make_parser(monkeypatch, tmp_path, '[{"bbox": [1,2,3,4], "category": "Text", "text": "abc"')

    results = parser.parse_file(str(image_path), prompt_mode="prompt_layout_all_en")

    assert results[0].get("filtered") is True
    assert (tmp_path / "page" / "page.json").exists()


def test_parse_pdf_multipage_order_and_artifacts(monkeypatch, tmp_path, synthetic_pdf):
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE, num_thread=2)

    results = parser.parse_file(synthetic_pdf, prompt_mode="prompt_layout_all_en")

    assert [r["page_no"] for r in results] == [0, 1, 2]
    save_dir = tmp_path / "synthetic"
    for page in range(3):
        assert (save_dir / f"synthetic_page_{page}.json").exists()
        assert (save_dir / f"synthetic_page_{page}.md").exists()
    assert len(parser._calls) == 3


def test_parse_pdf_page_subset(monkeypatch, tmp_path, synthetic_pdf):
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE)

    results = parser.parse_file(synthetic_pdf, prompt_mode="prompt_layout_all_en", pages=[1])

    assert [r["page_no"] for r in results] == [1]
    assert (tmp_path / "synthetic" / "synthetic_page_1.json").exists()
    assert not (tmp_path / "synthetic" / "synthetic_page_0.json").exists()


def test_parse_file_rejects_unknown_extension(monkeypatch, tmp_path):
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE)
    bogus = tmp_path / "file.docx"
    bogus.write_text("x")
    with pytest.raises(ValueError):
        parser.parse_file(str(bogus))


def test_temperature_override_does_not_mutate_parser(monkeypatch, tmp_path, synthetic_page_image):
    image_path = tmp_path / "page.png"
    synthetic_page_image.save(image_path)
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE)
    before = parser.temperature

    parser.parse_image(
        str(image_path), "page", "prompt_layout_all_en", str(tmp_path), temperature=0.7,
    )

    assert parser.temperature == before
    assert parser._calls[0]["temperature"] == 0.7


def test_get_prompt_grounding_appends_scaled_bbox(monkeypatch, tmp_path, synthetic_page_image):
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE)
    prompt = parser.get_prompt(
        "prompt_grounding_ocr",
        bbox=[10, 20, 200, 300],
        origin_image=synthetic_page_image,
        image=synthetic_page_image,
    )
    assert prompt.rstrip().endswith("]")
    assert "[" in prompt


def test_get_prompt_svg_substitutes_dimensions(monkeypatch, tmp_path, synthetic_page_image):
    parser = make_parser(monkeypatch, tmp_path, LAYOUT_RESPONSE)
    prompt = parser.get_prompt(
        "prompt_image_to_svg", origin_image=synthetic_page_image, image=synthetic_page_image,
    )
    assert str(synthetic_page_image.width) in prompt
    assert str(synthetic_page_image.height) in prompt


def test_get_prompt_general_custom(monkeypatch, tmp_path):
    parser = make_parser(monkeypatch, tmp_path, "answer")
    assert parser.get_prompt("prompt_general", custom_prompt="What is this?") == "What is this?"
    assert "describe" in parser.get_prompt("prompt_general")
