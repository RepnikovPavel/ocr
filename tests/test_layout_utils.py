import json

from PIL import Image

from dots_mocr.utils.image_utils import smart_resize
from dots_mocr.utils.layout_utils import (
    draw_layout_on_image,
    is_legal_bbox,
    parse_scene_text_output,
    post_process_cells,
    post_process_output,
    post_process_scene_text,
    pre_process_bboxes,
)

CATEGORIES = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title",
]


def make_cells():
    return [
        {"bbox": [56, 56, 700, 112], "category": "Title", "text": "Synthetic Title"},
        {"bbox": [56, 200, 700, 400], "category": "Text", "text": "Body **text**"},
        {"bbox": [56, 500, 400, 800], "category": "Picture"},
    ]


def test_pre_and_post_process_are_inverse(synthetic_page_image):
    origin = synthetic_page_image
    input_h, input_w = smart_resize(origin.height // 2, origin.width // 2)
    original_bboxes = [[100, 120, 400, 360], [0, 0, 840, 1120]]
    model_space = pre_process_bboxes(origin, original_bboxes, input_width=input_w, input_height=input_h)
    cells = [{"bbox": bbox, "category": "Text", "text": "x"} for bbox in model_space]
    restored = post_process_cells(origin, cells, input_width=input_w, input_height=input_h)
    for restored_cell, bbox in zip(restored, original_bboxes):
        for got, expected in zip(restored_cell["bbox"], bbox):
            assert abs(got - expected) <= 4, (restored_cell["bbox"], bbox)


def test_post_process_cells_identity_when_same_size(synthetic_page_image):
    cells = make_cells()
    out = post_process_cells(
        synthetic_page_image, cells,
        input_width=synthetic_page_image.width,
        input_height=synthetic_page_image.height,
    )
    assert [c["bbox"] for c in out] == [c["bbox"] for c in cells]
    # input cells must not be mutated
    assert cells[0]["bbox"] == [56, 56, 700, 112]


def test_is_legal_bbox():
    assert is_legal_bbox(make_cells())
    assert not is_legal_bbox([{"bbox": [10, 10, 10, 40]}])
    assert not is_legal_bbox([{"bbox": [10, 50, 40, 10]}])


def test_post_process_output_valid_json(synthetic_page_image):
    response = json.dumps(make_cells())
    cells, filtered = post_process_output(
        response, "prompt_layout_all_en", synthetic_page_image, synthetic_page_image,
    )
    assert not filtered
    assert len(cells) == 3
    assert cells[0]["category"] == "Title"
    assert all(cell["category"] in CATEGORIES for cell in cells)


def test_post_process_output_malformed_json_falls_back(synthetic_page_image):
    response = '[{"bbox": [1, 2, 3, 4], "category": "Text", "text": "abc"'  # truncated
    cleaned, filtered = post_process_output(
        response, "prompt_layout_all_en", synthetic_page_image, synthetic_page_image,
    )
    assert filtered
    assert isinstance(cleaned, str)


def test_draw_layout_on_image_keeps_size(synthetic_page_image):
    out = draw_layout_on_image(synthetic_page_image, make_cells())
    assert isinstance(out, Image.Image)
    assert out.size == synthetic_page_image.size


def test_parse_scene_text_output():
    response = "(10, 20), (110, 20), (110, 60), (10, 60) HELLO\n(5, 6), (50, 6), (50, 30), (5, 30) WORLD"
    instances = parse_scene_text_output(response)
    assert len(instances) == 2
    assert instances[0]["text"] == "HELLO"
    assert instances[0]["points"] == [10, 20, 110, 20, 110, 60, 10, 60]
    assert instances[1]["text"] == "WORLD"


def test_post_process_scene_text_scales_back():
    origin = Image.new("RGB", (2000, 2800), "white")
    input_h, input_w = smart_resize(2800, 2000)
    response = f"(0, 0), ({input_w}, 0), ({input_w}, {input_h}), (0, {input_h}) FULL"
    instances, failed = post_process_scene_text(response, origin, None)
    assert not failed
    points = instances[0]["points"]
    assert abs(points[2] - 2000) <= 3
    assert abs(points[5] - 2800) <= 3


def test_post_process_scene_text_failure_passthrough():
    origin = Image.new("RGB", (100, 100), "white")
    out, failed = post_process_scene_text("no coordinates here", origin, None)
    assert failed
    assert out == "no coordinates here"
