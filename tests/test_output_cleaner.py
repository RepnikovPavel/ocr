import json

from dots_mocr.utils.output_cleaner import OutputCleaner


def test_clean_truncated_json_recovers_complete_items():
    items = [
        {"bbox": [1, 2, 3, 4], "category": "Text", "text": "first"},
        {"bbox": [5, 6, 7, 8], "category": "Title", "text": "second"},
    ]
    raw = json.dumps(items)
    truncated = raw[:-20]  # cut inside the last dict
    cleaned = OutputCleaner().clean_model_output(truncated)
    assert isinstance(cleaned, list)
    assert len(cleaned) >= 1
    assert cleaned[0]["category"] == "Text"
    assert cleaned[0]["bbox"] == [1, 2, 3, 4]


def test_clean_list_with_three_coordinate_bbox_keeps_text():
    data = [
        {"bbox": [1, 2, 3], "category": "Text", "text": "broken bbox"},
        {"bbox": [1, 2, 3, 4], "category": "Title", "text": "ok"},
    ]
    cleaned = OutputCleaner().clean_model_output(data)
    assert {"category": "Text", "text": "broken bbox"} in cleaned
    assert any(item.get("bbox") == [1, 2, 3, 4] for item in cleaned)


def test_clean_deduplicates_repeated_bboxes():
    data = [
        {"bbox": [1, 2, 3, 4], "category": "Text", "text": "a"},
        {"bbox": [1, 2, 3, 4], "category": "Text", "text": "a"},
        {"bbox": [9, 9, 20, 20], "category": "Title", "text": "b"},
    ]
    cleaned = OutputCleaner().clean_model_output(data)
    bboxes = [tuple(item["bbox"]) for item in cleaned if "bbox" in item]
    assert bboxes.count((1, 2, 3, 4)) == 1
