import base64
import io

import pytest
from PIL import Image

from dots_mocr.utils.consts import IMAGE_FACTOR, MAX_PIXELS, MIN_PIXELS
from dots_mocr.utils.image_utils import (
    PILimage_to_base64,
    ceil_by_factor,
    fetch_image,
    floor_by_factor,
    get_input_dimensions,
    round_by_factor,
    smart_resize,
    to_rgb,
)


def test_factor_helpers():
    assert round_by_factor(29, 28) == 28
    assert round_by_factor(43, 28) == 56
    assert ceil_by_factor(29, 28) == 56
    assert floor_by_factor(55, 28) == 28


@pytest.mark.parametrize(
    "height,width",
    [(1120, 840), (1000, 750), (3508, 2480), (28, 28), (5000, 5000), (997, 1231)],
)
def test_smart_resize_invariants(height, width):
    h, w = smart_resize(height, width)
    assert h % IMAGE_FACTOR == 0 and w % IMAGE_FACTOR == 0
    assert MIN_PIXELS <= h * w <= MAX_PIXELS
    # aspect ratio is approximately preserved
    if min(height, width) > 200:
        assert abs((h / w) - (height / width)) / (height / width) < 0.1


def test_smart_resize_upscales_tiny_images():
    h, w = smart_resize(10, 10)
    assert h * w >= MIN_PIXELS


def test_smart_resize_downscales_huge_images():
    h, w = smart_resize(20000, 20000, max_pixels=MAX_PIXELS)
    assert h * w <= MAX_PIXELS


def test_smart_resize_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError):
        smart_resize(28, 28 * 250)


def test_fetch_image_from_pil_keeps_size_without_limits(synthetic_page_image):
    out = fetch_image(synthetic_page_image)
    assert out.size == synthetic_page_image.size
    assert out.mode == "RGB"


def test_fetch_image_applies_max_pixels(synthetic_page_image):
    out = fetch_image(synthetic_page_image, max_pixels=200_000)
    assert out.width * out.height <= 200_000
    assert out.width % IMAGE_FACTOR == 0 and out.height % IMAGE_FACTOR == 0


def test_fetch_image_from_path(tmp_path, synthetic_page_image):
    path = tmp_path / "page.png"
    synthetic_page_image.save(path)
    out = fetch_image(str(path))
    assert out.size == synthetic_page_image.size


def test_fetch_image_from_base64(synthetic_page_image):
    uri = PILimage_to_base64(synthetic_page_image)
    out = fetch_image(uri)
    assert out.size == synthetic_page_image.size


def test_pil_image_to_base64_roundtrip(synthetic_page_image):
    uri = PILimage_to_base64(synthetic_page_image)
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split("base64,", 1)[1])
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.size == synthetic_page_image.size


def test_to_rgb_flattens_alpha_on_white():
    rgba = Image.new("RGBA", (32, 32), (255, 0, 0, 0))
    out = to_rgb(rgba)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_get_input_dimensions(synthetic_page_image):
    w, h = get_input_dimensions(synthetic_page_image, MIN_PIXELS, MAX_PIXELS)
    assert w % IMAGE_FACTOR == 0 and h % IMAGE_FACTOR == 0
