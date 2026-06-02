import numpy as np

from raw_transform import crop_bayer, crop_rect, stabilization_translation, translate_bayer


def test_translate_bayer_integer_shift():
    bayer = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out = translate_bayer(bayer, 1, 0)
    assert out[0, 1] == bayer[0, 0]
    assert out[0, 0] == 0


def test_crop_bayer_even_align():
    bayer = np.ones((10, 10), dtype=np.uint16)
    cropped = crop_bayer(bayer, 3, 5, 4, 4)
    assert cropped.shape == (4, 4)


def test_stabilization_translation_integer():
    tx, ty = stabilization_translation(
        10.0, 5.0, (80.0, 60.0), (160, 120), integer=True,
    )
    assert isinstance(tx, int) and isinstance(ty, int)


def test_crop_rect_matches_extract_crop_origin():
    x, y, w, h = crop_rect(320, 240, 100, 100, True, (640, 480))
    assert w == 100 and h == 100
    assert x == 320 - 50 and y == 240 - 50
