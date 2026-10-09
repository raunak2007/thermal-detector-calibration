"""Perturbations are box-preserving, and the control is a true null."""

import numpy as np
from PIL import Image

from src.perturbations import CONTROL, PERTURBATIONS, apply


def _img(size=96):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8), "RGB")


def test_severity_zero_is_identity():
    img = _img()
    base = np.asarray(img.convert("RGB"))
    for name in PERTURBATIONS:
        out = np.asarray(apply(name, img, 0.0, boxes=[(0, 0.5, 0.5, 0.2, 0.2)]))
        assert np.array_equal(out, base), f"{name} severity 0 should be identity"


def test_photometric_keep_size():
    img = _img(128)
    for name in PERTURBATIONS:
        out = apply(name, img, 1.0, boxes=[(0, 0.5, 0.5, 0.2, 0.2)])
        assert out.size == img.size, f"{name} changed image size"


def test_control_restores_boxes_and_touches_background():
    img = _img(200)
    boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
    clean = np.asarray(img.convert("RGB"))
    out = np.asarray(apply(CONTROL, img, 1.0, boxes=boxes))
    # box region is pixel-identical
    x0, x1 = int(0.4 * 200), int(0.6 * 200)
    assert np.array_equal(out[x0:x1, x0:x1], clean[x0:x1, x0:x1]), "control must restore the box region"
    # and something in the background changed
    assert not np.array_equal(out, clean), "control must mark the background"
