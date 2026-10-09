"""Field-realistic degradations for airborne thermal imagery, plus a dead control.

Every perturbation maps a severity in [0, 1] to a transform of a PIL RGB image, with
severity 0 the identity, so a sweep that starts at 0 gives the clean baseline for free.
All of them are photometric: they change pixels, not geometry, so the ground-truth boxes
stay fixed across the whole severity sweep and the audit never has to re-register labels.

The axes are the ones a drone's thermal sensor actually loses a target to:
  range_detail_loss  the target shrinks toward a few warm pixels (distance / altitude)
  atmospheric_blur   scattering and defocus through air
  low_contrast       thermal washout, little temperature difference to the background
  sensor_noise       readout / shot noise of an uncooled microbolometer
  motion_blur        smear from a moving platform

`background_marks` is the control: it drops stray warm/cold blobs only where there are no
targets and restores every ground-truth box region pixel-for-pixel, so a detector that is
reading the targets cannot change its answer. If detection quality moves under it, the model
is reacting to irrelevant background, and the main degradation results are that much noise.
"""

from __future__ import annotations

import hashlib
import io
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RGB = "RGB"
Box = tuple[float, float, float, float, float]  # (cls, cx, cy, w, h), normalised [0,1]


def _rng(image: Image.Image, severity: float, salt: str) -> np.random.Generator:
    h = hashlib.sha256()
    h.update(image.tobytes()[:4096])
    h.update(f"{severity:.4f}|{salt}".encode())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "big"))


def _boxes_to_rects(boxes: Sequence[Box], w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Normalised (cls, cx, cy, bw, bh) -> integer pixel (x0, y0, x1, y1), clamped."""
    rects = []
    for _, cx, cy, bw, bh in boxes:
        x0 = int(round((cx - bw / 2) * w)); x1 = int(round((cx + bw / 2) * w))
        y0 = int(round((cy - bh / 2) * h)); y1 = int(round((cy + bh / 2) * h))
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        rects.append((x0, y0, x1, y1))
    return rects


def clean(image: Image.Image, severity: float) -> Image.Image:
    return image.convert(RGB)


def atmospheric_blur(image: Image.Image, severity: float) -> Image.Image:
    if severity <= 0:
        return image.convert(RGB)
    return image.convert(RGB).filter(ImageFilter.GaussianBlur(0.4 + 3.6 * severity))


def low_contrast(image: Image.Image, severity: float) -> Image.Image:
    """Compress the dynamic range toward mid-grey: a target with little thermal contrast."""
    if severity <= 0:
        return image.convert(RGB)
    arr = np.asarray(image.convert(RGB), dtype=np.float32)
    keep = 1.0 - 0.85 * severity  # severity 1 keeps 15% of the contrast
    out = 128.0 + (arr - 128.0) * keep
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), RGB)


def sensor_noise(image: Image.Image, severity: float) -> Image.Image:
    if severity <= 0:
        return image.convert(RGB)
    arr = np.asarray(image.convert(RGB), dtype=np.float32)
    noise = _rng(image, severity, "sensor").normal(0, 55.0 * severity, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), RGB)


def range_detail_loss(image: Image.Image, severity: float) -> Image.Image:
    """Downscale then upscale to the original size: detail is gone as if the target were
    farther away, but the frame and the box coordinates are unchanged."""
    if severity <= 0:
        return image.convert(RGB)
    img = image.convert(RGB)
    w, h = img.size
    factor = 1.0 - 0.85 * severity  # severity 1 -> 15% linear resolution
    sw, sh = max(1, int(round(w * factor))), max(1, int(round(h * factor)))
    small = img.resize((sw, sh), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def motion_blur(image: Image.Image, severity: float) -> Image.Image:
    """Directional smear from platform motion, at a fixed per-image angle."""
    if severity <= 0:
        return image.convert(RGB)
    from scipy.ndimage import convolve

    length = int(round(1 + 18 * severity))
    angle = float(_rng(image, severity, "motion").uniform(0, np.pi))
    kernel = np.zeros((length, length), dtype=np.float32)
    cy = cx = (length - 1) / 2.0
    for t in np.linspace(-cx, cx, length * 2):
        x = int(round(cx + t * np.cos(angle))); y = int(round(cy + t * np.sin(angle)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[int(cy), int(cx)] = 1.0
    kernel /= kernel.sum()
    arr = np.asarray(image.convert(RGB), dtype=np.float32)
    out = np.stack([convolve(arr[..., c], kernel, mode="reflect") for c in range(3)], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), RGB)


def background_marks(image: Image.Image, severity: float, boxes: Sequence[Box] = ()) -> Image.Image:
    """Control: stray blobs only in the background, every target region restored exactly.

    A faithful detector's boxes must not move, because the pixels inside and around every
    ground-truth box are identical to the clean image. Only the empty background changes.
    """
    if severity <= 0:
        return image.convert(RGB)
    base = image.convert(RGB)
    w, h = base.size
    rects = _boxes_to_rects(boxes, w, h)
    pad = int(0.04 * max(w, h))  # keep marks clear of the box neighbourhood too

    def in_any_box(x: int, y: int) -> bool:
        return any(x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad
                   for (x0, y0, x1, y1) in rects)

    marked = base.copy()
    draw = ImageDraw.Draw(marked)
    rng = _rng(base, severity, "bgmarks")
    n = int(round(3 + 25 * severity))
    placed = 0
    for _ in range(n * 6):
        if placed >= n:
            break
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if in_any_box(x, y):
            continue
        r = int(rng.integers(4, 22))
        shade = int(rng.integers(0, 255))  # warm or cold blob
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade, shade))
        placed += 1
    # restore every target region pixel-for-pixel
    arr_base = np.asarray(base).copy()
    arr_marked = np.asarray(marked).copy()
    for (x0, y0, x1, y1) in rects:
        arr_marked[y0:y1, x0:x1] = arr_base[y0:y1, x0:x1]
    return Image.fromarray(arr_marked, RGB)


# name -> callable. background_marks is the control; it is the only one that reads boxes.
PERTURBATIONS: dict[str, Callable] = {
    "clean": clean,
    "range_detail_loss": range_detail_loss,
    "atmospheric_blur": atmospheric_blur,
    "low_contrast": low_contrast,
    "sensor_noise": sensor_noise,
    "motion_blur": motion_blur,
    "background_marks": background_marks,
}

CONTROL = "background_marks"
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def apply(name: str, image: Image.Image, severity: float, boxes: Sequence[Box] = ()) -> Image.Image:
    """Apply a named perturbation. Only the control uses `boxes`; the rest ignore it."""
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown perturbation {name!r}; have {sorted(PERTURBATIONS)}")
    if name == CONTROL:
        return background_marks(image, severity, boxes)
    return PERTURBATIONS[name](image, severity)
