"""Samples for the audit: the real HIT-UAV test split, and a synthetic thermal stand-in.

A Sample is one image plus its ground-truth boxes in YOLO form (class, cx, cy, w, h), all
normalised to [0, 1]. Two sources produce Samples with the same shape:

- `load_yolo_split(images_dir, labels_dir)` reads a YOLO-format directory (image files with
  a matching `.txt` of boxes). This is what the audit runs on, over HIT-UAV's test split after
  it has been materialised to YOLO layout (see modal_app.py), which is also the layout the
  Ultralytics fine-tune consumes, so one copy of the data serves both.

- `synthetic_thermal(...)` draws cool-background frames with a few warm target blobs and known
  boxes. No download, no GPU, so the perturbations, the matching, and the metrics are all
  exercised before anything touches Modal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Box = tuple[float, float, float, float, float]  # (cls, cx, cy, w, h) normalised
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Sample:
    id: str
    image: Image.Image
    boxes: list[Box] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic thermal source (no download, no GPU) — for the smoke run and tests.
# ---------------------------------------------------------------------------
def synthetic_thermal(n: int = 12, seed: int = 0, size: int = 640) -> list[Sample]:
    """Cool-background frames with warm person/vehicle blobs and known boxes."""
    rng = np.random.default_rng(seed)
    out: list[Sample] = []
    for i in range(n):
        base = rng.normal(40, 8, (size, size)).clip(0, 90)  # cool background
        boxes: list[Box] = []
        n_targets = int(rng.integers(2, 8))
        for _ in range(n_targets):
            cls = int(rng.integers(0, 2))  # 0 person (small), 1 vehicle (larger)
            bw = (0.03 if cls == 0 else 0.08) * float(rng.uniform(0.8, 1.4))
            bh = bw * float(rng.uniform(1.2, 2.2) if cls == 0 else rng.uniform(0.5, 0.9))
            cx = float(rng.uniform(0.1, 0.9)); cy = float(rng.uniform(0.1, 0.9))
            x0 = int((cx - bw / 2) * size); x1 = int((cx + bw / 2) * size)
            y0 = int((cy - bh / 2) * size); y1 = int((cy + bh / 2) * size)
            yy, xx = np.mgrid[0:size, 0:size]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            sx, sy = max(2, (x1 - x0) / 2), max(2, (y1 - y0) / 2)
            blob = np.exp(-(((xx - mx) / sx) ** 2 + ((yy - my) / sy) ** 2))
            base = base + 170 * blob  # warm target
            boxes.append((cls, cx, cy, bw, bh))
        img = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGB")
        out.append(Sample(id=f"synthetic_{i:04d}", image=img, boxes=boxes))
    return out


# ---------------------------------------------------------------------------
# Real source: a YOLO-format split on disk (HIT-UAV test, after materialisation).
# ---------------------------------------------------------------------------
def _read_yolo_label(path: Path) -> list[Box]:
    boxes: list[Box] = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, cx, cy, w, h = parts[:5]
            boxes.append((float(c), float(cx), float(cy), float(w), float(h)))
    return boxes


def load_yolo_split(images_dir: str, labels_dir: str, max_images: int | None = None) -> list[Sample]:
    """Load images and their YOLO `.txt` boxes from a directory pair."""
    idir, ldir = Path(images_dir), Path(labels_dir)
    files = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if max_images is not None:
        files = files[:max_images]
    out: list[Sample] = []
    for p in files:
        img = Image.open(p).convert("RGB")
        boxes = _read_yolo_label(ldir / (p.stem + ".txt"))
        out.append(Sample(id=p.stem, image=img, boxes=boxes))
    return out


def iter_source(name: str, **kwargs) -> list[Sample]:
    if name == "synthetic":
        return synthetic_thermal(**kwargs)
    if name == "yolo":
        return load_yolo_split(**kwargs)
    raise KeyError(f"unknown source {name!r}")
