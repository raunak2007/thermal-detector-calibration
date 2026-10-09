"""No-GPU end-to-end check: synthetic thermal frames + a classical blob detector.

Proves the perturbation sweep, the matching, the D-ECE / mAP / risk-coverage metrics, and the
control all work before anything touches Modal. The detector is a threshold-and-label blob
finder, not a learned model, so the numbers are only a pipeline check, not a result.

    python -m src.run_local_smoke
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .dataset import synthetic_thermal
from .detect import Det
from .eval import run_sweep
from .metrics import (aggregate, baseline_calibration, control_check,
                      integrity_summary, risk_coverage, to_frame)
from .perturbations import PERTURBATIONS


def blob_detector_responder(image: Image.Image) -> list[Det]:
    """Threshold warm pixels, label blobs, emit a box and a brightness-based score per blob."""
    from scipy import ndimage

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    H, W = gray.shape
    thresh = 120.0
    mask = gray > thresh
    labels, n = ndimage.label(mask)
    dets: list[Det] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        if len(xs) < 6:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        cls = 0 if area < 0.004 * H * W else 1
        cx = ((x0 + x1) / 2) / W; cy = ((y0 + y1) / 2) / H
        w = (x1 - x0 + 1) / W; h = (y1 - y0 + 1) / H
        bright = float(gray[ys, xs].mean())
        score = max(0.02, min(0.99, (bright - thresh) / (255.0 - thresh)))
        dets.append((cls, score, cx, cy, w, h))
    return dets


def main() -> None:
    samples = synthetic_thermal(n=12, seed=1)
    perts = list(PERTURBATIONS.keys())
    sev = [0.0, 0.5, 1.0]
    records, gt_per_class = run_sweep(samples, blob_detector_responder, perts, sev, progress=False)
    df = to_frame(records)

    print("gt per class:", gt_per_class, " detections:", len(df))
    base = baseline_calibration(df, gt_per_class)
    print("\n=== baseline (clean) ===")
    print(f"mAP {base['map']}  precision {base['precision']}  "
          f"mean confidence {base['mean_confidence']}  D-ECE {base['d_ece']}")

    agg = aggregate(df, gt_per_class)
    print("\n=== integrity summary (largest mAP-vs-confidence gap first) ===")
    print(integrity_summary(agg).to_string(index=False))
    print("\n=== control check ===")
    print(control_check(agg))
    print("\n=== risk-coverage (clean) ===")
    print(risk_coverage(df).to_string(index=False))


if __name__ == "__main__":
    main()
