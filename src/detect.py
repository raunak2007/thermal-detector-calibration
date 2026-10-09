"""Detections and ground-truth matching.

A detection is (cls, score, cx, cy, w, h) with the box normalised to [0, 1], the same frame
as the ground truth. `match_detections` does the standard greedy, per-class, highest-confidence
-first assignment at an IoU threshold, and labels every detection a true or false positive.
Those labels are what the calibration metrics read: a detection's score is its stated
confidence, and whether it is a true positive is the ground truth that score should have predicted.
"""

from __future__ import annotations

from typing import Callable, Sequence

from PIL import Image

Det = tuple[int, float, float, float, float, float]   # (cls, score, cx, cy, w, h)
Box = tuple[float, float, float, float, float]        # (cls, cx, cy, w, h)
# A detector responder turns an image into detections. The real one wraps YOLO; the smoke
# one is a stub. Ground truth is never passed in.
Responder = Callable[[Image.Image], list[Det]]


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(dets: Sequence[Det], gts: Sequence[Box], iou_thr: float = 0.5) -> list[dict]:
    """Greedy per-class matching. Returns one record per detection with its TP/FP label."""
    gt_boxes = [(int(c), _xywh_to_xyxy(cx, cy, w, h)) for (c, cx, cy, w, h) in gts]
    matched = [False] * len(gt_boxes)
    out: list[dict] = []
    for (cls, score, cx, cy, w, h) in sorted(dets, key=lambda d: d[1], reverse=True):
        dbox = _xywh_to_xyxy(cx, cy, w, h)
        best_j, best_iou = -1, iou_thr
        for j, (gc, gbox) in enumerate(gt_boxes):
            if matched[j] or gc != int(cls):
                continue
            v = iou(dbox, gbox)
            if v >= best_iou:
                best_iou, best_j = v, j
        is_tp = best_j >= 0
        if is_tp:
            matched[best_j] = True
        out.append({"cls": int(cls), "score": float(score), "is_tp": int(is_tp)})
    return out
