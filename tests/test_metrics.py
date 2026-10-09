"""IoU, matching, AP, and D-ECE behave."""

import numpy as np

from src.detect import iou, match_detections
from src.metrics import _ap, d_ece


def _xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_iou():
    a = _xyxy(0.5, 0.5, 0.2, 0.2)
    assert iou(a, a) == 1.0
    assert iou(a, _xyxy(0.9, 0.9, 0.05, 0.05)) == 0.0


def test_match_tp_fp_and_class():
    gts = [(0, 0.5, 0.5, 0.2, 0.2)]
    # a good same-class detection is a TP; a second one on the same GT is an FP
    dets = [(0, 0.9, 0.5, 0.5, 0.2, 0.2), (0, 0.8, 0.51, 0.5, 0.2, 0.2)]
    recs = match_detections(dets, gts, iou_thr=0.5)
    assert recs[0]["is_tp"] == 1 and recs[1]["is_tp"] == 0
    # right place, wrong class -> FP
    recs = match_detections([(1, 0.9, 0.5, 0.5, 0.2, 0.2)], gts, iou_thr=0.5)
    assert recs[0]["is_tp"] == 0


def test_ap_perfect_and_empty():
    scores = np.array([0.9, 0.8, 0.7]); tp = np.array([1, 1, 1])
    assert abs(_ap(scores, tp, n_gt=3) - 1.0) < 1e-6
    assert _ap(np.array([]), np.array([]), n_gt=2) == 0.0


def test_dece_calibrated_vs_overconfident():
    # scores equal to the empirical precision in each region -> near-zero D-ECE
    scores = np.array([0.2] * 10 + [0.8] * 10)
    correct = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1] + [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    ece_cal, _ = d_ece(scores, correct)
    # fully overconfident: always 0.99 but half wrong -> large D-ECE
    s2 = np.array([0.99] * 20); c2 = np.array([1, 0] * 10)
    ece_over, _ = d_ece(s2, c2)
    assert ece_cal < 0.15 and ece_over > 0.4
