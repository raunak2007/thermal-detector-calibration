"""Scoring: true quality (mAP), reported quality (confidence calibration), and the control.

Every row is one detection: {image_id, perturbation, severity, cls, score, is_tp}. From those:

- `average_precision` / `aggregate` give mAP@0.5, the true quality, per (perturbation, severity).
- `d_ece` gives detection Expected Calibration Error: bin detections by confidence, compare the
  mean confidence in each bin to the fraction that are true positives. A detector whose 0.9-confidence
  boxes are right 90% of the time has D-ECE near 0; one that stays confident while its boxes go wrong
  has a large D-ECE. That is the reported-vs-true question for detection.
- `integrity_summary` tracks how mAP and mean confidence move from the clean image to full severity,
  so a model that keeps reporting high confidence while mAP collapses is visible as a positive gap.
- `control_check` reads the dead control: marks only in the background should not move mAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .perturbations import CONTROL


def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df["score"] = df["score"].astype(float)
    df["is_tp"] = df["is_tp"].astype(int)
    return df


def _ap(scores: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """VOC all-point average precision for one class."""
    if n_gt <= 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(is_tp[order]); fp = np.cumsum(1 - is_tp[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate(([0.0], recall, [recall[-1] if recall[-1] < 1 else 1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_ap(group: pd.DataFrame, gt_per_class: dict[int, int]) -> float:
    aps = []
    for cls, n_gt in gt_per_class.items():
        if n_gt <= 0:
            continue
        sub = group[group["cls"] == cls]
        aps.append(_ap(sub["score"].to_numpy(), sub["is_tp"].to_numpy(), n_gt))
    aps = [a for a in aps if a == a]
    return float(np.mean(aps)) if aps else float("nan")


def d_ece(scores: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(scores); ece = 0.0; rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (scores >= lo) & (scores < hi) if hi < 1.0 else (scores >= lo) & (scores <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            rows.append((lo, hi, 0, np.nan, np.nan)); continue
        prec = float(correct[in_bin].mean()); conf = float(scores[in_bin].mean())
        ece += (cnt / n) * abs(prec - conf)
        rows.append((lo, hi, cnt, prec, conf))
    tbl = pd.DataFrame(rows, columns=["bin_lo", "bin_hi", "n", "precision", "mean_conf"])
    return ece, tbl


def aggregate(df: pd.DataFrame, gt_per_class: dict[int, int]) -> pd.DataFrame:
    """Per (perturbation, severity): mAP, mean detection confidence, D-ECE, detection count."""
    rows = []
    for (pert, sev), g in df.groupby(["perturbation", "severity"]):
        s = g["score"].to_numpy(); tp = g["is_tp"].to_numpy()
        ece, _ = d_ece(s, tp)
        rows.append(dict(
            perturbation=pert, severity=float(sev),
            map=100.0 * mean_ap(g, gt_per_class),
            mean_confidence=100.0 * float(s.mean()) if len(s) else float("nan"),
            d_ece=ece, n_dets=len(g),
            precision=100.0 * float(tp.mean()) if len(tp) else float("nan"),
        ))
    return pd.DataFrame(rows).sort_values(["perturbation", "severity"]).reset_index(drop=True)


def integrity_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Per perturbation: how mAP (true) and confidence (reported) move clean -> max severity."""
    rows = []
    for pert, sub in agg.groupby("perturbation"):
        sub = sub.sort_values("severity")
        m0, mm = sub["map"].iloc[0], sub["map"].iloc[-1]
        c0, cm = sub["mean_confidence"].iloc[0], sub["mean_confidence"].iloc[-1]
        rows.append(dict(
            perturbation=pert, is_control=(pert == CONTROL),
            map_at_0=round(m0, 1), map_at_max=round(mm, 1), map_drop=round(m0 - mm, 1),
            conf_at_0=round(c0, 1), conf_at_max=round(cm, 1), conf_drop=round(c0 - cm, 1),
            dece_at_0=round(sub["d_ece"].iloc[0], 3), dece_at_max=round(sub["d_ece"].iloc[-1], 3),
            integrity_gap=round((m0 - mm) - (c0 - cm), 1),
        ))
    return pd.DataFrame(rows).sort_values("integrity_gap", ascending=False).reset_index(drop=True)


def control_check(agg: pd.DataFrame) -> dict:
    sub = agg[agg["perturbation"] == CONTROL].sort_values("severity")
    if sub.empty:
        return {"control_present": False}
    return {
        "control_present": True, "control": CONTROL,
        "map_drop_under_control_pts": round(float(sub["map"].iloc[0] - sub["map"].iloc[-1]), 1),
        "interpretation": ("Marks only in the background should not move mAP. A drop means the "
                           "detector is reacting to irrelevant background, not the targets."),
    }


def _clean_slice(df: pd.DataFrame) -> pd.DataFrame:
    """One canonical clean pass. Severity 0 is identical under every perturbation, so pooling
    all of them would multiply the detection count against a single GT count; take just the
    'clean' perturbation (or, if absent, a single perturbation's severity-0 rows)."""
    clean = df[(df["perturbation"] == "clean") & (df["severity"] == 0.0)]
    if clean.empty:
        zero = df[df["severity"] == 0.0]
        if not zero.empty:
            first = zero["perturbation"].iloc[0]
            clean = zero[zero["perturbation"] == first]
    return clean


def baseline_calibration(df: pd.DataFrame, gt_per_class: dict[int, int]) -> dict:
    """Clean-image headline: mAP, mean confidence, and D-ECE on the undisturbed frames."""
    clean = _clean_slice(df)
    if clean.empty:
        return {"baseline_present": False}
    s = clean["score"].to_numpy(); tp = clean["is_tp"].to_numpy()
    ece, tbl = d_ece(s, tp)
    return {
        "baseline_present": True,
        "map": round(100.0 * mean_ap(clean, gt_per_class), 1),
        "precision": round(100.0 * float(tp.mean()), 1),
        "mean_confidence": round(100.0 * float(s.mean()), 1),
        "d_ece": round(ece, 3),
        "reliability_bins": tbl,
    }


def risk_coverage(df: pd.DataFrame, points=(1.0, 0.75, 0.5, 0.25)) -> pd.DataFrame:
    """On the clean frames: precision of the kept detections as coverage drops, most-confident first."""
    clean = _clean_slice(df).sort_values("score", ascending=False).reset_index(drop=True)
    n = len(clean)
    if n == 0:
        return pd.DataFrame(columns=["coverage", "precision", "kept"])
    rows = []
    for c in points:
        k = max(1, int(round(c * n)))
        rows.append(dict(coverage=c, precision=round(100.0 * clean["is_tp"].iloc[:k].mean(), 1), kept=k))
    return pd.DataFrame(rows)
