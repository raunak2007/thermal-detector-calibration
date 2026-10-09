"""The perturbation sweep: run a detector across samples, perturbations, and severities.

One loop, used by both the local smoke run and the Modal run. A `responder` turns an image
into detections and never sees the ground truth. Because every perturbation is photometric,
the ground-truth boxes are identical across the whole severity sweep for a given image, so
matching always uses the sample's own boxes and no label re-registration is needed.
"""

from __future__ import annotations

from collections import Counter

from .dataset import Sample
from .detect import Responder, match_detections
from .perturbations import SEVERITIES, apply


def run_sweep(
    samples: list[Sample],
    responder: Responder,
    perturbations: list[str],
    severities: list[float] = SEVERITIES,
    iou_thr: float = 0.5,
    progress: bool = True,
) -> tuple[list[dict], dict[int, int]]:
    """Returns (per-detection records, ground-truth count per class).

    The gt-per-class count is constant across perturbations (boxes never move), so it is the
    shared denominator for every (perturbation, severity) group's mAP.
    """
    gt_per_class: Counter = Counter()
    for s in samples:
        for (c, *_rest) in s.boxes:
            gt_per_class[int(c)] += 1

    records: list[dict] = []
    total = len(samples) * len(perturbations) * len(severities)
    done = 0
    for sample in samples:
        for pert in perturbations:
            for sev in severities:
                img = apply(pert, sample.image, sev, sample.boxes)
                dets = responder(img)
                for rec in match_detections(dets, sample.boxes, iou_thr):
                    rec.update(image_id=sample.id, perturbation=pert, severity=float(sev))
                    records.append(rec)
                done += 1
                if progress and done % 50 == 0:
                    print(f"  {done}/{total}")
    return records, dict(gt_per_class)
