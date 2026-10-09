# Thermal-detector calibration: does an airborne detector know when it is wrong?

A benchmark that fine-tunes an object detector on airborne thermal imagery, then degrades the
imagery the way a drone's sensor loses a target in the field and watches two things at once:
how far the detector's true accuracy (mAP) falls, and whether the confidence on its boxes stays
honest as that happens. The headline measurement is the calibration of the box confidence,
detection ECE, because a detector whose confidence stays high while its boxes go wrong is the
failure mode that matters for anything acting on its output.

This is the reported-versus-true-quality-under-controlled-intervention method (SkyDiscover,
CRN-Bench) pointed at fielded perception: inputs that look plausible but degrade in the field,
and ground truth that is scarce.

## The method

- **Model.** A YOLOv8 detector fine-tuned on HIT-UAV (high-altitude infrared thermal UAV
  imagery; people and vehicles). Thermal is a fielded sensor modality, not a proxy.
- **True vs reported quality.** True quality is mAP@0.5 against the real boxes. Reported quality
  is the detector's own box confidence, scored with detection ECE (D-ECE): bin detections by
  confidence, compare each bin's confidence to the fraction that are true positives.
- **Perturbations.** The field failure modes: range detail-loss (the target shrinks toward a few
  warm pixels), atmospheric blur, low thermal contrast, sensor noise, motion blur. All are
  photometric, so the ground-truth boxes are fixed across the whole sweep.
- **Dead control.** `background_marks` drops stray warm/cold blobs only where there are no
  targets and restores every box region pixel-for-pixel. A detector reading the targets cannot
  move. If mAP drops under it, the detector is reacting to irrelevant background, and that is the
  noise floor the real degradations have to clear.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                 # unit tests: perturbations, matching, AP, D-ECE
python -m src.run_local_smoke       # no-GPU end-to-end on synthetic thermal + a blob detector
```

Then on Modal (uses your account and GPU credits):

```bash
pip install modal && modal setup    # once

# 0. confirm HIT-UAV lays out the way the loader expects (no GPU, no training)
modal run modal_app.py::inspect

# 1. fine-tune + audit (downloads and fine-tunes the first time; cached in a Volume)
modal run modal_app.py --epochs 20 --max-test-images 200
```

Start with `::inspect`. HIT-UAV is distributed as image files with YOLO `.txt` labels; the
materialiser pairs them, infers the train/val/test split, and prints what it found. If the layout
differs from what it expects, the inspect step shows the directory tree so the pairing can be fixed
before any GPU time is spent.

## What you get

`results_thermal/` with `records.csv` (every detection, its score, and whether it is a true
positive), `aggregate.csv` (mAP, mean confidence, D-ECE per perturbation and severity), and
`integrity_summary.csv` (mAP drop vs confidence drop, ranked). Score it further with your
calibration code: the reliability binning and risk-coverage curve port directly.

## Honest limitations

- **Detector.** YOLOv8n is the floor model, chosen to fine-tune fast on an A10G. A larger backbone
  is a one-line swap and the natural A/B.
- **Perturbations are photometric.** Range is simulated as detail loss at fixed box coordinates
  rather than a true geometric rescale, which keeps labels fixed; true rescale with box transforms
  is the v2.
- **Confidence signal.** The box confidence is the detector's own. A stronger probe reads the
  per-class logits before NMS.

## Layout

```
modal_app.py            fine-tune + audit on Modal; local entrypoints `main` and `inspect`
src/perturbations.py    field degradations + the dead control (box-preserving)
src/dataset.py          YOLO-split loader (HIT-UAV test) + synthetic thermal source
src/detect.py           detections + IoU / TP-FP matching
src/metrics.py          mAP, D-ECE, risk-coverage, integrity summary, control check
src/eval.py             the perturbation sweep (one loop, smoke and Modal)
src/run_local_smoke.py  no-GPU end-to-end check
tests/                  perturbation and metric unit tests
```
