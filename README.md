# Does an airborne thermal detector know when it is wrong?

A small benchmark that fine-tunes an object detector on airborne thermal imagery, then degrades
the test frame the way a drone's sensor loses a target in the field and watches two things at
once: how far the detector's true accuracy (mAP) falls, and whether the confidence on its boxes
stays honest as that happens. The question is whether a fielded perception system can tell from
its own confidence that the sensor has gone bad. The short answer, here, is no.

**One-pager with the figures and the full result: [`thermal_calibration_onepager.pdf`](thermal_calibration_onepager.pdf).**

## What it finds

Fine-tuned on HIT-UAV thermal person detection, the detector scores mAP@0.5 of 92.2 on clean
frames. Under every realistic degradation (range detail-loss, atmospheric blur, low thermal
contrast, sensor noise, motion blur) its true accuracy collapses, while a dead control that only
touches background pixels stays flat, so the collapse is a real reaction to the degraded targets.

- **It fails by going blind, not by lying.** The mechanism is recall collapse. Under the three
  harshest degradations, detections fall from ~2500 to as few as 20 and recall from 96% to under
  6%. mAP collapses because targets are missed, not because the surviving boxes are confidently
  wrong (the survivors are mostly *more* precise than on clean).
- **The confidence does not track it.** Mean box confidence falls far less than accuracy, and the
  confidence becomes badly miscalibrated, D-ECE rising from 0.17 to ~0.47–0.49, in the
  under-confident direction. The confidence stream neither spikes nor stays calibrated, so it will
  not tell you the sensor has degraded or how many targets you are now missing.
- **The ranking still triages.** Keeping only the top quarter of detections by confidence holds
  precision near 99% on clean and under degradation alike. That certifies the detections you keep,
  not the targets already missed.
- **It is a property of the detector, not the floor model.** YOLOv8s (3x the parameters)
  reproduces every part of this, down to the same motion-blur calibration exception. The larger
  model does not even beat the smaller on clean.
- **The standard fix backfires.** Temperature scaling fit on clean (the calibration you could
  actually deploy) makes the degraded D-ECE 26% *worse*, not better, because clean wants to sharpen
  its confidence and degraded wants to soften it, opposite directions a single scalar can't serve.
  Even the oracle per-condition temperature only halves it. You cannot calibrate your way out of a
  distribution shift from in-distribution data. See the appendix in the one-pager; run it with
  `python temp_scaling.py`.

Every number carries a frame-level bootstrap confidence interval; two caveats are stated rather
than hidden (atmospheric blur's D-ECE rests on only 20 detections, and motion blur genuinely
stays calibrated).

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q              # unit tests: perturbations, matching, AP, D-ECE
python -m src.run_local_smoke    # no-GPU end-to-end on synthetic thermal + a blob detector
```

Then on Modal (uses your account and GPU credits):

```bash
pip install modal && modal setup                         # once
modal run modal_app.py::inspect                          # confirm HIT-UAV layout, no GPU
modal run modal_app.py::main --epochs 20 --max-test-images 300
modal run modal_app.py::main --max-test-images 300 --model yolov8s.pt   # the model A/B
```

Each run writes `records.csv`, `aggregate.csv`, and `integrity_summary.csv` to the Modal volume
(under `results_thermal_<model>/`) and prints the baseline, the integrity summary, the control
check, and the risk-coverage table.

## Downstream: publishing into Lattice

[`lattice/`](lattice/) takes the result one level up the stack and maps
detections onto real Anduril Lattice `Entity` objects (via `anduril-lattice-sdk`),
but gated by the finding. Because the study showed the confidence is
untrustworthy under degradation and the detector fails by going blind, a naive
shim would be unsafe: Lattice has no confidence field to put the score in, and on
the worst frames a blind detector publishing almost nothing reads to an operator
as "area clear". So the mapper runs a trust gate off the measured reliability,
asserts a track's disposition only as hard as the regime earns, and on blind
frames suppresses the boxes and flips the producer's own `health` to FAIL instead
of going silent. Runs with no credentials against a mock transport:

```bash
python -m lattice.publish --severity 1.0     # the trusted/degraded/blind split + scorecard
```

See [`lattice/README.md`](lattice/README.md) for what is real (the SDK types and
field mapping, validated in tests) and what is stubbed and labelled (geolocation,
the live transport).

## Method

- **True vs reported quality.** True quality is mAP@0.5 against the real boxes. Reported quality is
  the detector's box confidence, scored with detection ECE (D-ECE): bin detections by confidence,
  compare each bin's confidence to the fraction that are true positives.
- **Perturbations** are the field failure modes, all photometric so the ground-truth boxes stay
  fixed across the whole sweep: `range_detail_loss`, `atmospheric_blur`, `low_contrast`,
  `sensor_noise`, `motion_blur`.
- **Dead control.** `background_marks` alters only pixels where there are no targets. If mAP drops
  under it, the detector is reacting to background; that drop is the noise floor the real
  degradations have to clear. Here it moves mAP ~3–4 points while the real degradations move it
  40–90.

This is the reported-vs-true-quality-under-controlled-intervention method (after SkyDiscover and
CRN-Bench) pointed at fielded perception.

## Honest limits

- YOLOv8n and YOLOv8s are small; both saturate at 92.2 on clean HIT-UAV, so this task does not
  stress capacity. A harder, multi-class aerial set is the natural next dataset.
- Perturbations are photometric; range is simulated as detail-loss at fixed box coordinates rather
  than a true geometric rescale, which keeps labels fixed across the sweep.
- Single class (person), one thermal dataset. The transferable finding is the pattern (confidence
  failing to signal a recall collapse), not the exact magnitudes.
- Confidence intervals are a frame-level bootstrap; mAP and recall are point estimates, since
  missed targets are not in the per-detection log.

## Layout

```
modal_app.py            fine-tune + audit on Modal; local entrypoints `main` and `inspect`
src/perturbations.py    field degradations + the dead control (box-preserving)
src/dataset.py          YOLO-split loader (HIT-UAV test) + synthetic thermal source
src/detect.py           detections + IoU / TP-FP matching
src/metrics.py          mAP, D-ECE, risk-coverage, integrity summary, control check
src/eval.py             the perturbation sweep (one loop, smoke and Modal)
lattice/                maps detections onto Lattice Entities, gated by the finding
tests/                  perturbation, metric, and Lattice-mapping unit tests
```
