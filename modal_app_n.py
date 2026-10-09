"""Run the audit on Modal: fine-tune YOLOv8 on HIT-UAV, then probe its confidence calibration.

Three steps, one GPU function:
  1. materialise HIT-UAV (Voxel51/Kiuyha thermal UAV set) into YOLO layout,
  2. fine-tune a YOLOv8 detector on the train split,
  3. run the perturbation sweep on the test split and return per-detection records.
Metrics and plots run locally on the way back.

First, confirm the dataset lays out the way the loader expects (cheap, no GPU, no training):
    modal run modal_app.py::inspect

Then the real run (downloads + fine-tunes the first time; weights are cached in a Volume):
    modal run modal_app.py --epochs 20 --max-test-images 200

The audit output is the construction-drawing study's structure pointed at thermal detection:
true quality is mAP@0.5, reported quality is the detector's box confidence scored with D-ECE,
and the dead control marks only the background with every target region restored.
"""

import modal

app = modal.App("thermal-detector-calibration")

HF_REPO = "Kiuyha/hit-uav-thermal-human-detection"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1",
        "ultralytics==8.3.0",          # YOLOv8 train + predict
        "huggingface_hub", "datasets",
        "pillow", "numpy<2.2", "scipy", "pandas", "pyyaml",
    )
    .add_local_python_source("src")
)

cache = modal.Volume.from_name("hituav-cache", create_if_missing=True)
CACHE = "/cache"
ALL_PERTURBATIONS = [
    "clean", "range_detail_loss", "atmospheric_blur", "low_contrast",
    "sensor_noise", "motion_blur", "background_marks",
]


# ---------------------------------------------------------------------------
# Materialise HIT-UAV into a YOLO directory (images/{split}, labels/{split}).
# ---------------------------------------------------------------------------
def _materialise(dest: str) -> dict:
    """Download the HF repo and arrange it as YOLO layout. Returns paths + class names.

    HIT-UAV ships as image files with matching YOLO .txt labels. This walks the snapshot,
    pairs images to labels by stem, infers train/val/test from path names (falling back to a
    deterministic split), and writes a clean tree plus data.yaml. It prints what it found so a
    first run can be checked with `inspect` before any training.
    """
    import os
    import shutil
    from pathlib import Path

    import yaml
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(HF_REPO, repo_type="dataset", cache_dir=f"{CACHE}/hf"))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in root.rglob("*") if p.suffix.lower() in img_exts]
    labels = {p.stem: p for p in root.rglob("*.txt") if p.parent.name.lower() != "."}

    def split_of(p: Path) -> str:
        parts = {s.lower() for s in p.parts}
        for s in ("test", "val", "valid", "train"):
            if s in parts:
                return "val" if s in ("val", "valid") else s
        return "auto"

    pairs = [(p, labels.get(p.stem)) for p in images if p.stem in labels]
    print(f"[materialise] {len(images)} images, {len(labels)} label files, {len(pairs)} paired")
    if not pairs:
        tree = "\n".join(sorted({str(p.relative_to(root).parent) for p in images})[:40])
        raise RuntimeError(f"No image/label pairs found. Directories under the snapshot:\n{tree}")

    # class names: look for a data.yaml / classes.txt in the snapshot, else infer from labels
    names = None
    for cand in list(root.rglob("*.yaml")) + list(root.rglob("classes.txt")):
        try:
            if cand.suffix == ".yaml":
                y = yaml.safe_load(cand.read_text())
                if isinstance(y, dict) and "names" in y:
                    names = y["names"]; break
            else:
                names = {i: n for i, n in enumerate(cand.read_text().split())}; break
        except Exception:
            continue
    if names is None:
        max_cls = 0
        for _, lp in pairs:
            for line in lp.read_text().splitlines():
                if line.split():
                    max_cls = max(max_cls, int(float(line.split()[0])))
        names = {i: f"class_{i}" for i in range(max_cls + 1)}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    print(f"[materialise] classes: {names}")

    d = Path(dest)
    for p, lp in pairs:
        sp = split_of(p)
        if sp == "auto":
            sp = "val" if (hash(p.stem) % 5 == 0) else "train"
        for sub in ("images", "labels"):
            (d / sub / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy(p, d / "images" / sp / p.name)
        shutil.copy(lp, d / "labels" / sp / (p.stem + ".txt"))

    splits = {s.name for s in (d / "images").iterdir()} if (d / "images").exists() else set()
    test_split = "test" if "test" in splits else ("val" if "val" in splits else "train")
    data_yaml = d / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(d), "train": "images/train",
        "val": f"images/{'val' if 'val' in splits else 'train'}",
        "names": {int(k): v for k, v in names.items()},
    }))
    print(f"[materialise] splits present: {sorted(splits)}; audit uses '{test_split}'")
    return {"data_yaml": str(data_yaml),
            "test_images": str(d / "images" / test_split),
            "test_labels": str(d / "labels" / test_split),
            "names": {int(k): v for k, v in names.items()}}


@app.function(image=image, gpu="A10G", volumes={CACHE: cache}, timeout=4 * 60 * 60)
def run(epochs: int = 20, max_test_images: int = 200,
        severities: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]):
    import os
    from ultralytics import YOLO

    from src.dataset import load_yolo_split
    from src.detect import Det
    from src.eval import run_sweep

    paths = _materialise(f"{CACHE}/hituav_yolo")

    # fine-tune (cached across runs by the Volume)
    weights = f"{CACHE}/yolov8n_hituav.pt"
    if not os.path.exists(weights):
        model = YOLO("yolov8n.pt")
        model.train(data=paths["data_yaml"], epochs=epochs, imgsz=640,
                    project=f"{CACHE}/runs", name="hituav", exist_ok=True, verbose=False)
        best = f"{CACHE}/runs/hituav/weights/best.pt"
        YOLO(best).save(weights)
        cache.commit()
    model = YOLO(weights)

    def responder(image) -> list[Det]:
        r = model.predict(image, imgsz=640, verbose=False, conf=0.05)[0]
        out = []
        for b in r.boxes:
            cx, cy, w, h = b.xywhn[0].tolist()
            out.append((int(b.cls.item()), float(b.conf.item()), cx, cy, w, h))
        return out

    samples = load_yolo_split(paths["test_images"], paths["test_labels"], max_images=max_test_images)
    print(f"[audit] {len(samples)} test frames, {sum(len(s.boxes) for s in samples)} boxes")
    records, gt_per_class = run_sweep(samples, responder, ALL_PERTURBATIONS, severities)

    # Persist metrics + CSVs to the Volume here, on Modal, so a client disconnect
    # (closed lid, wifi blip) never loses the GPU run. Retrieve later with
    #   modal volume get hituav-cache results_thermal ./results_thermal
    try:
        import pathlib

        from src.metrics import (aggregate, baseline_calibration, control_check,
                                 integrity_summary, risk_coverage, to_frame)

        gpc = {int(k): int(v) for k, v in gt_per_class.items()}
        df = to_frame(records)
        agg = aggregate(df, gpc)
        base = baseline_calibration(df, gpc)
        summary = integrity_summary(agg)

        outd = pathlib.Path(f"{CACHE}/results_thermal")
        outd.mkdir(exist_ok=True)
        df.to_csv(outd / "records.csv", index=False)
        agg.to_csv(outd / "aggregate.csv", index=False)
        summary.to_csv(outd / "integrity_summary.csv", index=False)
        cache.commit()

        print(f"\nclasses: {paths['names']}")
        print("\n=== baseline (clean) ===")
        print(f"mAP@0.5 {base['map']}  precision {base['precision']}  "
              f"mean confidence {base['mean_confidence']}  D-ECE {base['d_ece']}")
        print("\n=== integrity summary (mAP drop vs confidence drop, largest gap first) ===")
        print(summary.to_string(index=False))
        print("\n=== control check ===")
        print(control_check(agg))
        print("\n=== risk-coverage (clean) ===")
        print(risk_coverage(df).to_string(index=False))
        print(f"\n[audit] wrote CSVs to the volume at {CACHE}/results_thermal/")
    except Exception as e:  # never let the metrics step lose the raw GPU result
        print(f"[audit] metrics/persist step failed (raw records still returned): {e!r}")

    return {"records": records, "gt_per_class": gt_per_class, "names": paths["names"]}


@app.local_entrypoint()
def inspect():
    """Download + lay out the dataset and print what the loader will see. No GPU, no training."""
    print(_inspect.remote())


@app.function(image=image, volumes={CACHE: cache}, timeout=60 * 60)
def _inspect():
    from src.dataset import load_yolo_split
    paths = _materialise(f"{CACHE}/hituav_yolo")
    s = load_yolo_split(paths["test_images"], paths["test_labels"], max_images=3)
    return {"paths": paths, "sample_box_counts": [len(x.boxes) for x in s]}


@app.local_entrypoint()
def main(epochs: int = 20, max_test_images: int = 200, out: str = "results_thermal",
         severities: str = "0,0.25,0.5,0.75,1.0"):
    import pathlib

    from src.metrics import (aggregate, baseline_calibration, control_check,
                             integrity_summary, risk_coverage, to_frame)

    sev = [float(x) for x in severities.split(",") if x.strip() != ""]
    res = run.remote(epochs=epochs, max_test_images=max_test_images, severities=sev)
    records, gt_per_class, names = res["records"], res["gt_per_class"], res["names"]
    gt_per_class = {int(k): int(v) for k, v in gt_per_class.items()}

    df = to_frame(records)
    agg = aggregate(df, gt_per_class)
    base = baseline_calibration(df, gt_per_class)
    summary = integrity_summary(agg)

    d = pathlib.Path(out); d.mkdir(exist_ok=True)
    df.to_csv(d / "records.csv", index=False)
    agg.to_csv(d / "aggregate.csv", index=False)
    summary.to_csv(d / "integrity_summary.csv", index=False)

    print(f"\nclasses: {names}")
    print("\n=== baseline (clean) ===")
    print(f"mAP@0.5 {base['map']}  precision {base['precision']}  "
          f"mean confidence {base['mean_confidence']}  D-ECE {base['d_ece']}")
    print("\n=== integrity summary (mAP drop vs confidence drop, largest gap first) ===")
    print(summary.to_string(index=False))
    print("\n=== control check ===")
    print(control_check(agg))
    print("\n=== risk-coverage (clean) ===")
    print(risk_coverage(df).to_string(index=False))
    print(f"\nwrote CSVs to {out}/")
