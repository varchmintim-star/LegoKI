"""Ein vortrainiertes YOLO-Nano-Modell auf LEGO-Steine anpassen."""

import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def publish_model(best, data, run):
    target = ROOT / "models/lego_best.pt"
    shutil.copy2(best, target)
    manifest_path = Path(data).parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        metadata = {"trained_classes": sorted(manifest["class_counts"]["train"]),
                    "unlearned_classes": manifest["unlearned_classes"],
                    "source_run": str(run), "data": str(data),
                    "limitations": manifest["limitations"]}
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return target


def select_live_checkpoint(checkpoints, dataset):
    """Checkpoint für die feste Live-Schwelle 0.5 per Validierungs-F1 wählen."""
    import numpy as np
    from ultralytics import YOLO

    images = sorted((dataset / "images/val").glob("*"))
    scores = {}
    for checkpoint in checkpoints:
        model = YOLO(str(checkpoint))
        predictions = model.predict([str(p) for p in images], device="cpu", conf=0.5,
                                    imgsz=640, verbose=False)
        tp = fp = fn = 0
        for path, result in zip(images, predictions):
            height, width = result.orig_shape
            targets = []
            for row in (dataset / "labels/val" / f"{path.stem}.txt").read_text().splitlines():
                c, x, y, w, h = map(float, row.split())
                targets.append((int(c), np.array([(x-w/2)*width, (y-h/2)*height,
                                                  (x+w/2)*width, (y+h/2)*height])))
            used = set()
            for c, box in zip(result.boxes.cls.tolist(), result.boxes.xyxy.cpu().numpy()):
                matches = []
                for index, (target_class, target) in enumerate(targets):
                    if index in used or int(c) != target_class:
                        continue
                    overlap_size = np.maximum(0, np.minimum(box[2:], target[2:]) -
                                              np.maximum(box[:2], target[:2]))
                    overlap = float(np.prod(overlap_size))
                    union = float(np.prod(box[2:]-box[:2]) +
                                  np.prod(target[2:]-target[:2]) - overlap)
                    matches.append((overlap / union, index))
                if matches and max(matches)[0] >= 0.5:
                    tp += 1
                    used.add(max(matches)[1])
                else:
                    fp += 1
            fn += len(targets) - len(used)
        scores[str(checkpoint)] = {"tp": tp, "fp": fp, "fn": fn,
                                   "f1": 2*tp / max(1, 2*tp+fp+fn)}
    selected = max(checkpoints, key=lambda p: scores[str(p)]["f1"])
    return selected, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/prepared/lego_v1/data.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch <= 0:
        parser.error("Epochen und Batchgröße müssen > 0 sein.")
    if not args.data.exists():
        parser.error("Datensatz fehlt. Zuerst prepare_dataset.py ausführen.")
    import torch
    from ultralytics import YOLO

    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    (ROOT / "models").mkdir(exist_ok=True)
    model = YOLO(str(ROOT / "models/yolo11n.pt"))
    model.train(data=str(args.data), epochs=args.epochs, imgsz=640, batch=args.batch,
                device=device, workers=0, project=str(ROOT / "runs"), name="lego",
                seed=42, patience=20, amp=False, plots=True, cache=False,
                hsv_h=0.0, hsv_s=0.15, hsv_v=0.15,
                degrees=5, translate=0.1, scale=0.25, fliplr=0.5,
                mosaic=0.3, close_mosaic=10)
    selected, scores = select_live_checkpoint(
        [model.trainer.best, model.trainer.last], args.data.parent)
    target = publish_model(selected, args.data, model.trainer.save_dir)
    (model.trainer.save_dir / "live_selection.json").write_text(json.dumps({
        "criterion": "Validierungs-F1 bei confidence=0.5 und IoU=0.5",
        "selected": str(selected), "scores": scores}, indent=2) + "\n")
    print(f"Modell für Live-Erkennung: {target} (Quelle: {selected})")


if __name__ == "__main__":
    main()
