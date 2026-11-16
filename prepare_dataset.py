"""Aufnahmesitzungen getrennt für Training und Validierung vorbereiten."""

from collections import Counter
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import cv2

from collect_dataset import CLASSES


def prepare(raw=Path("dataset/raw"), output=Path("dataset/prepared/lego_v1")):
    sessions = []
    for session in sorted(raw.iterdir()):
        if not session.is_dir():
            continue
        images = sorted(p for p in (session / "images").glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            continue
        session_classes = json.loads((session / "classes.json").read_text())
        if (len(session_classes) not in (12, len(CLASSES)) or
                session_classes != list(CLASSES[:len(session_classes)])):
            raise ValueError(f"Abweichende Klassenreihenfolge: {session}")
        report_path = session / "review/labeling_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        excluded = report.get("exclude_from_training_until_reviewed", {})
        images = [p for p in images if p.name not in excluded]
        if images:
            sessions.append((session, images))
    if len(sessions) < 2:
        raise ValueError("Mindestens zwei beschriftete Aufnahmesitzungen erforderlich.")
    if output.exists():
        raise FileExistsError(f"{output} existiert bereits; vorhandene Daten bleiben erhalten.")

    # Die älteste Sitzung bleibt zur Validierung reserviert. Neuere Aufnahmen
    # ergänzen das Training, damit ältere gelernte Klassen nicht verloren gehen.
    splits = {"train": [], "val": sessions[:1]}
    heldout_groups = {}
    for session, images in sessions[1:]:
        report_path = session / "review/labeling_report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        heldout = set(report.get("validation_images", []))
        if not heldout.issubset({p.name for p in images}):
            raise ValueError(f"Unbekannte oder ausgeschlossene Kontrollbilder: {session}")
        train_images = [p for p in images if p.name not in heldout]
        val_images = [p for p in images if p.name in heldout]
        if train_images:
            splits["train"].append((session, train_images))
        if val_images:
            splits["val"].append((session, val_images))
            heldout_groups[session.name] = report.get("split_policy", "Explizite Kontrollgruppe")
    prepared = []
    counts = {split: Counter() for split in splits}
    hashes = {split: set() for split in splits}
    for split, groups in splits.items():
        for session, images in groups:
            session_class_count = len(json.loads((session / "classes.json").read_text()))
            for path in images:
                label = session / "labels" / f"{path.stem}.txt"
                frame = cv2.imread(str(path))
                if frame is None or not label.exists():
                    raise ValueError(f"Bild/Label fehlt oder ist unlesbar: {path}")
                rows = label.read_text().splitlines()
                if not rows:
                    raise ValueError(f"Leere Beschriftung ungeprüft: {label}")
                for row in rows:
                    c, x, y, w, h = map(float, row.split())
                    if not (c.is_integer() and 0 <= c < session_class_count and
                            0 < w <= 1 and 0 < h <= 1 and
                            x - w / 2 >= -1e-7 and x + w / 2 <= 1 + 1e-7 and
                            y - h / 2 >= -1e-7 and y + h / 2 <= 1 + 1e-7):
                        raise ValueError(f"Ungültige Markierung: {label}: {row}")
                    counts[split][CLASSES[int(c)]] += 1
                hashes[split].add(hashlib.sha256(frame.tobytes()).hexdigest())
                prepared.append((split, session, path, label))
    if hashes["train"] & hashes["val"]:
        raise ValueError("Identische Bilder in Training und Validierung gefunden.")
    for split, session, image, label in prepared:
        for directory in ("images", "labels"):
            (output / directory / split).mkdir(parents=True, exist_ok=True)
        stem = f"{session.name}_{image.stem}"
        shutil.copy2(image, output / "images" / split / f"{stem}{image.suffix}")
        shutil.copy2(label, output / "labels" / split / f"{stem}.txt")
    config = {"train": "images/train", "val": "images/val",
              "names": dict(enumerate(CLASSES))}
    # JSON ist gültiges YAML und vermeidet eine zusätzliche Abhängigkeit.
    (output / "data.yaml").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "sessions": {split: [s.name for s, _ in groups] for split, groups in splits.items()},
        "image_counts": {split: sum(len(images) for _, images in groups)
                         for split, groups in splits.items()},
        "class_counts": {split: dict(counter) for split, counter in counts.items()},
        "unlearned_classes": sorted(set(CLASSES) - counts["train"].keys()),
        "unvalidated_classes": sorted(counts["train"].keys() - counts["val"].keys()),
        "heldout_groups": heldout_groups,
        "limitations": "Kleiner Datensatz mit denselben Steinen und derselben Anlage. "
                        "Kein unabhängiger Praxistest. Keine Testaufnahmen vorhanden." +
                        (" Ein Teil der Validierung ist eine zurückgehaltene Ansichtsgruppe "
                         "innerhalb einer Trainingssitzung." if heldout_groups else ""),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return output / "data.yaml", manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dataset/prepared/lego_v3"))
    args = parser.parse_args()
    config, manifest = prepare(output=args.output)
    print(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
