"""Live-Erkennung mit dem trainierten LEGO-Modell."""

import argparse
import json
import os
from pathlib import Path
import platform

import cv2

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/lego_best.pt")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.camera < 0 or not 0 < args.confidence <= 1:
        parser.error("Kameraindex muss >= 0 sein, confidence zwischen 0 und 1.")
    if not args.weights.exists():
        parser.error("Trainiertes Modell fehlt. Zuerst train_model.py ausführen.")
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metadata = args.weights.with_suffix(".json")
    active = None
    if metadata.exists():
        trained = set(json.loads(metadata.read_text())["trained_classes"])
        active = [i for i, name in model.names.items() if name in trained]
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    window = "LEGO KI - Farbe und Groesse"
    try:
        if not cap.isOpened():
            raise RuntimeError("Kamera nicht verfügbar. Index und Zugriffsrechte prüfen.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Kein Kamerabild empfangen.")
            result = model.predict(frame, conf=args.confidence, classes=active,
                                   device=args.device, imgsz=640, verbose=False)[0]
            cv2.imshow(window, result.plot())
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
