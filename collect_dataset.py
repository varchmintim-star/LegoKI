"""Kamerabilder einfrieren, LEGO-Steine markieren und im YOLO-Format speichern."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import textwrap
from uuid import uuid4

import cv2
import numpy as np


COLORS = ("gelb", "gruen", "hellgruen", "weiss", "blau", "rot")
CLASSES = tuple(f"{color}_{size}" for color in COLORS for size in ("2x2", "2x4"))


def yolo_box(box, width, height):
    x, y, w, h = box
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("Rahmen liegt außerhalb des Bildes oder ist leer.")
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def save_sample(root, frame, annotations):
    if not annotations:
        raise ValueError("Zuerst mindestens einen Stein markieren.")
    height, width = frame.shape[:2]
    lines = []
    for class_id, box in annotations:
        coords = yolo_box(box, width, height)
        lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in coords))
    stem = uuid4().hex
    image_path = root / "images" / f"{stem}.jpg"
    label_path = root / "labels" / f"{stem}.txt"
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError("Bild konnte nicht gespeichert werden.")
    label_path.write_text("\n".join(lines) + "\n")
    return image_path


def save_unlabeled(root, frame):
    folder = root / "unlabeled"
    folder.mkdir(exist_ok=True)
    path = folder / f"{uuid4().hex}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError("Foto konnte nicht gespeichert werden.")
    return path


class AnnotationEditor:
    """Ein Rahmen bleibt unbestätigt, bis Farbe UND Größe gewählt wurden."""

    def __init__(self):
        self.frozen = None
        self.annotations = []
        self.pending = None
        self.color = None
        self.size = None
        self.drag = None
        self.endpoint = None
        self.status = "Bild einfrieren, dann Rahmen ziehen."

    def new_box(self, box):
        if self.pending is not None:
            return
        self.pending = box
        self.color = self.size = None
        self.status = "Jetzt Farbe und Groesse waehlen, dann bestaetigen."

    def confirm(self):
        if self.pending is None or self.color is None or self.size is None:
            self.status = "Bitte Rahmen, Farbe UND Groesse waehlen."
            return False
        self.annotations.append((self.color * 2 + self.size, self.pending))
        self.pending = None
        self.color = self.size = None
        self.status = "Markiert. Naechsten Rahmen ziehen oder Bild speichern."
        return True

    def cancel(self):
        self.pending = None
        self.color = self.size = None
        self.drag = self.endpoint = None

    def freeze(self, frame):
        self.frozen = frame.copy()
        self.annotations = []
        self.cancel()
        self.status = "Rahmen um einen Stein ziehen."

    def resume(self):
        self.frozen = None
        self.annotations = []
        self.cancel()

    def can_save(self):
        return self.frozen is not None and bool(self.annotations) and self.pending is None



class CaptureUI:
    WIDTH = 860
    HEIGHT = 700
    PANEL = 320

    def __init__(self, root):
        self.root = root
        self.editor = AnnotationEditor()
        self.live = None
        self.buttons = []
        self.view_width = self.view_height = 0

    def action(self, name):
        e = self.editor
        if name.startswith("color:") and e.pending is not None:
            e.color = int(name.split(":")[1])
        elif name.startswith("size:") and e.pending is not None:
            e.size = int(name.split(":")[1])
        elif name == "confirm":
            e.confirm()
        elif name == "cancel":
            e.cancel()
            e.status = "Rahmen verworfen. Neuen Rahmen ziehen."
        elif name == "freeze" and self.live is not None:
            if e.frozen is None:
                e.freeze(self.live)
            else:
                e.status = "Zum Verwerfen des Bildes V druecken."
        elif name == "discard":
            e.resume()
            e.status = "Livebild. Leertaste friert ein."
        elif name == "undo":
            if e.pending is not None:
                e.cancel()
            elif e.annotations:
                e.annotations.pop()
        elif name == "save":
            if e.can_save():
                path = save_sample(self.root, e.frozen, e.annotations)
                print(f"Gespeichert: {path}")
                e.resume()
                e.status = "Gespeichert! Steine neu anordnen."
            else:
                e.status = "Alle Rahmen bestaetigen; mindestens ein Stein erforderlich."
        elif name == "photo":
            if e.annotations or e.pending is not None:
                e.status = "Markierungen vorhanden. Mit S speichern oder V verwerfen."
            elif self.live is not None:
                frame = self.live if e.frozen is None else e.frozen
                path = save_unlabeled(self.root, frame)
                print(f"Zum Labeln vorgemerkt: {path}")
                e.resume()
                e.status = "Foto gespeichert. Labeling spaeter durch Codex."

    def mouse(self, event, x, y, flags, userdata):
        e = self.editor
        if event == cv2.EVENT_LBUTTONDOWN:
            for (left, top, right, bottom), name in self.buttons:
                if left <= x < right and top <= y < bottom:
                    self.action(name)
                    return
        if e.frozen is None or e.pending is not None:
            return
        inside = 0 <= x < self.view_width and 0 <= y < self.view_height
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            e.drag = e.endpoint = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and e.drag is not None:
            e.endpoint = (min(max(x, 0), self.view_width), min(max(y, 0), self.view_height))
        elif event == cv2.EVENT_LBUTTONUP and e.drag is not None:
            end = (min(max(x, 0), self.view_width), min(max(y, 0), self.view_height))
            height, width = e.frozen.shape[:2]
            x0, x1 = sorted((e.drag[0], end[0]))
            y0, y1 = sorted((e.drag[1], end[1]))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                left = round(x0 * width / self.view_width)
                top = round(y0 * height / self.view_height)
                right = round(x1 * width / self.view_width)
                bottom = round(y1 * height / self.view_height)
                e.new_box((left, top, right - left, bottom - top))
            e.drag = e.endpoint = None

    def key(self, key):
        e = self.editor
        if ord("1") <= key <= ord("6"):
            self.action(f"color:{key - ord('1')}")
        elif key in (ord("t"), ord("z")):
            self.action("size:0")
        elif key in (ord("r"), ord("x")):
            self.action("size:1")
        else:
            action = {32: "freeze", 13: "confirm", 10: "confirm", 27: "cancel",
                      ord("s"): "save", ord("u"): "undo", ord("v"): "discard",
                      ord("p"): "photo"}.get(key)
            if action:
                self.action(action)

    def render(self):
        e = self.editor
        frame = self.live if e.frozen is None else e.frozen
        height, width = frame.shape[:2]
        scale = min(self.WIDTH / width, self.HEIGHT / height)
        self.view_width, self.view_height = round(width * scale), round(height * scale)
        display = np.full((self.HEIGHT, self.WIDTH + self.PANEL, 3), 28, np.uint8)
        display[:self.view_height, :self.view_width] = cv2.resize(
            frame, (self.view_width, self.view_height))
        boxes = [(box, CLASSES[class_id], (80, 240, 80)) for class_id, box in e.annotations]
        if e.pending is not None:
            color_label = COLORS[e.color] if e.color is not None else "Farbe?"
            size_label = ("2x2", "2x4")[e.size] if e.size is not None else "Groesse?"
            boxes.append((e.pending, f"{color_label} {size_label} - Enter", (0, 215, 255)))
        for (x, y, w, h), label, color in boxes:
            p1 = (round(x * scale), round(y * scale))
            p2 = (round((x + w) * scale), round((y + h) * scale))
            cv2.rectangle(display, p1, p2, color, 2)
            cv2.putText(display, label, (p1[0], max(18, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        if e.drag is not None:
            cv2.rectangle(display, e.drag, e.endpoint, (0, 215, 255), 2)
        self.buttons = []

        def text(label, y, color=(230, 230, 230)):
            cv2.putText(display, label, (self.WIDTH + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        def button(label, name, x, y, w=142, selected=False, enabled=True):
            left = self.WIDTH + 12 + x
            # Alle Trefferbereiche liegen außerhalb des eigentlichen Kamerabilds.
            rect = (left, y, left + w, y + 30)
            color = (100, 95, 35) if selected else (65, 65, 65) if enabled else (38, 38, 38)
            cv2.rectangle(display, rect[:2], rect[2:], color, -1)
            cv2.putText(display, label, (left + 7, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255) if enabled else (110, 110, 110), 1)
            if enabled:
                self.buttons.append((rect, name))

        text("LEGO TRAININGSBILDER", 25)
        text("LIVE" if e.frozen is None else "STANDBILD", 49, (0, 215, 255))
        button("Bild einfrieren", "freeze", 0, 64, 294, enabled=e.frozen is None)
        text("1. Rahmen ziehen", 120)
        text("2. Farbe waehlen", 145)
        for i, label in enumerate(("1 Gelb", "2 Gruen", "3 Hellgruen", "4 Weiss", "5 Blau", "6 Rot")):
            button(label, f"color:{i}", (i % 2) * 152, 155 + (i // 2) * 36,
                   selected=e.color == i, enabled=e.pending is not None)
        text("3. Groesse waehlen", 284)
        button("2 x 2 (Z)", "size:0", 0, 294, selected=e.size == 0, enabled=e.pending is not None)
        button("2 x 4 (X)", "size:1", 152, 294, selected=e.size == 1, enabled=e.pending is not None)
        button("Bestaetigen (Enter)", "confirm", 0, 336, 294,
               enabled=e.pending is not None and e.color is not None and e.size is not None)
        button("Rahmen weg (Esc)", "cancel", 0, 374, enabled=e.pending is not None)
        button("Zurueck (U)", "undo", 152, 374, enabled=bool(e.annotations) or e.pending is not None)
        text(f"{len(e.annotations)} Steine bestaetigt", 430)
        button("Bild speichern (S)", "save", 0, 440, 294, enabled=e.can_save())
        button("Bild verwerfen (V)", "discard", 0, 477, 294, enabled=e.frozen is not None)
        button("Nur Foto speichern (P)", "photo", 0, 518, 294,
               enabled=not e.annotations and e.pending is None)
        for i, line in enumerate(textwrap.wrap(e.status, 39)):
            text(line, 580 + i * 17, (0, 215, 255))
        return display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("dataset/raw"))
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("Kameraindex muss >= 0 sein.")
    root = args.output / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8])
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    window = "LEGO Trainingsbilder"
    try:
        if not cap.isOpened():
            raise RuntimeError("Kamera nicht verfügbar. Index und Kamerazugriff prüfen.")
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        (root / "classes.json").write_text(json.dumps(CLASSES, indent=2) + "\n")
        (root / "session.json").write_text(json.dumps({
            "camera_index": args.camera,
            "created_at": datetime.now().astimezone().isoformat(),
            "annotation_format": "YOLO: class_id center_x center_y width height (normalized)",
        }, indent=2) + "\n")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ui = CaptureUI(root)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, ui.mouse)
        print("Leertaste: Standbild | Rahmen ziehen, danach Farbe + Groesse anklicken.")
        print("Enter: Rahmen bestaetigen | S: Bild speichern | V: verwerfen | Q: Ende")
        print("P: Nur Foto speichern, spaeter durch Codex beschriften lassen.")
        print(f"Aufnahmesitzung: {root}")
        while True:
            ok, ui.live = cap.read()
            if not ok:
                raise RuntimeError("Kein Kamerabild empfangen.")
            cv2.imshow(window, ui.render())
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            ui.key(key)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
