"""LEGO-Farbe und 2x2/2x4-Form anhand eines leeren Referenzbilds erkennen."""

import argparse
import platform

import cv2
import numpy as np


def color_name(hsv_pixels):
    """Medianwerte verringern den Einfluss einzelner Glanzlichter."""
    h, s, v = np.median(hsv_pixels, axis=0)
    if v < 65:
        return "Schwarz"
    if s < 45:
        return "Weiss" if v > 185 else "Grau"
    if (h < 22 or h > 170) and v < 145:
        return "Braun"
    if h < 10 or h >= 170:
        return "Rot"
    if h < 23:
        return "Orange"
    if h < 38:
        return "Gelb"
    if h < 90:
        return "Gruen"
    if h < 133:
        return "Blau"
    return "Violett/Rosa"


def detect(frame, background, threshold=35, min_area=900):
    current = cv2.GaussianBlur(frame, (7, 7), 0)
    reference = cv2.GaussianBlur(background, (7, 7), 0)
    difference = cv2.absdiff(current, reference).max(axis=2)
    mask = (difference > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= frame.shape[1] - 1 or y + h >= frame.shape[0] - 1:
            continue
        rect = cv2.minAreaRect(contour)
        short, long = sorted(rect[1])
        if short < 1 or area / (short * long) < 0.78:
            continue
        ratio = long / short
        if ratio <= 1.25:
            size = "2x2?"
        elif 1.7 <= ratio <= 2.3:
            size = "2x4?"
        else:
            size = "Unbekannt"
        interior = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(interior, [contour], -1, 255, cv2.FILLED)
        interior = cv2.erode(interior, np.ones((9, 9), np.uint8))
        pixels = hsv[interior > 0]
        if not pixels.size:
            continue
        detections.append((cv2.boxPoints(rect).astype(np.int32),
                           color_name(pixels), size))
    return detections, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=35,
                        help="Differenz zum Hintergrund (1–255, Standard: 35)")
    parser.add_argument("--min-area", type=int, default=900,
                        help="Minimale Steinfläche in Pixeln")
    args = parser.parse_args()
    if args.camera < 0 or not 1 <= args.threshold <= 255 or args.min_area <= 0:
        parser.error("Ungültiger Kameraindex, Schwellwert oder Mindestfläche.")
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    window = "LEGO - Farbe und Groesse"
    background = None
    try:
        if not cap.isOpened():
            raise RuntimeError("Kamera nicht verfügbar. Kameraindex und Zugriffsrechte prüfen.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Leere Fläche zeigen, dann B drücken. Danach einzelne Steine auflegen.")
        print("B: Hintergrund erneuern | Q/Escape: Beenden")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Kein Kamerabild empfangen.")
            display = frame.copy()
            if background is None:
                message = "Leere Flaeche zeigen, dann B druecken"
            else:
                found, _ = detect(frame, background, args.threshold, args.min_area)
                message = f"{len(found)} Objekt(e) | B: Hintergrund | Q: Ende"
                for box, color, size in found:
                    cv2.polylines(display, [box], True, (0, 255, 0), 2)
                    x, y = box.min(axis=0)
                    cv2.putText(display, f"{color} {size}", (int(x), max(55, int(y) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(display, (0, 0), (display.shape[1], 35), (0, 0, 0), -1)
            cv2.putText(display, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("b"), ord("B")):
                background = frame.copy()
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
