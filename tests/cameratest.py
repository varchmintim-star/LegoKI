"""Live-Vorschau einer Webcam; beenden mit Q oder Escape."""

import argparse
import platform

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1,
                        help="Kameraindex (Standard: 1)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.camera < 0 or args.width <= 0 or args.height <= 0:
        parser.error("Kameraindex muss >= 0 sein, Breite und Höhe müssen > 0 sein.")

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    window = "LEGO Sorter - Webcam"
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Kamera {args.camera} konnte nicht geöffnet werden. "
                "USB-Verbindung und Kameraberechtigung prüfen, andere Kamera-Apps "
                "schließen oder --camera 0 bzw. --camera 2 ausprobieren."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        print(f"Kamera {args.camera} geöffnet. Beenden mit Q oder Escape.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError("Kein Kamerabild empfangen. USB-Verbindung prüfen.")
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
