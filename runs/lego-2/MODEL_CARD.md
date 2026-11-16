# LEGO-Modell v2, 16.11.2026

YOLO11n Transfer Learning, 60 Epochen auf Apple M3 (MPS), Early Stopping aus.
76 Trainingsbilder mit 241 Objekten; 8 Validierungsbilder mit 32 Objekten.
Trainingssitzungen und Kontrollsitzung sind getrennt. Kein Testdatensatz vorhanden.

## Klassen

2×2: Gelb, Blau, Hellgrün, Rot.
2×4: Grün, Hellgrün, Rot, Weiß.
Männchen als Klasse 12; bestehende Klassen-IDs 0–11 bleiben gleich.

Fehlend: Blau/Gelb 2×4, Grün/Weiß 2×2.

## Auswahl und Prüfungen

Der letzte Trainingsstand wird für die Liveansicht verwendet. Der Kontrollsatz
enthält keine Männchen, Rot 2×2 oder Hellgrün 2×4 und kann deshalb keine
Modellauswahl für diese neuen Klassen begründen. Die alten Klassen bleiben
zusätzlich mit best.pt/last.pt verglichen; siehe live_selection.json.

Auf den alten Kontrollbildern: 31 richtige Treffer von 32 Objekten,
ein Fehlalarm und ein übersehener Stein bei confidence 0.5 und IoU 0.5.
Auf den Trainingsbildern: 241/241 Objekte erkannt, ohne zusätzliche Erkennungen.
Das ist ausschließlich eine Funktionsprüfung, keine Generalisierungsmessung.
Alle neuen Klassen wurden auf ihren Trainingsbeispielen erkannt. Frische
Kontrollaufnahmen für die neuen Klassen stehen aus.

Bei Rot und Hellgrün sind beide Größen im Training vertreten. Bei den anderen
Farben ist die Größe weiterhin an die Farbe gekoppelt. Mehr Perspektiven,
Hintergründe, Beleuchtungen und unabhängige Tests sind noch nötig.

## Verwendung

`.venv/bin/python lego_camera.py --camera 1`

Aktiv: models/lego_best.pt und models/lego_best.json.
Vorheriger Stand: models/archive/lego_v1/.
Daten: dataset/prepared/lego_v2/.
Prüfbericht: review/verification.json.
