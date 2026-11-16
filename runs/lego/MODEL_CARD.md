# LEGO-Modell, erster Trainingslauf

YOLO11n Transfer Learning, 30 Epochen auf Apple M3 (MPS); Early Stopping.
38 Trainingsbilder mit 120 Objekten, 8 Validierungsbilder mit 32 Objekten.
Sitzungen getrennt, kein unabhängiger Testdatensatz.

Für die Live-Erkennung wurde last.pt ausgewählt: Bei confidence 0.5 und
IoU 0.5 auf den Validierungsbildern 30 richtige Erkennungen, 2 Fehlalarme und
2 übersehene Steine. Validierungs-F1: 0.9375. Die Zahlen sind durch Auswahl
auf demselben kleinen Validierungssatz optimistisch; keine allgemeine Trefferquote.
Die mitgelieferten Trainings-Gewichte best.pt sind nach mAP50-95 ausgewählt;
für die feste Live-Schwelle schnitt last.pt besser ab, insbesondere bei Grün.

Gelernt: Gelb/Blau/Hellgrün 2×2; Grün/Rot/Weiß 2×4.
Andere Kombinationen fehlen. Größe und Farbe sind im Datensatz gekoppelt;
eine unabhängige Größenerkennung ist damit nicht nachgewiesen.

Start: `.venv/bin/python lego_camera.py --camera 1`
Modell: `models/lego_best.pt`, Klassen-Metadaten: `models/lego_best.json`.
Untrainierte Klassen werden in der Liveansicht ausgefiltert.
