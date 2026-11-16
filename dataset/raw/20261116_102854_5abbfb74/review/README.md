# Neue Beschriftung vom 16.11.2026

38 Fotos bearbeitet. Originale unverändert unter `unlabeled/`; alle sind erledigt.
Die Trainingskopien unter `images/` sind horizontale Ausschnitte ohne unklare
Randfragmente; `labels/` enthält YOLO-Koordinaten relativ zu diesen PNG-Kopien.
Der Bericht enthält Klassen, Originalkoordinaten und den jeweiligen Ausschnitt.

Neue Klassen in den Aufnahmen: rot_2x2, hellgruen_2x4 und maennchen.
Die Figur erhält ID 12. Die bisherigen IDs 0–11 bleiben unverändert.
Die Original-Klassenliste ist als original_classes.json gesichert.

Die Sitzung wurde anschließend im zweiten Trainingslauf `runs/lego-2`
zusammen mit der vorherigen Sitzung zum Training verwendet.
Für die neuen Klassen sind weitere unabhängige Validierungs- und Testaufnahmen
nötig. Diese Sitzung nicht zufällig auf Training und Test verteilen.
