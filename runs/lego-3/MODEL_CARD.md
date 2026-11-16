# LEGO-Modell v3, 16.11.2026

## Training

60 zusätzliche Epochen YOLO11n auf Apple M3 (MPS), ausgehend vom gesicherten
v2-Modell, mit sämtlichen bisherigen Trainingsbildern und der neuen Figuren-Serie.
115 Trainingsbilder, 319 Objekte; 16 Kontrollbilder, 48 Objekte.
50 Figuren im Training, acht Figuren zur Kontrolle.

48 neue Fotos wurden mit 96 Objektrahmen beschriftet: Figur und Sockel getrennt.
39 Fotos zum Training, acht Ansichten links als Kontrollgruppe zurückgehalten;
ein stark unscharfes Foto ausgeschlossen. Teilfiguren am Rand aus drei Kopien
herausgeschnitten. Originale unverändert erhalten.

## Modellauswahl und Prüfung

best.pt und last.pt bei confidence 0.5 und IoU 0.5 verglichen:
best.pt: 46 richtige Treffer, ein Fehlalarm, zwei übersehene Objekte.
last.pt: 45 richtige Treffer, zwei Fehlalarme, drei übersehene Objekte.
Alle acht Figuren werden von beiden erkannt. Deshalb best.pt für die Liveansicht.

Ausgangsmodell v2 auf denselben Kontrollen: 40/48 richtige Treffer,
vier Fehlalarme, acht übersehene Objekte; nur drei der acht Figuren erkannt.

Zur Kontrolle auf mögliches Vergessen der Klassen wurden zusätzlich die
Trainingsbilder geprüft: 319/319 Objekte erkannt, keine zusätzlichen Erkennungen.
Dies ist nur eine Funktionsprüfung auf Trainingsdaten und kein Qualitätsnachweis.
Die neun trainierten Klassen waren bereits im Ausgangsmodell vorhanden; das
Weitertraining erweitert vor allem die Figurenvarianten und -ansichten.

## Grenzen

Die acht neuen Kontrollen stammen aus derselben Sitzung und zeigen dieselben
Figuren und dieselbe Anlage. Positionen wurden gruppenweise zurückgehalten;
es gibt keinen unabhängigen Test und keine belastbare allgemeine Trefferquote.
Der Kontrollsatz wurde auch zur Modellauswahl verwendet.
Rot 2×2 und Hellgrün 2×4 sind weiterhin nicht im Kontrollsatz vertreten.

Gelb/Blau 2×4 sowie Grün/Weiß 2×2 fehlen weiter vollständig im Training.

## Start

`.venv/bin/python lego_camera.py --camera 1`

Aktiv: models/lego_best.pt und models/lego_best.json.
Vorheriger Stand: models/archive/lego_v2/.
Daten: dataset/prepared/lego_v3/.
Ergebnisse: review/verification.json.
Sichtprüfung: review/figures_validation.jpg.
