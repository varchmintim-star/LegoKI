# LEGO Sorter – Webcam

## Gespeicherten Stand starten

Das Repository enthält den Code, die Aufnahmen und Labels, den vorbereiteten
Datensatz, die Trainingsergebnisse und das trainierte Modell. Die virtuelle
Python-Umgebung wird lokal neu erstellt:

```sh
git clone https://github.com/varchmintim-star/LegoKI.git
cd LegoKI
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python lego_camera.py --camera 1
```

Der erste Trainingslauf wurde unter Python 3.14 auf einem Mac mit Apple M3
ausgeführt. Die genauen Paketversionen stehen in
`runs/lego/requirements-lock.txt`. Kameraindex und macOS-Kameraberechtigung
müssen zur lokalen Installation passen. **Q** beendet die Liveansicht.

Die Pfade in historischen Trainingsprotokollen beziehen sich teilweise auf
den ursprünglichen Rechner. Die Startskripte und die Datensatzkonfiguration
verwenden den jeweils aktuellen Projektordner.

## KI-Modell trainieren und live verwenden

Training-Abhängigkeiten installieren:

```sh
.venv/bin/python -m pip install -r requirements-training.txt
```

Der aktuelle vorbereitete Datensatz unter `dataset/prepared/lego_v2/` enthält
76 Trainingsbilder mit 241 Objekten aus zwei Sitzungen sowie acht
Validierungsbilder der ältesten Sitzung. Vier ältere Bilder mit unklaren
Randfragmenten sind ausgeschlossen. Der erste Datensatz bleibt unter `lego_v1/` erhalten.
Der Aufbau erfolgt mit `prepare_dataset.py`; vorhandene vorbereitete Daten
werden dabei nicht überschrieben. Klassenverteilung und Einschränkungen stehen
in `dataset/prepared/lego_v2/manifest.json`.

```sh
.venv/bin/python train_model.py --epochs 60 --patience 0
```

Das Training passt vortrainierte YOLO11-Nano-Gewichte lokal an die LEGO-Daten an.
Es nutzt nach Verfügbarkeit die Apple-GPU (MPS), sonst CPU. Standard: 60 Epochen,
Batchgröße 4, Bildgröße 640. Farbtonänderungen als Datenaugmentation sind
deaktiviert, damit die Farblabels gültig bleiben. Ergebnisse und Gewichte landen
in einem neuen Ordner unter `runs/lego*`. Ohne neue unvalidierte Klassen wird
aus dem besten und letzten Checkpoint nach Validierungs-F1 bei confidence 0.5
und IoU 0.5 ausgewählt. Fehlen neue Klassen im Kontrollsatz, wird der letzte
vollständig trainierte Stand übernommen; der alte Kontrollsatz kann deren
Lernfortschritt nicht bewerten. `--patience 0` deaktiviert Early Stopping.
Das Live-Modell liegt unter `models/lego_best.pt`. Auswahl und Einschränkungen
stehen in `runs/lego*/live_selection.json`.

```sh
.venv/bin/python lego_camera.py --camera 1
```

**Q** oder **Escape** beendet die Live-Erkennung. `--confidence 0.5` steuert
die Anzeigeschwelle. `lego_detector.py` ist weiterhin nur der frühere
heuristische Prototyp; für das trainierte Modell `lego_camera.py` verwenden.

Der zweite Trainingslauf liegt unter `runs/lego-2/`: 60 Epochen auf Apple M3,
76 Trainingsbilder, acht Validierungsbilder. Das vorherige Live-Modell ist
unter `models/archive/lego_v1/` gesichert.

Aktuell gelernt:

- **2×2:** Gelb, Blau, Hellgrün, Rot.
- **2×4:** Grün, Hellgrün, Rot, Weiß.
- **Männchen** als eigene Klasse.

Es fehlen weiterhin Blau/Gelb als 2×4 sowie Grün/Weiß als 2×2. Die Liveansicht
filtert untrainierte Klassen über `models/lego_best.json` aus. Für Rot und
Hellgrün kommen nun beide Größen im Training vor. Bei den anderen Farben
sind Farbe und Größe weiterhin gekoppelt.

Die Funktionsprüfung auf den Trainingsbildern erkennt 241 von 241 markierten
Objekten. Dies belegt keine Generalisierung. Auf den acht alten Kontrollbildern
werden 31 von 32 Steinen korrekt erkannt, bei einem Fehlalarm (confidence 0.5,
IoU 0.5). Für Männchen, Rot 2×2 und Hellgrün 2×4 fehlen unabhängige Kontrollbilder.
Die Zahlen sind deshalb keine allgemeine Trefferquote. Details stehen unter
`runs/lego-2/review/verification.json` und `runs/lego-2/MODEL_CARD.md`.

Grundlage: [Ultralytics-Training](https://docs.ultralytics.com/modes/train/).

## Trainingsdaten für ein KI-Modell aufnehmen

```sh
.venv/bin/python collect_dataset.py --camera 1
```

13 Klassen: `gelb`, `gruen`, `hellgruen`, `weiss`, `blau`, `rot`, jeweils
mit Suffix `_2x2` oder `_2x4`, sowie `maennchen`. Grün und Hellgrün sind
getrennte Klassen. Die Figurenklasse wurde als ID 12 angehängt; die bisherigen
Steinklassen behalten ihre IDs. Ältere Sitzungen mit zwölf Klassen bleiben gültig.

Die Sitzung `20261116_102854_5abbfb74` enthält 38 weitere beschriftete Fotos
mit 121 Objekten, darunter elf Männchen sowie erstmals rote 2×2- und hellgrüne
2×4-Steine. Bericht und Kontrollübersichten liegen im Unterordner `review/`.
Diese Daten wurden zusammen mit der vorherigen Sitzung im zweiten
Trainingslauf verwendet. Für die neuen Klassen fehlen weiterhin unabhängige
Validierungsaufnahmen.

### Beschriftung durch Codex

Mit **Nur Foto speichern / P** ein Bild direkt aus der Liveansicht speichern.
Steine zwischen den Fotos neu anordnen. Die Bilder landen pro Sitzung in
`unlabeled/`, getrennt von beschrifteten Trainingsbildern. Danach Codex im Chat
bitten, die neuen Fotos zu labeln. Es läuft keine automatische Beschriftung im
Hintergrund. Unsichere Größen und stark abgeschnittene Steine müssen geprüft
werden. Alle Steine möglichst vollständig im Bild platzieren.

### Selbst beschriften

1. Steine in der später verwendeten seitlichen Kameraansicht platzieren.
2. **Bild einfrieren** anklicken oder **Leertaste** drücken.
3. Mit gedrückter linker Maustaste einen Rahmen um den gesamten sichtbaren Stein
   ziehen, einschließlich Noppen und Seitenflächen, ohne Schatten.
4. **Danach** rechts die Farbe und die Größe **2×2** oder **2×4** anklicken.
   Beide Angaben müssen für jeden neuen Steinrahmen ausgewählt werden. Bei
   einer Figur stattdessen **Männchen / M** wählen; Farbe und Größe entfallen.
5. **Bestätigen** anklicken oder **Enter** drücken. Der Rahmen wird grün.
   Dann den nächsten Rahmen ziehen und wieder Farbe und Größe zuweisen.
6. Wenn alle sichtbaren Zielsteine markiert sind, **Bild speichern** anklicken
   oder **S** drücken. Gespeichert wird das Originalbild ohne Einblendungen.
7. **Rahmen weg / Escape** verwirft den unbestätigten Rahmen, **Zurück / U**
   entfernt die letzte Markierung. **Bild verwerfen / V** kehrt ohne Speicherung
   zum Livebild zurück. **Q** beendet das Programm.

Die Kameravorschau und die anklickbaren Bedienelemente stehen nebeneinander.
Alternativ nach dem Rahmenziehen: **1–6** für Gelb, Grün, Hellgrün, Weiß, Blau,
Rot; **Z** für 2×2, **X** für 2×4; **Enter** zum Bestätigen. Tasten im
Kamerafenster drücken. Solange ein Rahmen unbestätigt ist, sind weitere Rahmen
und das Speichern gesperrt, damit keine falsche Klasse übernommen wird.

Jeder Start erzeugt eine eigene Sitzung unter `dataset/raw/`, mit `images/`,
`labels/`, `classes.json` (stabile Klassenreihenfolge) und `session.json`.
Aufnahmen werden zunächst lokal gespeichert. Dieser gesicherte Projektstand
enthält sie im Repository; neue Aufnahmen werden erst mit einem weiteren
Git-Commit und Push auf GitHub übertragen.

Als ersten Datensatz etwa 50–100 abwechslungsreiche Beispiele je Klasse sammeln;
der tatsächliche Bedarf hängt von den Ergebnissen ab. Position, Drehung und Licht
variieren; anfangs ohne Verdeckung. Weitere Sitzungen für Validierung und Test
separat aufnehmen. Nahezu identische Bilder derselben Szene später nicht auf
Training und Test verteilen. Die zwölf Klassen müssen in allen Datenteilen vorkommen.

Diese Funktion sammelt beschriftete Daten. Ein KI-Modell wird damit noch nicht
trainiert; `lego_detector.py` ist weiterhin der frühere heuristische Prototyp.

## Farbe und Steingröße erkennen

```sh
.venv/bin/python lego_detector.py --camera 1
```

1. Kamera senkrecht und fest über der Arbeitsfläche montieren. Gleichmäßig
   beleuchten und möglichst Belichtung/Weißabgleich konstant halten.
2. Eine leere, matte Fläche zeigen, kurz warten, dann im Vorschaufenster **B**
   drücken. Das speichert den Hintergrund für diese Sitzung.
3. Steine flach, mit Noppen nach oben und mit Abstand zueinander auflegen.
   Hände aus dem Bild nehmen. Farbe und vermutete Größe erscheinen am Objekt.
4. Nach Bewegung der Kamera oder Änderung des Lichts alle Steine entfernen und
   mit **B** den Hintergrund neu erfassen. **Q** oder **Escape** beendet das Programm.

Die Größen sind Schätzungen aus dem Seitenverhältnis: quadratisch → `2x2?`,
etwa doppelt so lang wie breit → `2x4?`. Es werden keine Noppen gezählt:
andere quadratische/rechteckige Teile mit diesen Proportionen können genauso
klassifiziert werden. Für den ersten Versuch nur 2×2- und 2×4-Steine verwenden.
Unpassende Proportionen erscheinen als `Unbekannt`. Berührende Steine, Schatten
und Perspektive können die Erkennung stören. Die Farbschwellen sind Startwerte
und müssen ggf. an die tatsächliche Beleuchtung angepasst werden.

Bei vielen Störungen `--threshold 50` ausprobieren; bei fehlenden Objekten
`--threshold 20`. Kleine Objekte lassen sich mit `--min-area 400` zulassen.

## Einrichtung unter macOS

Externe Webcam per USB anschließen und im Projektordner ausführen:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/cameratest.py --camera 1
```

Der Kameraindex hängt von den angeschlossenen Geräten ab. Falls die eingebaute
Kamera erscheint oder kein Gerät gefunden wird, mit `--camera 0` oder
`--camera 2` erneut starten. Index 1 ist nur ein Startwert und garantiert keine
Auswahl der externen Webcam. Andere Apps, die die Webcam verwenden, schließen.

Beim ersten Start den Kamerazugriff erlauben. Falls nötig unter
**Systemeinstellungen → Datenschutz & Sicherheit → Kamera** den Zugriff für
die ausführende Anwendung (z. B. Terminal oder Visual Studio Code) aktivieren
und die Anwendung neu starten.

Die Vorschau fordert 1280 × 720 Pixel an; die tatsächliche Auflösung hängt von
der Webcam ab. Andere Größe anfordern:

```sh
python tests/cameratest.py --camera 1 --width 1920 --height 1080
```

Mit **Q**, **Escape** oder durch Schließen des Vorschaufensters beenden.
Für die LEGO-Erkennung die Kamera fest und senkrecht über einem matten,
einfarbigen Untergrund montieren und die Fläche gleichmäßig ausleuchten.
