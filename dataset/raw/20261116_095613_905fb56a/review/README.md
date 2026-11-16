# Beschriftete Aufnahme vom 16.11.2026

38 Originale unter `unlabeled/` bleiben unverändert erhalten; alle sind bearbeitet.
38 beschriftete PNG-Trainingskopien liegen unter `images/`, 120 YOLO-Markierungen
unter `labels/`. Die PNG-Kopien sind horizontale Ausschnitte der Originale,
um unklare Randfragmente vollständig auszuschließen. Die YOLO-Koordinaten
beziehen sich auf die jeweiligen PNG-Kopien, nicht auf die Originalfotos.

`labeling_report.json` ordnet jede Kopie dem Original zu und enthält den
Ausschnitt, die Klassen, Originalkoordinaten und die Klassenstatistik.
`labels_overview_1.jpg` und `labels_overview_2.jpg` zeigen alle Markierungen.
Die Originale nicht zusätzlich als unbeschriftete Trainingsbilder einlesen.

Die ganze Sitzung bei der späteren Datentrennung zusammenhalten: Die Fotos
zeigen dieselben Steine in ähnlichen Positionen, eine zufällige Verteilung
auf Training und Test würde die gemessene Qualität beschönigen.
