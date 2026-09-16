# Bikefit Videoanalyse

Eine Anwendung zur KI basierten Analyse der Sitzposition auf dem Rennrad.

## Architektur

Das Projekt besteht aus zwei verteilten Komponenten:

* **frontend**: Ein leichtgewichtiger FastAPI Webserver. Er bedient die Benutzeroberflaeche per HTMX und leitet Uploads an den Worker weiter.
* **worker**: Ein dedizierter Service fuer die rechenintensive YOLO11x Pose Analyse und Videoverarbeitung.

## Installation

### Worker (Maschine Zwei)

1. Wechsle in den Ordner worker
2. Installiere die Abhaengigkeiten: pip install -r requirements.txt
3. Starte den Server: uvicorn main:app --host 0.0.0.0 --port 8000

### Frontend (Maschine Eins)

1. Wechsle in den Ordner frontend
2. Passe die IP Adresse des Workers in main.py an
3. Installiere die Abhaengigkeiten: pip install -r requirements.txt
4. Starte den Webserver: uvicorn main:app --host 0.0.0.0 --port 8080

## Funktion

Nutzer laden ein Video hoch und geben ihre Koerpermasse an. Die App berechnet den Kniewinkel am unteren Totpunkt sowie die Rumpfneigung und Ellbogenwinkel. Das Ergebnis ist ein kommentiertes Video und eine konkrete Empfehlung zur Sattelverstellung in Millimetern.