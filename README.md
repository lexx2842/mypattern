# MyPattern - Personal Health Detective

MyPattern ist eine persönliche Gesundheits-App, die dabei hilft, individuelle Muster und Zusammenhänge in deinen Gesundheitsdaten zu entdecken.

## Features

- **Dashboard**: Übersicht über alle wichtigen Gesundheitsmetriken
- **Manuelle Eingabe**: Subjektive Werte wie Stimmung, Energie, Schmerzen eingeben
- **Device Sync**: Synchronisation mit Garmin Connect
- **Insights**: Personal-Baseline und Anomalie-Erkennung
- **Hypothesen**: Verwaltung von Gesundheits-Hypothesen und Mini-Experimenten
- Wetter-API Integration

## Installation

1. **Virtuelle Umgebung aktivieren**:
   ```bash
   cd /Users/alexlichtenberger/Downloads/NewHealth
   source venv/bin/activate
   ```

2. **Abhängigkeiten installieren** (falls noch nicht geschehen):
   ```bash
   pip install -r requirements.txt
   ```

3. **Datenbank initialisieren** (falls noch nicht geschehen):
   ```bash
   python database.py
   ```

## Verwendung

1. **App starten**:
   ```bash
   python main.py
   ```

2. **Im Browser öffnen**: http://localhost:8080

## Datenstruktur

Die App verwendet eine SQLite-Datenbank mit folgenden Tabellen:

- `garmin_data`: Garmin-Wearable-Daten (HR, HRV, Schlaf, etc.)
- `subjective_data`: Subjektive Bewertungen (Stimmung, Energie, Schmerzen)
- `food_intake`: Nahrungsaufnahme mit Tags
- `environmental_data`: Umweltdaten (Wetter, Pollen)
- `hypotheses`: Gesundheits-Hypothesen
- `experiments`: A/B-Tests und Mini-Experimente

## Beispieldaten

Die App kommt mit 12 Monaten Beispieldaten, die realistische Korrelationen enthalten:

- **Pasta + spätes Essen** → reduzierte HRV am nächsten Morgen
- **Nüsse + hohe Pollenbelastung** → erhöhte Allergiesymptome
- **Schlechte Schlafqualität** → reduzierte Energie am nächsten Tag

## Tech Stack

- **Python 3.12**
- **NiceGUI** für die Benutzeroberfläche
- **SQLite** für die Datenspeicherung
- **Plotly** für Visualisierungen
- **Pandas** für Datenanalyse
- **GarminConnect** für Wearable-Integration (geplant)

## Brand-Farbe

Die App verwendet die Farbe `#2ECC71` (Grün) als Markenfarbe.

## Nächste Schritte

- Erweiterte Korrelationsanalyse
- A/B-Test-Framework für Hypothesen
- Export-Funktionen für Daten


