# Pollen Data Integration

## Ambee Pollen API Setup

Die App nutzt jetzt die Ambee Pollen API für echte Pollendaten aus der Schweiz.

### API-Key erhalten

1. **Registrierung**: Gehen Sie zu [getambee.com/api/pollen](https://www.getambee.com/api/pollen)
2. **Kostenloser Plan**: 100 API-Aufrufe pro Tag, kein API-Key erforderlich für den kostenlosen Plan
3. **API-Key**: Nach der Registrierung erhalten Sie einen API-Key

### API-Key konfigurieren

**Option 1: Environment Variable (empfohlen)**
```bash
export AMBEE_API_KEY="ihr-api-key-hier"
```

**Option 2: Direkt in der Datei**
Ersetzen Sie `YOUR_AMBEE_API_KEY` in `main.py` Zeile 642 mit Ihrem echten API-Key.

### Pollenarten

Die API liefert Daten für:
- **Baumpollen** (tree_pollen): Birke, Hasel, Erle, etc.
- **Graspollen** (grass_pollen): Gräser, Roggen, etc.
- **Unkrautpollen** (weed_pollen): Beifuß, Ambrosia, etc.

### Risikostufen

- **Low** (1): Geringe Belastung
- **Moderate** (2): Moderate Belastung
- **High** (3): Hohe Belastung
- **Very High** (4): Sehr hohe Belastung
- **Extreme** (5): Extreme Belastung

### Fallback

Wenn kein API-Key gesetzt ist oder die API nicht verfügbar ist, verwendet die App automatisch Sample-Daten für Pollen.

### Testen

1. API-Key setzen
2. "Sync Weather Data" im Device Sync Tab klicken
3. Echte Pollendaten werden geladen und angezeigt

### Kosten

- **Kostenlos**: 100 Aufrufe pro Tag
- **Paid Plans**: Ab $29/Monat für mehr Aufrufe


