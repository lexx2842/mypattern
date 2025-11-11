# MyPattern – Personal Health Detective

MyPattern is a personal analytics dashboard that helps you explore how daily behaviors (sleep, nutrition, activity, mood, environment) influence how you feel. It brings device sync, manual logging, data science tools, and hypothesis tracking into one coherent UI so you can run your own “n=1” health experiments.

---

## 🔍 Highlights

- **Interactive Dashboard** – Resting HR, HRV, sleep quality, mood, calories, protein, weather charts, plus auto-generated “Recent Insights”.
- **Data Sync & Imports** – Garmin Connect (optional), Open-Meteo/Ambee weather & pollen, MyNetDiary CSV for nutrition.
- **Manual Logging** – Quick sliders/inputs for mood, energy, pain, allergy, stress, sleep quality, meals, notes.
- **Insights & Analytics** – Personal baseline, anomaly feed, correlations/lag analysis for health, environment, and nutrition variables.
- **Neural Net (Random Forest)** – Train a model against any target (HRV, mood, calories, protein…) to inspect feature importance and prediction quality.
- **Hypotheses Workspace** – Track triggers/effects, confidence, and experiments to validate your own health hypotheses.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.12** (recommended/tested)
- `pip` and (optionally) virtual environment tooling (`venv`, pyenv, conda, etc.)
- Git (if cloning the repository)

### Installation

```bash
git clone https://github.com/<your-username>/mypattern.git
cd mypattern
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Initialize the database (with sample data)

```bash
python database.py
```

Running this script creates the SQLite database (`health_data.db`) and populates **12 months** of realistic sample data: Garmin-style metrics, subjective ratings, weather/pollen, and nutrition. Great for demoing and testing correlations immediately.

### Launch the app

```bash
python main.py
```

Open your browser at: **http://127.0.0.1:8080** (or `http://localhost:8080`)

---

## 🥗 Importing MyNetDiary Nutrition
Simple csv solution, which is at the moment sufficient for the purpose of a prototype

1. Export your food log from MyNetDiary as CSV (columns like Date, Meal, Calories, Protein (g), Carbs (g), Fat (g), Fiber (g), Sugar (g), Sodium (mg), Notes recommended).
2. In the UI, go to **Sync Data → MyNetDiary Nutrition**.
3. Upload the CSV. The app normalizes dates, sums macros, and stores entries in the `nutrition_intake` table.
4. Dashboard charts/insights update instantly with calories & macros per day.

If required columns (Date, Calories) are missing, the app will warn you; otherwise, you’ll see a success notification with the import count.

---

## 🗄️ Data Model Overview (SQLite)

- `garmin_data` – wearable metrics (resting HR, HRV, sleep segments, stress, steps, calories, SpO₂, etc.)
- `workouts` – workout sessions (type, duration, distance, heart rate, calories, intensity)
- `weight_data` – smart scale metrics
- `subjective_data` – mood, energy, pain, allergy symptoms, stress, sleep quality, notes
- `food_intake` – manual meal entries with tags/notes
- `environmental_data` – temperature, humidity, pollen, AQI, weather condition
- `nutrition_intake` – imported/extracted MyNetDiary calories & macros per meal
- `hypotheses` – triggers, expected effects, status, confidence, descriptions
- `experiments` – template for structured experiments (baseline, intervention, results)

Tables are auto-created via `database.py` or on first instantiation of `HealthDatabase`.

---

## 📊 Sample Data Correlations

The seeded dataset includes realistic cause/effect patterns so you can experiment immediately:

- Pasta + late eating → lower HRV the next day  
- Nuts + high pollen → higher allergy symptoms  
- Poor sleep → reduced energy the following day  
- High calorie dinners → slightly elevated resting HR and reduced HRV  
- Intense workouts → higher calories burned & better sleep duration

You can reset with fresh sample data anytime by rerunning `python database.py`.

---

## ⚙️ Tech Stack

- **Python 3.12**
- **NiceGUI** for the UI components and routing
- **Plotly** for interactive charts
- **SQLite** + `sqlite3`/`pandas` for storage and data access
- **Pandas / NumPy** for data transformations
- **scikit-learn RandomForestRegressor** as a lightweight “neural net” model
- **External APIs**: Garmin Connect (optional library), Open-Meteo (weather), Ambee (pollen)

---

## 🛣️ Roadmap Ideas

- Full Garmin OAuth workflow for live sync (current code supports local credential login)
- Deeper nutrition analytics (e.g., sodium, fiber, micronutrients)
- Scheduled anomaly detection alerts
- Advanced hypothesis testing workflows
- Data export/reporting tools
- Optional multi-user / team mode
- Contributions are welcome—open an issue or submit a PR with details and tests.

---

## 📄 License

This repository does not yet specify a license. If you plan to redistribute or modify the project, add a license file (e.g., MIT, Apache-2.0) and update this section accordingly.
