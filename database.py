import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
from typing import Optional, List, Dict, Any

class HealthDatabase:
    def __init__(self, db_path: str = "health_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Garmin data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS garmin_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                resting_hr INTEGER,
                hrv REAL,
                spo2 REAL,
                sleep_duration REAL,
                sleep_efficiency REAL,
                deep_sleep REAL,
                light_sleep REAL,
                rem_sleep REAL,
                awake_time REAL,
                body_battery REAL,
                stress_level REAL,
                steps INTEGER,
                calories INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Workouts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                activity_type TEXT,
                duration_minutes INTEGER,
                distance_km REAL,
                avg_hr INTEGER,
                max_hr INTEGER,
                calories_burned INTEGER,
                intensity_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Weight table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weight_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                weight_kg REAL,
                body_fat_percent REAL,
                muscle_mass_kg REAL,
                water_percent REAL,
                bone_mass_kg REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Subjective data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subjective_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                mood INTEGER CHECK(mood >= 1 AND mood <= 10),
                energy_level INTEGER CHECK(energy_level >= 1 AND energy_level <= 10),
                pain_level INTEGER CHECK(pain_level >= 0 AND pain_level <= 10),
                allergy_symptoms INTEGER CHECK(allergy_symptoms >= 0 AND allergy_symptoms <= 10),
                stress_level INTEGER CHECK(stress_level >= 1 AND mood <= 10),
                sleep_quality INTEGER CHECK(sleep_quality >= 1 AND sleep_quality <= 10),
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Food intake table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_intake (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                meal_type TEXT CHECK(meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
                food_items TEXT,
                tags TEXT,  -- JSON string with tags like ['pasta', 'nuts', 'gluten']
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Environmental data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environmental_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                location TEXT DEFAULT 'Zurich',
                temperature REAL,
                humidity REAL,
                pollen_hazelnut INTEGER,
                pollen_birch INTEGER,
                pollen_grass INTEGER,
                air_quality_index INTEGER,
                weather_condition TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Nutrition intake table (MyNetDiary CSV imports)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nutrition_intake (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                meal_type TEXT,
                calories REAL,
                protein_g REAL,
                carbs_g REAL,
                fat_g REAL,
                fiber_g REAL,
                sugar_g REAL,
                sodium_mg REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Hypotheses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                trigger_condition TEXT,  -- e.g., "pasta + late_eating"
                expected_effect TEXT,    -- e.g., "low_hrv_next_morning"
                status TEXT CHECK(status IN ('active', 'testing', 'confirmed', 'rejected')) DEFAULT 'active',
                confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER,
                experiment_type TEXT CHECK(experiment_type IN ('A/B', 'intervention', 'elimination')),
                start_date DATE,
                end_date DATE,
                intervention_description TEXT,
                baseline_metrics TEXT,  -- JSON string
                results TEXT,           -- JSON string
                status TEXT CHECK(status IN ('planned', 'active', 'completed')) DEFAULT 'planned',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def populate_sample_data(self):
        """Populate database with 12 months of sample data including correlations"""
        conn = sqlite3.connect(self.db_path)
        
        # Generate 12 months of data
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=i) for i in range(365)]
        
        # Sample data generation with correlations
        sample_data = []
        
        for i, date in enumerate(dates):
            # Base values
            base_hr = 45
            base_hrv = 45
            base_mood = 7
            base_energy = 7
            
            # Seasonal patterns
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 0.1 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Pollen season (spring)
            pollen_factor = 0
            if 60 <= day_of_year <= 150:  # March to May
                pollen_factor = 0.3 * np.sin(np.pi * (day_of_year - 60) / 90)
            
            # Random pasta consumption (affects HRV next day)
            pasta_today = random.random() < 0.15  # 15% chance
            pasta_yesterday = i > 0 and sample_data[i-1].get('pasta_consumed', False)
            
            # Nuts consumption (affects allergy symptoms)
            nuts_today = random.random() < 0.2  # 20% chance
            
            # Late eating
            late_eating = random.random() < 0.25  # 25% chance
            
            # Calculate correlated values
            hr = base_hr + random.gauss(0, 3) + seasonal_factor * 2
            hrv = base_hrv + random.gauss(0, 5) - (pasta_yesterday * 8) - (pollen_factor * 5)
            mood = max(1, min(10, base_mood + random.gauss(0, 1.5) - (pollen_factor * 2)))
            energy = max(1, min(10, base_energy + random.gauss(0, 1.5) - (pasta_yesterday * 1.5)))
            allergy_symptoms = max(0, min(10, pollen_factor * 3 + (nuts_today * 2) + random.gauss(0, 1)))
            
            # Garmin data
            garmin_data = {
                'date': date.strftime('%Y-%m-%d'),
                'resting_hr': int(hr),
                'hrv': round(hrv, 1),
                'spo2': round(98 + random.gauss(0, 0.5), 1),
                'sleep_duration': round(7.5 + random.gauss(0, 1), 1),
                'sleep_efficiency': round(85 + random.gauss(0, 5), 1),
                'deep_sleep': round(1.5 + random.gauss(0, 0.3), 1),
                'light_sleep': round(4.5 + random.gauss(0, 0.5), 1),
                'rem_sleep': round(1.5 + random.gauss(0, 0.3), 1),
                'awake_time': round(0.2 + random.gauss(0, 0.1), 1),
                'body_battery': round(75 + random.gauss(0, 10), 0),
                'stress_level': round(25 + random.gauss(0, 8), 0),
                'steps': int(8000 + random.gauss(0, 2000)),
                'calories': int(2000 + random.gauss(0, 300))
            }
            
            # Workout data (3-4 workouts per week)
            workout_data = None
            if random.random() < 0.5:  # 50% chance of workout
                activity_types = ['Running', 'Cycling', 'Strength Training', 'Yoga', 'Swimming']
                activity_type = random.choice(activity_types)
                
                # Workout intensity affects HRV next day
                intensity = random.uniform(0.3, 1.0)
                duration = int(30 + random.gauss(0, 20) * intensity)
                
                workout_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'activity_type': activity_type,
                    'duration_minutes': max(15, duration),
                    'distance_km': round(random.uniform(2, 15) if activity_type in ['Running', 'Cycling'] else 0, 1),
                    'avg_hr': int(120 + random.gauss(0, 20) * intensity),
                    'max_hr': int(150 + random.gauss(0, 15) * intensity),
                    'calories_burned': int(200 + random.gauss(0, 100) * intensity),
                    'intensity_score': round(intensity, 2)
                }
            
            # Weight data (measured 2-3 times per week)
            weight_data = None
            if random.random() < 0.3:  # 30% chance of weight measurement
                # Gradual weight change over time with some fluctuation
                base_weight = 75.0
                trend = (i / 365) * 2  # 2kg change over year
                daily_fluctuation = random.gauss(0, 0.3)
                
                weight_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'weight_kg': round(base_weight + trend + daily_fluctuation, 1),
                    'body_fat_percent': round(15 + random.gauss(0, 2), 1),
                    'muscle_mass_kg': round(35 + random.gauss(0, 1), 1),
                    'water_percent': round(60 + random.gauss(0, 2), 1),
                    'bone_mass_kg': round(3.2 + random.gauss(0, 0.1), 1)
                }
            
            # Subjective data
            subjective_data = {
                'date': date.strftime('%Y-%m-%d'),
                'mood': int(mood),
                'energy_level': int(energy),
                'pain_level': int(max(0, min(10, random.gauss(2, 1.5) + (pasta_yesterday * 1)))),
                'allergy_symptoms': int(allergy_symptoms),
                'stress_level': int(max(1, min(10, 5 + random.gauss(0, 1.5) + (late_eating * 1.5)))),
                'sleep_quality': int(max(1, min(10, 7 + random.gauss(0, 1.5) - (late_eating * 1.5)))),
                'notes': ''
            }
            
            # Food data
            food_items = []
            tags = []
            if pasta_today:
                food_items.append('Pasta')
                tags.append('pasta')
                tags.append('gluten')
            if nuts_today:
                food_items.append('Nuts')
                tags.append('nuts')
            
            food_data = {
                'date': date.strftime('%Y-%m-%d'),
                'meal_type': 'dinner' if pasta_today else 'lunch',
                'food_items': ', '.join(food_items) if food_items else 'Normal meal',
                'tags': str(tags),
                'notes': 'Late eating' if late_eating else ''
            }
            
            # Nutrition intake (aggregate macros, distribute across meals)
            total_calories = 2100 + random.gauss(0, 250)
            if pasta_today:
                total_calories += 250
            if late_eating:
                total_calories += 150
            total_protein = max(60, 100 + random.gauss(0, 15) - (late_eating * 5))
            total_carbs = max(150, 230 + random.gauss(0, 30) + (pasta_today * 40))
            total_fat = max(50, 70 + random.gauss(0, 10))
            total_fiber = max(15, 25 + random.gauss(0, 5))
            total_sugar = max(20, 35 + random.gauss(0, 8) + (nuts_today * 5))
            total_sodium = max(1000, 1800 + random.gauss(0, 250))

            meal_ratios = {
                'breakfast': 0.25,
                'lunch': 0.35,
                'dinner': 0.3 + (0.1 if late_eating else 0),
                'snack': 0.1 + (0.05 if nuts_today else 0)
            }
            ratio_sum = sum(meal_ratios.values())
            for meal, ratio in meal_ratios.items():
                share = ratio / ratio_sum
                nutrition_row = {
                    'date': date.strftime('%Y-%m-%d'),
                    'meal_type': meal,
                    'calories': round(total_calories * share, 1),
                    'protein_g': round(total_protein * share, 1),
                    'carbs_g': round(total_carbs * share, 1),
                    'fat_g': round(total_fat * share, 1),
                    'fiber_g': round(total_fiber * share, 1),
                    'sugar_g': round(total_sugar * share, 1),
                    'sodium_mg': round(total_sodium * share, 1),
                    'notes': 'Late meal' if (meal == 'dinner' and late_eating) else ('Nuts snack' if (meal == 'snack' and nuts_today) else '')
                }
                self._insert_nutrition_row(conn, nutrition_row)

            # Environmental data
            env_data = {
                'date': date.strftime('%Y-%m-%d'),
                'location': 'Zurich',
                'temperature': round(15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + random.gauss(0, 3), 1),
                'humidity': round(60 + random.gauss(0, 15), 0),
                'pollen_hazelnut': int(pollen_factor * 100),
                'pollen_birch': int(pollen_factor * 80),
                'pollen_grass': int(pollen_factor * 60),
                'air_quality_index': int(50 + random.gauss(0, 15)),
                'weather_condition': random.choice(['sunny', 'cloudy', 'rainy', 'partly_cloudy'])
            }
            
            # Store data
            sample_data.append({
                'pasta_consumed': pasta_today,
                'nuts_consumed': nuts_today,
                'late_eating': late_eating
            })
            
            # Insert into database
            self._insert_garmin_data(conn, garmin_data)
            self._insert_subjective_data(conn, subjective_data)
            self._insert_food_data(conn, food_data)
            self._insert_environmental_data(conn, env_data)
            
            # Insert workout and weight data if available
            if workout_data:
                self._insert_workout_data(conn, workout_data)
            if weight_data:
                self._insert_weight_data(conn, weight_data)
        
        # Add some sample hypotheses
        hypotheses = [
            {
                'title': 'Pasta + Late Eating → Low HRV',
                'description': 'Eating pasta late in the evening leads to reduced HRV the next morning',
                'trigger_condition': 'pasta + late_eating',
                'expected_effect': 'low_hrv_next_morning',
                'status': 'active',
                'confidence_score': 0.7
            },
            {
                'title': 'Nuts + Pollen → Allergy Symptoms',
                'description': 'Consuming nuts during high pollen season increases allergy symptoms',
                'trigger_condition': 'nuts + high_pollen',
                'expected_effect': 'increased_allergy_symptoms',
                'status': 'active',
                'confidence_score': 0.8
            },
            {
                'title': 'Poor Sleep → Low Energy',
                'description': 'Poor sleep quality leads to reduced energy levels the next day',
                'trigger_condition': 'poor_sleep_quality',
                'expected_effect': 'low_energy_next_day',
                'status': 'confirmed',
                'confidence_score': 0.9
            }
        ]
        
        for hyp in hypotheses:
            self._insert_hypothesis(conn, hyp)
        
        conn.commit()
        conn.close()
        print("Sample data populated successfully!")
    
    def _insert_garmin_data(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO garmin_data (date, resting_hr, hrv, spo2, sleep_duration, sleep_efficiency, 
                                   deep_sleep, light_sleep, rem_sleep, awake_time, body_battery, 
                                   stress_level, steps, calories)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['resting_hr'], data['hrv'], data['spo2'], 
              data['sleep_duration'], data['sleep_efficiency'], data['deep_sleep'], 
              data['light_sleep'], data['rem_sleep'], data['awake_time'], 
              data['body_battery'], data['stress_level'], data['steps'], data['calories']))
    
    def _insert_subjective_data(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO subjective_data (date, mood, energy_level, pain_level, allergy_symptoms, 
                                       stress_level, sleep_quality, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['mood'], data['energy_level'], data['pain_level'], 
              data['allergy_symptoms'], data['stress_level'], data['sleep_quality'], data['notes']))
    
    def _insert_food_data(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO food_intake (date, meal_type, food_items, tags, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (data['date'], data['meal_type'], data['food_items'], data['tags'], data['notes']))
    
    def _insert_environmental_data(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO environmental_data (date, location, temperature, humidity, pollen_hazelnut, 
                                          pollen_birch, pollen_grass, air_quality_index, weather_condition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['location'], data['temperature'], data['humidity'], 
              data['pollen_hazelnut'], data['pollen_birch'], data['pollen_grass'], 
              data['air_quality_index'], data['weather_condition']))
    
    def _insert_nutrition_row(self, conn, row: dict):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO nutrition_intake
            (date, meal_type, calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row.get('date'),
            row.get('meal_type'),
            row.get('calories'),
            row.get('protein_g'),
            row.get('carbs_g'),
            row.get('fat_g'),
            row.get('fiber_g'),
            row.get('sugar_g'),
            row.get('sodium_mg'),
            row.get('notes') or ''
        ))
    
    def _insert_workout_data(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO workouts (date, activity_type, duration_minutes, distance_km, avg_hr, 
                                max_hr, calories_burned, intensity_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['activity_type'], data['duration_minutes'], data['distance_km'], 
              data['avg_hr'], data['max_hr'], data['calories_burned'], data['intensity_score']))
    
    def _insert_weight_data(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO weight_data (date, weight_kg, body_fat_percent, muscle_mass_kg, 
                                   water_percent, bone_mass_kg)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['weight_kg'], data['body_fat_percent'], data['muscle_mass_kg'], 
              data['water_percent'], data['bone_mass_kg']))
    
    def _insert_hypothesis(self, conn, data):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO hypotheses (title, description, trigger_condition, expected_effect, 
                                  status, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (data['title'], data['description'], data['trigger_condition'], 
              data['expected_effect'], data['status'], data['confidence_score']))
    
    def get_dashboard_data(self, days: int = 30) -> Dict[str, Any]:
        """Get data for dashboard visualization"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Garmin data
        garmin_df = pd.read_sql_query('''
            SELECT * FROM garmin_data 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        # Subjective data
        subjective_df = pd.read_sql_query('''
            SELECT * FROM subjective_data 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        # Environmental data
        env_df = pd.read_sql_query('''
            SELECT * FROM environmental_data 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        # Food data
        food_df = pd.read_sql_query('''
            SELECT * FROM food_intake 
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        # Nutrition data
        nutrition_df = pd.read_sql_query('''
            SELECT * FROM nutrition_intake
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', conn, params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        # Hypotheses
        hypotheses_df = pd.read_sql_query('''
            SELECT * FROM hypotheses 
            ORDER BY confidence_score DESC
        ''', conn)
        
        conn.close()
        
        return {
            'garmin': garmin_df,
            'subjective': subjective_df,
            'environmental': env_df,
            'food': food_df,
            'hypotheses': hypotheses_df,
            'nutrition': nutrition_df,
        }
    
    def insert_manual_data(self, date: str, data_type: str, data: Dict[str, Any]):
        """Insert manually entered data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_type == 'subjective':
            cursor.execute('''
                INSERT OR REPLACE INTO subjective_data 
                (date, mood, energy_level, pain_level, allergy_symptoms, stress_level, sleep_quality, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, data.get('mood'), data.get('energy_level'), data.get('pain_level'),
                  data.get('allergy_symptoms'), data.get('stress_level'), data.get('sleep_quality'), data.get('notes')))
        
        elif data_type == 'food':
            cursor.execute('''
                INSERT INTO food_intake (date, meal_type, food_items, tags, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (date, data.get('meal_type'), data.get('food_items'), data.get('tags'), data.get('notes')))
        
        conn.commit()
        conn.close()
    
    def get_data_for_date(self, date: str) -> Dict[str, Any]:
        """Get all data for a specific date"""
        conn = sqlite3.connect(self.db_path)
        
        garmin = pd.read_sql_query('SELECT * FROM garmin_data WHERE date = ?', conn, params=[date])
        subjective = pd.read_sql_query('SELECT * FROM subjective_data WHERE date = ?', conn, params=[date])
        food = pd.read_sql_query('SELECT * FROM food_intake WHERE date = ?', conn, params=[date])
        env = pd.read_sql_query('SELECT * FROM environmental_data WHERE date = ?', conn, params=[date])
        
        conn.close()
        
        return {
            'garmin': garmin.to_dict('records')[0] if not garmin.empty else {},
            'subjective': subjective.to_dict('records')[0] if not subjective.empty else {},
            'food': food.to_dict('records') if not food.empty else [],
            'environmental': env.to_dict('records')[0] if not env.empty else {}
        }
    
    def insert_environmental_data(self, data: Dict[str, Any]):
        """Insert environmental data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO environmental_data 
            (date, location, temperature, humidity, pollen_hazelnut, pollen_birch, 
             pollen_grass, air_quality_index, weather_condition)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data['date'], data['location'], data['temperature'], data['humidity'],
              data['pollen_hazelnut'], data['pollen_birch'], data['pollen_grass'],
              data['air_quality_index'], data['weather_condition']))
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection for external use"""
        return sqlite3.connect(self.db_path)

if __name__ == "__main__":
    db = HealthDatabase()
    db.populate_sample_data()
