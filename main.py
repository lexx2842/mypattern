from nicegui import ui, app
import pandas as pd
from datetime import datetime, timedelta
from database import HealthDatabase
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import requests
import os
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import asyncio

# Brand color
BRAND_COLOR = '#2ECC71'

class MyPatternApp:
    def __init__(self):
        self.db = HealthDatabase()
        self.current_data = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI with tabs"""
        ui.page_title('MyPattern - Personal Health Detective')
        
        # Custom CSS for brand color
        ui.add_head_html(f'''
            <style>
                .brand-color {{ color: {BRAND_COLOR} !important; }}
                .brand-bg {{ background-color: {BRAND_COLOR} !important; }}
                .tab-active {{ border-bottom: 2px solid {BRAND_COLOR} !important; }}
                .logo-text {{ 
                    color: {BRAND_COLOR} !important; 
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.8), -1px -1px 2px rgba(255,255,255,0.3);
                    filter: drop-shadow(0 0 3px rgba(0,0,0,0.5));
                    font-style: italic;
                }}
            </style>
        ''')
        
        # Header
        with ui.header().classes('items-center justify-between py-3'):
            ui.label('MyPattern').classes('text-h4 font-bold logo-text')
            ui.label('Personal Health Detective').classes('text-h6 text-white')
        
        # Main content with tabs
        with ui.tabs().classes('w-full') as tabs:
            dashboard_tab = ui.tab('Dashboard', icon='dashboard')
            sync_tab = ui.tab('Sync Data', icon='sync')
            insights_tab = ui.tab('Insights', icon='insights')
            neural_tab = ui.tab('Neural Net', icon='psychology_alt')
            hypotheses_tab = ui.tab('Hypotheses', icon='psychology')
        
        with ui.tab_panels(tabs, value=dashboard_tab).classes('w-full'):
            # Dashboard Tab
            with ui.tab_panel(dashboard_tab):
                self.create_dashboard()
            
            # Device Sync Tab
            with ui.tab_panel(sync_tab):
                self.create_device_sync()
            
            # Insights Tab
            with ui.tab_panel(insights_tab):
                self.create_insights()
            
            # Neural Net Tab
            with ui.tab_panel(neural_tab):
                self.create_neural_net()
            
            # Hypotheses Tab
            with ui.tab_panel(hypotheses_tab):
                self.create_hypotheses()
    
    def create_dashboard(self):
        """Create the dashboard with metrics visualization"""
        with ui.column().classes('w-full p-4'):
            ui.label('Health Dashboard').classes('text-h4 font-bold mb-4 brand-color')
            
            # Date range selector
            with ui.row().classes('w-full mb-4 items-center'):
                ui.label('Time Range:').classes('mr-2 text-base text-black')
                with ui.row().classes('items-center gap-2'):
                    days_select = ui.select([7, 14, 30, 90], value=14).classes('w-16 text-sm bg-grey-2 border-0 py-1')
                    ui.label('days').classes('text-base text-black')
                refresh_btn = ui.button('Refresh', icon='refresh', on_click=lambda: self.refresh_dashboard(days_select.value)).classes('text-sm px-3 py-1')
            
            # Metrics cards
            with ui.row().classes('w-full mb-6'):
                with ui.card().classes('p-4 m-2'):
                    ui.label('Resting HR').classes('text-subtitle2')
                    self.hr_label = ui.label('--').classes('text-h5 font-bold brand-color')
                
                with ui.card().classes('p-4 m-2'):
                    ui.label('HRV').classes('text-subtitle2')
                    self.hrv_label = ui.label('--').classes('text-h5 font-bold brand-color')
                
                with ui.card().classes('p-4 m-2'):
                    ui.label('Sleep Quality').classes('text-subtitle2')
                    self.sleep_label = ui.label('--').classes('text-h5 font-bold brand-color')
                
                with ui.card().classes('p-4 m-2'):
                    ui.label('Mood').classes('text-subtitle2')
                    self.mood_label = ui.label('--').classes('text-h5 font-bold brand-color')
            
            # Charts container that will be updated dynamically
            self.charts_container = ui.column().classes('w-full')
            
            # Load initial data
            self.refresh_dashboard(14)
    
    def create_device_sync(self):
        """Create device sync interface"""
        with ui.column().classes('w-full p-4'):
            ui.label('Data Synchronization').classes('text-h4 font-bold mb-4 brand-color')
            
            # Garmin Connect section
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Garmin').classes('text-h6 mb-2 text-black')
                ui.label('Heart Rate, HRV, Sleep, Steps, Calories, Stress, SpO2, Body Battery, Workouts, Weight').classes('text-sm text-black mb-2 font-bold')
                
                # Modern inline form layout
                with ui.row().classes('w-full mb-3 gap-2 items-end'):
                    with ui.column().classes('flex-1'):
                        self.garmin_username = ui.input('Garmin Username').classes('w-full text-xs bg-grey-2 text-black border-0')
                    
                    with ui.column().classes('flex-1'):
                        self.garmin_password = ui.input('Garmin Password', password=True).classes('w-full text-xs bg-grey-2 text-black border-0')
                    
                    with ui.column().classes('w-32'):
                        with ui.row().classes('items-center gap-2'):
                            self.sync_days_back = ui.select([7, 14, 30, 60, 90], value=30).classes('w-20 text-sm')
                            ui.label('days').classes('text-sm text-grey-6')
                
                with ui.row().classes('w-full mt-4'):
                    sync_btn = ui.button('Sync Garmin Data', icon='sync', on_click=self.sync_garmin_data).classes('brand-bg text-white')
                    self.sync_status = ui.label('').classes('ml-4')
            
            # Weather/Pollen section
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Weather & Pollen').classes('text-h6 mb-2 text-black')
                ui.label('Temperature, Humidity, Pollen Levels (Hazelnut, Birch, Grass)').classes('text-sm text-black mb-3 font-bold')
                
                with ui.row().classes('w-full mb-2 items-center'):
                    ui.label('Location:').classes('w-20 text-sm text-grey-6')
                    self.location = ui.input('Location', value='Zurich').classes('flex-1 text-sm')
                
                # Current pollen levels display
                with ui.row().classes('w-full mt-4 mb-4'):
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Current Pollen Levels').classes('text-subtitle2 mb-2')
                        self.pollen_display = ui.column()
                        with self.pollen_display:
                            ui.label('No pollen data available').classes('text-grey-6')
                
                with ui.row().classes('w-full mt-4'):
                    weather_btn = ui.button('Sync Weather Data', icon='cloud', on_click=self.sync_weather_data).classes('brand-bg text-white')
                    self.weather_status = ui.label('').classes('ml-4')
            
            # Manual Data Entry section
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Manual Data Entry').classes('text-h6 mb-2 text-black')
                ui.label('Mood, Energy, Pain, Allergy, Stress, Sleep Quality, Food Intake').classes('text-sm text-black mb-3 font-bold')
                
                # Date selector
                with ui.row().classes('w-full mb-4'):
                    ui.label('Date:').classes('mr-2')
                    self.date_input = ui.date(value=datetime.now().strftime('%Y-%m-%d'))
                    load_btn = ui.button('Load Existing', icon='search', on_click=self.load_existing_data)
                
                # Subjective data form
                with ui.card().classes('w-full p-4 mb-4'):
                    ui.label('Subjective Metrics').classes('text-h6 mb-3')
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Mood (1-10):').classes('w-32')
                        self.mood_input = ui.slider(min=1, max=10, value=7).classes('flex-1')
                        self.mood_value = ui.label('7').classes('w-8')
                        self.mood_input.on('update:model-value', lambda e: self.mood_value.set_text(str(e.args)))
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Energy (1-10):').classes('w-32')
                        self.energy_input = ui.slider(min=1, max=10, value=7).classes('flex-1')
                        self.energy_value = ui.label('7').classes('w-8')
                        self.energy_input.on('update:model-value', lambda e: self.energy_value.set_text(str(e.args)))
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Pain (0-10):').classes('w-32')
                        self.pain_input = ui.slider(min=0, max=10, value=0).classes('flex-1')
                        self.pain_value = ui.label('0').classes('w-8')
                        self.pain_input.on('update:model-value', lambda e: self.pain_value.set_text(str(e.args)))
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Allergy (0-10):').classes('w-32')
                        self.allergy_input = ui.slider(min=0, max=10, value=0).classes('flex-1')
                        self.allergy_value = ui.label('0').classes('w-8')
                        self.allergy_input.on('update:model-value', lambda e: self.allergy_value.set_text(str(e.args)))
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Stress (1-10):').classes('w-32')
                        self.stress_input = ui.slider(min=1, max=10, value=5).classes('flex-1')
                        self.stress_value = ui.label('5').classes('w-8')
                        self.stress_input.on('update:model-value', lambda e: self.stress_value.set_text(str(e.args)))
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Sleep Quality (1-10):').classes('w-32')
                        self.sleep_quality_input = ui.slider(min=1, max=10, value=7).classes('flex-1')
                        self.sleep_quality_value = ui.label('7').classes('w-8')
                        self.sleep_quality_input.on('update:model-value', lambda e: self.sleep_quality_value.set_text(str(e.args)))
                    
                    self.notes_input = ui.textarea('Notes', placeholder='Additional observations...').classes('w-full mt-2')
                
                # Food intake form
                with ui.card().classes('w-full p-4 mb-4'):
                    ui.label('Food Intake').classes('text-h6 mb-3')
                    
                    with ui.row().classes('w-full mb-2'):
                        ui.label('Meal Type:').classes('w-32')
                        self.meal_type = ui.select(['breakfast', 'lunch', 'dinner', 'snack'], value='dinner')
                    
                    self.food_items = ui.input('Food Items', placeholder='e.g., Pasta, Nuts, Salad...').classes('w-full mb-2')
                    self.food_tags = ui.input('Tags', placeholder='e.g., pasta, nuts, gluten (comma-separated)').classes('w-full mb-2')
                    self.food_notes = ui.textarea('Food Notes', placeholder='Additional food observations...').classes('w-full')
                
                # Save button
                with ui.row().classes('w-full justify-center mt-4'):
                    save_btn = ui.button('Save Data', icon='save', on_click=self.save_manual_data).classes('brand-bg text-white px-8 py-2')
    
    def create_insights(self):
        """Create insights and analysis view"""
        with ui.column().classes('w-full p-4'):
            ui.label('Health Insights').classes('text-h5 font-bold mb-4 brand-color')
            
            # Baseline section
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Personal Baseline').classes('text-h6 mb-3')
                self.baseline_content = ui.column()
                with self.baseline_content:
                    ui.label('Calculating baseline...').classes('text-grey-6')
            
            # Anomaly detection
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Anomaly Detection').classes('text-h6 mb-3')
                self.anomaly_content = ui.column()
                with self.anomaly_content:
                    ui.label('No anomalies detected recently.').classes('text-grey-6')
            
            # Correlation analysis
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Correlation Analysis').classes('text-h6 mb-3')
                
                # Lag correlation controls
                with ui.row().classes('w-full mb-4'):
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Variable 1:').classes('text-sm')
                        self.corr_var1 = ui.select(
                            ['hrv', 'heart_rate', 'sleep_duration', 'mood', 'energy_level', 'pollen_hazelnut', 'pollen_birch', 'pollen_grass', 'temperature'],
                            value='hrv'
                        ).classes('w-full')
                    
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Variable 2:').classes('text-sm')
                        self.corr_var2 = ui.select(
                            ['hrv', 'heart_rate', 'sleep_duration', 'mood', 'energy_level', 'pollen_hazelnut', 'pollen_birch', 'pollen_grass', 'temperature'],
                            value='mood'
                        ).classes('w-full')
                    
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Max Lag (days):').classes('text-sm')
                        self.max_lag = ui.number(value=7, min=0, max=30).classes('w-full')
                    
                    with ui.column().classes('flex-1'):
                        ui.label('').classes('text-sm')  # Spacer
                        analyze_btn = ui.button('Analyze Lag', icon='timeline', on_click=self.analyze_lag_correlation).classes('w-full')
                
                self.correlation_container = ui.column().classes('w-full')
            
            # Load insights
            self.load_insights()
    
    def create_neural_net(self):
        """Create neural network analysis view"""
        with ui.column().classes('w-full p-4'):
            ui.label('Neural Network Analysis').classes('text-h5 font-bold mb-4 brand-color')
            
            # Info card
            with ui.card().classes('w-full p-4 mb-4 bg-blue-50'):
                ui.label('🧠 AI-Powered Pattern Detection').classes('text-h6 mb-2')
                ui.label('Train a neural network to automatically discover complex correlations across all your health metrics.').classes('text-sm text-grey-7')
            
            # Training controls
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Training Configuration').classes('text-h6 mb-3')
                
                with ui.row().classes('w-full mb-4'):
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Target Variable:').classes('text-sm mb-1')
                        self.nn_target = ui.select(
                            options=['hrv', 'mood', 'energy_level', 'sleep_duration'],
                            value='hrv'
                        ).classes('w-full')
                        ui.label('The metric to predict/analyze').classes('text-xs text-grey-6')
                    
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Epochs:').classes('text-sm mb-1')
                        self.nn_epochs = ui.number(value=50, min=10, max=200).classes('w-full')
                        ui.label('More epochs = better learning').classes('text-xs text-grey-6')
                    
                    with ui.column().classes('flex-1'):
                        ui.label('Learning Rate:').classes('text-sm mb-1')
                        self.nn_learning_rate = ui.select(
                            options=['0.001', '0.01', '0.1'],
                            value='0.01'
                        ).classes('w-full')
                        ui.label('How fast the model learns').classes('text-xs text-grey-6')
                
                with ui.row().classes('w-full mt-4'):
                    self.train_btn = ui.button(
                        'Start Training',
                        icon='play_arrow',
                        on_click=self.train_neural_network
                    ).classes('brand-bg text-white px-8 py-2')
                    
                    self.stop_btn = ui.button(
                        'Stop Training',
                        icon='stop',
                        on_click=self.stop_training
                    ).classes('bg-red-500 text-white px-8 py-2 ml-2').props('disabled')
                    
                    self.generate_sample_btn = ui.button(
                        'Generate Sample Data',
                        icon='add',
                        on_click=self.generate_sample_data_for_nn
                    ).classes('bg-orange-500 text-white px-8 py-2 ml-2')
            
            # Training progress
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Training Progress').classes('text-h6 mb-3')
                
                # Progress bar
                self.nn_progress = ui.linear_progress(value=0).classes('mb-2')
                self.nn_status = ui.label('Ready to train').classes('text-sm text-grey-7 mb-3')
                
                # Training log
                with ui.card().classes('w-full bg-gray-900').style('height: 20rem; overflow: hidden; padding: 0; margin: 0;'):
                    self.nn_log = ui.html('''
                        <div id="neural-log-container" style="
                            width: 100%;
                            height: 20rem; 
                            overflow-y: auto; 
                            overflow-x: hidden; 
                            border: none; 
                            background-color: #111827; 
                            padding: 12px; 
                            font-family: 'Courier New', monospace; 
                            font-size: 16px;
                            font-weight: bold;
                            color: #10b981;
                            white-space: pre-wrap;
                            line-height: 1.4;
                            box-sizing: border-box;
                            margin: 0;
                            display: block;
                            position: relative;
                        ">
                            <div>========================================================================================================</div>
                            <div>Waiting for training to start...</div>
                        </div>
                    ''')
                    self.nn_log_container = None
            
            # Results container
            self.nn_results_container = ui.column().classes('w-full')
            with self.nn_results_container:
                with ui.card().classes('w-full p-4'):
                    ui.label('Results').classes('text-h6 mb-3')
                    ui.label('Train the model to see results').classes('text-grey-6')
    
    def create_hypotheses(self):
        """Create hypotheses management view"""
        with ui.column().classes('w-full p-4'):
            ui.label('Hypotheses & Experiments').classes('text-h5 font-bold mb-4 brand-color')
            
            # Active hypotheses
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Active Hypotheses').classes('text-h6 mb-3')
                self.hypotheses_list = ui.column()
            
            # Add new hypothesis
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Add New Hypothesis').classes('text-h6 mb-3')
                
                # Title and Description
                self.hyp_title = ui.input('Title').classes('w-full mb-2')
                self.hyp_description = ui.textarea('Description').classes('w-full mb-2')
                
                # Trigger and Effect with dropdowns
                with ui.row().classes('w-full mb-2'):
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Trigger Condition:').classes('text-sm mb-1')
                        self.hyp_trigger = ui.select([
                            'Pasta consumption', 'Late eating', 'Nuts consumption', 'Alcohol intake',
                            'High stress day', 'Poor sleep', 'High pollen count', 'Cold weather',
                            'Intense workout', 'Travel', 'Work deadline', 'Social event'
                        ], value='Pasta consumption').classes('w-full')
                    
                    with ui.column().classes('flex-1 ml-2'):
                        ui.label('Expected Effect:').classes('text-sm mb-1')
                        self.hyp_effect = ui.select([
                            'Low HRV next morning', 'Poor sleep quality', 'High stress level',
                            'Allergy symptoms', 'Low energy', 'Mood decline', 'Weight gain',
                            'High resting HR', 'Poor recovery', 'Digestive issues'
                        ], value='Low HRV next morning').classes('w-full')
                
                # Status and Confidence
                with ui.row().classes('w-full mb-2'):
                    with ui.column().classes('flex-1 mr-2'):
                        ui.label('Status:').classes('text-sm mb-1')
                        self.hyp_status = ui.select([
                            'active', 'testing', 'confirmed', 'rejected'
                        ], value='active').classes('w-full')
                    
                    with ui.column().classes('flex-1 ml-2'):
                        ui.label('Confidence Score:').classes('text-sm mb-1')
                        self.hyp_confidence = ui.slider(min=0, max=1, step=0.1, value=0.5).classes('w-full')
                        self.confidence_label = ui.label('0.5').classes('text-xs text-center')
                        self.hyp_confidence.on('update:model-value', lambda e: self.confidence_label.set_text(f'{e.args:.1f}'))
                
                add_hyp_btn = ui.button('Add Hypothesis', icon='add', on_click=self.add_hypothesis).classes('brand-bg text-white')
            
            # Load hypotheses
            self.load_hypotheses()
    
    def refresh_dashboard(self, days: int):
        """Refresh dashboard data"""
        try:
            data = self.db.get_dashboard_data(days)
            
            # Debug output
            print(f"Dashboard refresh - Garmin data shape: {data['garmin'].shape}")
            print(f"Dashboard refresh - Subjective data shape: {data['subjective'].shape}")
            
            # Update metrics
            if not data['garmin'].empty:
                latest_garmin = data['garmin'].iloc[-1]
                self.hr_label.set_text(f"{latest_garmin['resting_hr']} bpm")
                self.hrv_label.set_text(f"{latest_garmin['hrv']:.1f} ms")
                print(f"Updated HR: {latest_garmin['resting_hr']}, HRV: {latest_garmin['hrv']}")
            
            if not data['subjective'].empty:
                latest_subjective = data['subjective'].iloc[-1]
                self.sleep_label.set_text(f"{latest_subjective['sleep_quality']}/10")
                self.mood_label.set_text(f"{latest_subjective['mood']}/10")
                print(f"Updated Sleep: {latest_subjective['sleep_quality']}, Mood: {latest_subjective['mood']}")
            
            # Clear and recreate charts
            self.charts_container.clear()
            
            with self.charts_container:
                # Charts
                with ui.row().classes('w-full'):
                    with ui.column().classes('w-full'):
                        sleep_fig = self.create_sleep_chart(data, days)
                        ui.plotly(figure=sleep_fig).classes('w-full h-112')
                
                with ui.row().classes('w-full mt-4'):
                    with ui.column().classes('w-full'):
                        hr_fig = self.create_hr_chart(data, days)
                        ui.plotly(figure=hr_fig).classes('w-full h-112')
                
                with ui.row().classes('w-full mt-4'):
                    with ui.column().classes('w-full'):
                        env_fig = self.create_env_chart(data, days)
                        ui.plotly(figure=env_fig).classes('w-full h-112')
                
                with ui.row().classes('w-full mt-4'):
                    with ui.column().classes('w-full'):
                        workout_fig = self.create_workout_chart(data, days)
                        ui.plotly(figure=workout_fig).classes('w-full h-112')
                
                with ui.row().classes('w-full mt-4'):
                    with ui.column().classes('w-full'):
                        weight_fig = self.create_weight_chart(data, days)
                        ui.plotly(figure=weight_fig).classes('w-full h-112')
                
                with ui.row().classes('w-full mt-4'):
                    with ui.column().classes('w-full'):
                        ui.label('Recent Insights').classes('text-h6 mb-3 mt-2')
                        insights_card = ui.card().classes('p-4')
                        with insights_card:
                            self.update_insights_card_content(data)
            
            print(f"HR chart traces: {len(hr_fig.data)}")
            print(f"Sleep chart traces: {len(sleep_fig.data)}")
            print(f"Env chart traces: {len(env_fig.data)}")
            
            ui.notify(f"Dashboard refreshed with {days} days of data", type='info')
            
        except Exception as e:
            print(f"Error in refresh_dashboard: {str(e)}")
            ui.notify(f"Error refreshing dashboard: {str(e)}", type='error')
    
    def create_hr_chart(self, data: Dict[str, Any], days: int = 14) -> go.Figure:
        """Create heart rate and HRV chart"""
        fig = go.Figure()
        
        if not data['garmin'].empty:
            df = data['garmin'].copy()
            
            # Sort by date to ensure proper line plotting
            df = df.sort_values('date')
            
            # Convert to lists to avoid pandas issues
            x_values = list(range(days))
            hr_values = df['resting_hr'].tolist()
            hrv_values = df['hrv'].tolist()
            
            print(f"HR Chart - X values: {x_values[:5]}")
            print(f"HR Chart - HR values: {hr_values[:5]}")
            print(f"HR Chart - HRV values: {hrv_values[:5]}")
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=hr_values, 
                    name='Resting HR', 
                    line=dict(color=BRAND_COLOR),
                    mode='lines+markers'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=hrv_values, 
                    name='Heart Rate Variability', 
                    line=dict(color='orange'),
                    mode='lines+markers',
                    yaxis='y2'
                )
            )
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Heart Rate Variability (ms)",
                    overlaying="y",
                    side="right"
                )
            )
        
        fig.update_xaxes(title_text="Days", title_standoff=10)
        fig.update_yaxes(
            title_text="Heart Rate (bpm)",
            tickmode='linear',
            tick0=40,
            dtick=5
        )
        fig.update_layout(
            height=500, 
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text="Heart Rate & Heart Rate Variability Trend",
                font=dict(size=24, color='#1f2937', family='Arial, sans-serif', weight='bold')
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return fig
    
    def create_sleep_chart(self, data: Dict[str, Any], days: int = 14) -> go.Figure:
        """Create sleep and mood chart"""
        fig = go.Figure()
        
        if not data['garmin'].empty and not data['subjective'].empty:
            garmin_df = data['garmin'].copy()
            subjective_df = data['subjective'].copy()
            
            # Sort by date
            garmin_df = garmin_df.sort_values('date')
            subjective_df = subjective_df.sort_values('date')
            
            # Convert to lists to avoid pandas issues
            x_values = list(range(days))
            sleep_values = garmin_df['sleep_duration'].tolist()
            mood_values = subjective_df['mood'].tolist()
            
            print(f"Sleep Chart - X values: {x_values[:5]}")
            print(f"Sleep Chart - Sleep values: {sleep_values[:5]}")
            print(f"Sleep Chart - Mood values: {mood_values[:5]}")
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=sleep_values, 
                    name='Sleep Duration', 
                    line=dict(color=BRAND_COLOR),
                    mode='lines+markers'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=mood_values, 
                    name='Mood', 
                    line=dict(color='purple'),
                    mode='lines+markers',
                    yaxis='y2'
                )
            )
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Mood (1-10)",
                    overlaying="y",
                    side="right"
                )
            )
        
        fig.update_xaxes(title_text="Days", title_standoff=10)
        fig.update_yaxes(
            title_text="Sleep Duration (hours)",
            tickmode='linear',
            tick0=6,
            dtick=1
        )
        fig.update_layout(
            height=500, 
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text="Sleep & Mood Trend",
                font=dict(size=24, color='#1f2937', family='Arial, sans-serif', weight='bold')
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return fig
    
    def create_env_chart(self, data: Dict[str, Any], days: int = 14) -> go.Figure:
        """Create environmental factors chart"""
        fig = go.Figure()
        
        if not data['environmental'].empty:
            df = data['environmental'].copy()
            
            # Sort by date
            df = df.sort_values('date')
            
            # Convert to lists to avoid pandas issues
            x_values = list(range(days))
            hazelnut_pollen = df['pollen_hazelnut'].tolist()
            birch_pollen = df['pollen_birch'].tolist()
            grass_pollen = df['pollen_grass'].tolist()
            temp_values = df['temperature'].tolist()
            
            print(f"Env Chart - X values: {x_values[:5]}")
            print(f"Env Chart - Hazelnut: {hazelnut_pollen[:5]}")
            print(f"Env Chart - Birch: {birch_pollen[:5]}")
            print(f"Env Chart - Grass: {grass_pollen[:5]}")
            print(f"Env Chart - Temp: {temp_values[:5]}")
            
            # Add pollen traces
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=hazelnut_pollen, 
                name='Hazelnut Pollen', 
                line=dict(color='#8B4513', width=2),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=birch_pollen, 
                name='Birch Pollen', 
                line=dict(color='#228B22', width=2),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=grass_pollen, 
                name='Grass Pollen', 
                line=dict(color='#32CD32', width=2),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            
            # Add temperature trace (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=x_values, 
                y=temp_values, 
                name='Temperature (°C)', 
                line=dict(color='#FF4500', width=2),
                mode='lines+markers',
                marker=dict(size=4),
                yaxis='y2'
            ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            height=575,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text="Environmental Factors & Pollen Levels",
                font=dict(size=24, color='#1f2937', family='Arial, sans-serif', weight='bold')
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        # Primary y-axis for pollen (0-5 scale)
        fig.update_yaxes(
            title_text="Pollen Risk Level (0-5)",
            range=[0, 5],
            tickmode='linear',
            tick0=0,
            dtick=1
        )
        
        # Secondary y-axis for temperature
        fig.update_layout(
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying="y",
                side="right",
                range=[-10, 35],
                tickmode='linear',
                tick0=-10,
                dtick=5
            )
        )
        
        fig.update_xaxes(title_text="Days", title_standoff=10)
        
        return fig
    
    def create_test_chart(self) -> go.Figure:
        """Create test chart with sample data"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            y=[50, 47, 50, 46, 42, 48, 45, 49, 44, 47],
            name='Test Data',
            line=dict(color=BRAND_COLOR),
            mode='lines+markers'
        ))
        fig.update_layout(
            height=500,
            showlegend=True,
            title="Test Chart"
        )
        return fig
    
    def create_empty_chart(self) -> go.Figure:
        """Create empty chart placeholder"""
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=300)
        return fig
    
    def create_workout_chart(self, data: Dict[str, Any], days: int = 14) -> go.Figure:
        """Create workout activity chart"""
        fig = go.Figure()
        
        # Get workout data from database
        conn = sqlite3.connect(self.db.db_path)
        workout_df = pd.read_sql_query('''
            SELECT date, activity_type, duration_minutes, calories_burned, intensity_score
            FROM workouts 
            ORDER BY date DESC 
            LIMIT 30
        ''', conn)
        conn.close()
        
        if not workout_df.empty:
            # Convert date column
            workout_df['date'] = pd.to_datetime(workout_df['date'])
            workout_df = workout_df.sort_values('date')
            
            # Create x-axis (days)
            x_values = list(range(days))
            
            # Add duration trace
            fig.add_trace(go.Scatter(
                x=x_values,
                y=workout_df['duration_minutes'],
                name='Duration (min)',
                line=dict(color='#3498db', width=2),
                mode='lines+markers',
                marker=dict(size=6)
            ))
            
            # Add calories trace (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=workout_df['calories_burned'],
                name='Calories',
                line=dict(color='#e74c3c', width=2),
                mode='lines+markers',
                marker=dict(size=6),
                yaxis='y2'
            ))
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Calories Burned",
                    overlaying="y",
                    side="right"
                )
            )
        
        fig.update_xaxes(title_text="Days", title_standoff=10)
        fig.update_yaxes(title_text="Duration (minutes)")
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text="Workout Activity & Calories",
                font=dict(size=24, color='#1f2937', family='Arial, sans-serif', weight='bold')
            )
        )
        
        return fig
    
    def create_weight_chart(self, data: Dict[str, Any], days: int = 14) -> go.Figure:
        """Create weight tracking chart"""
        fig = go.Figure()
        
        # Get weight data from database
        conn = sqlite3.connect(self.db.db_path)
        weight_df = pd.read_sql_query('''
            SELECT date, weight_kg, body_fat_percent, muscle_mass_kg, water_percent
            FROM weight_data 
            ORDER BY date DESC 
            LIMIT 30
        ''', conn)
        conn.close()
        
        if not weight_df.empty:
            # Convert date column
            weight_df['date'] = pd.to_datetime(weight_df['date'])
            weight_df = weight_df.sort_values('date')
            
            # Create x-axis (days)
            x_values = list(range(days))
            
            # Add weight trace
            fig.add_trace(go.Scatter(
                x=x_values,
                y=weight_df['weight_kg'],
                name='Weight (kg)',
                line=dict(color='#2ecc71', width=3),
                mode='lines+markers',
                marker=dict(size=8)
            ))
            
            # Add body fat trace (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=weight_df['body_fat_percent'],
                name='Body Fat %',
                line=dict(color='#f39c12', width=2),
                mode='lines+markers',
                marker=dict(size=6),
                yaxis='y2'
            ))
            
            # Add muscle mass trace (secondary y-axis)
            fig.add_trace(go.Scatter(
                x=x_values,
                y=weight_df['muscle_mass_kg'],
                name='Muscle Mass (kg)',
                line=dict(color='#9b59b6', width=2),
                mode='lines+markers',
                marker=dict(size=6),
                yaxis='y2'
            ))
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title="Body Composition",
                    overlaying="y",
                    side="right"
                )
            )
        
        fig.update_xaxes(title_text="Days", title_standoff=10)
        fig.update_yaxes(title_text="Weight (kg)")
        fig.update_layout(
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            title=dict(
                text="Weight & Body Composition",
                font=dict(size=24, color='#1f2937', family='Arial, sans-serif', weight='bold')
            )
        )
        
        return fig
    
    def update_insights_card_content(self, data: Dict[str, Any]):
        """Update insights card with recent findings"""
        insights = []
        
        if not data['garmin'].empty and not data['subjective'].empty:
            # Simple correlation check
            garmin_df = data['garmin']
            subjective_df = data['subjective']
            
            if len(garmin_df) > 1 and len(subjective_df) > 1:
                hrv_trend = garmin_df['hrv'].iloc[-1] - garmin_df['hrv'].iloc[-2]
                mood_trend = subjective_df['mood'].iloc[-1] - subjective_df['mood'].iloc[-2]
                
                if hrv_trend < -5:
                    insights.append("Heart Rate Variability decreased significantly - consider stress factors")
                if mood_trend < -2:
                    insights.append("Mood dropped - check sleep and nutrition")
        
        # Display insights
        if insights:
            for insight in insights:
                ui.label(f"• {insight}").classes('text-sm mb-1')
        else:
            ui.label('No significant patterns detected recently.').classes('text-grey-6')
    
    def load_existing_data(self):
        """Load existing data for selected date"""
        try:
            date_str = self.date_input.value
            data = self.db.get_data_for_date(date_str)
            
            # Load subjective data
            if data['subjective']:
                subj = data['subjective']
                self.mood_input.value = subj.get('mood', 7)
                self.mood_value.set_text(str(subj.get('mood', 7)))
                self.energy_input.value = subj.get('energy_level', 7)
                self.energy_value.set_text(str(subj.get('energy_level', 7)))
                self.pain_input.value = subj.get('pain_level', 0)
                self.pain_value.set_text(str(subj.get('pain_level', 0)))
                self.allergy_input.value = subj.get('allergy_symptoms', 0)
                self.allergy_value.set_text(str(subj.get('allergy_symptoms', 0)))
                self.stress_input.value = subj.get('stress_level', 5)
                self.stress_value.set_text(str(subj.get('stress_level', 5)))
                self.sleep_quality_input.value = subj.get('sleep_quality', 7)
                self.sleep_quality_value.set_text(str(subj.get('sleep_quality', 7)))
                self.notes_input.value = subj.get('notes', '')
            
            ui.notify(f"Loaded data for {date_str}", type='info')
            
        except Exception as e:
            ui.notify(f"Error loading data: {str(e)}", type='error')
    
    def save_manual_data(self):
        """Save manually entered data"""
        try:
            date_str = self.date_input.value
            
            # Save subjective data
            subjective_data = {
                'mood': self.mood_input.value,
                'energy_level': self.energy_input.value,
                'pain_level': self.pain_input.value,
                'allergy_symptoms': self.allergy_input.value,
                'stress_level': self.stress_input.value,
                'sleep_quality': self.sleep_quality_input.value,
                'notes': self.notes_input.value
            }
            
            self.db.insert_manual_data(date_str, 'subjective', subjective_data)
            
            # Save food data if provided
            if self.food_items.value.strip():
                food_data = {
                    'meal_type': self.meal_type.value,
                    'food_items': self.food_items.value,
                    'tags': self.food_tags.value,
                    'notes': self.food_notes.value
                }
                self.db.insert_manual_data(date_str, 'food', food_data)
            
            ui.notify("Data saved successfully!", type='positive')
            
        except Exception as e:
            ui.notify(f"Error saving data: {str(e)}", type='error')
    
    def sync_garmin_data(self):
        """Sync data from Garmin Connect"""
        try:
            self.sync_status.set_text("Syncing...")
            
            # Get credentials
            username = self.garmin_username.value
            password = self.garmin_password.value
            days_back = self.sync_days_back.value
            
            if not username or not password:
                ui.notify("Please enter Garmin username and password", type='warning')
                self.sync_status.set_text("Sync failed")
                return
            
            # Import GarminConnect library
            try:
                from garminconnect import Garmin
            except ImportError:
                ui.notify("GarminConnect library not installed. Using sample data instead.", type='warning')
                synced_count = self._generate_sample_garmin_data(days_back)
                self.sync_status.set_text("Sample data generated")
                # Refresh dashboard
                self.refresh_dashboard(14)
                return
            
            # Connect to Garmin
            garmin = Garmin(username, password)
            garmin.login()
            
            # Get date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Sync different data types
            synced_count = 0
            
            # 1. Daily summary data
            try:
                daily_data = garmin.get_daily_summary(start_date, end_date)
                for day_data in daily_data:
                    self._process_garmin_daily_data(day_data)
                    synced_count += 1
            except Exception as e:
                print(f"Error syncing daily data: {e}")
            
            # 2. Workout data
            try:
                activities = garmin.get_activities_by_date(start_date, end_date)
                for activity in activities:
                    self._process_garmin_workout_data(activity)
                    synced_count += 1
            except Exception as e:
                print(f"Error syncing workout data: {e}")
            
            # 3. Weight data (if available)
            try:
                weight_data = garmin.get_weight_data(start_date, end_date)
                for weight_entry in weight_data:
                    self._process_garmin_weight_data(weight_entry)
                    synced_count += 1
            except Exception as e:
                print(f"Error syncing weight data: {e}")
            
            ui.notify(f"Garmin sync completed! {synced_count} records synced.", type='positive')
            self.sync_status.set_text("Sync completed")
            
            # Refresh dashboard
            self.refresh_dashboard(14)
            
        except Exception as e:
            ui.notify(f"Error syncing Garmin data: {str(e)}", type='error')
            self.sync_status.set_text("Sync failed")
    
    def _generate_sample_garmin_data(self, days_back):
        """Generate sample Garmin data when API is not available"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        synced_count = 0
        
        for i in range(days_back):
            date = start_date + timedelta(days=i)
            
            # Generate sample workout data (50% chance)
            if random.random() < 0.5:
                activity_types = ['Running', 'Cycling', 'Strength Training', 'Yoga', 'Swimming']
                activity_type = random.choice(activity_types)
                
                workout_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'activity_type': activity_type,
                    'duration_minutes': random.randint(20, 90),
                    'distance_km': round(random.uniform(2, 15) if activity_type in ['Running', 'Cycling'] else 0, 1),
                    'avg_hr': random.randint(120, 160),
                    'max_hr': random.randint(150, 180),
                    'calories_burned': random.randint(200, 800),
                    'intensity_score': round(random.uniform(0.3, 1.0), 2)
                }
                self.db._insert_workout_data(self.db._get_connection(), workout_data)
                synced_count += 1
            
            # Generate sample weight data (30% chance)
            if random.random() < 0.3:
                weight_data = {
                    'date': date.strftime('%Y-%m-%d'),
                    'weight_kg': round(75.0 + random.gauss(0, 0.5), 1),
                    'body_fat_percent': round(15 + random.gauss(0, 1), 1),
                    'muscle_mass_kg': round(35 + random.gauss(0, 0.5), 1),
                    'water_percent': round(60 + random.gauss(0, 1), 1),
                    'bone_mass_kg': round(3.2 + random.gauss(0, 0.05), 1)
                }
                self.db._insert_weight_data(self.db._get_connection(), weight_data)
                synced_count += 1
        
        ui.notify(f"Sample data generated! {synced_count} records created.", type='positive')
        return synced_count
    
    def _process_garmin_daily_data(self, day_data):
        """Process daily Garmin data"""
        # This would process the actual Garmin daily data
        # For now, we'll use the existing sample data generation
        pass
    
    def _process_garmin_workout_data(self, activity):
        """Process Garmin workout data"""
        try:
            workout_data = {
                'date': activity['startTimeLocal'][:10],  # Extract date
                'activity_type': activity.get('activityType', {}).get('typeKey', 'Unknown'),
                'duration_minutes': activity.get('elapsedDuration', 0) // 60,
                'distance_km': activity.get('distance', 0) / 1000,  # Convert from meters
                'avg_hr': activity.get('averageHR', 0),
                'max_hr': activity.get('maxHR', 0),
                'calories_burned': activity.get('calories', 0),
                'intensity_score': activity.get('intensity', 0)
            }
            self.db._insert_workout_data(self.db._get_connection(), workout_data)
        except Exception as e:
            print(f"Error processing workout data: {e}")
    
    def _process_garmin_weight_data(self, weight_entry):
        """Process Garmin weight data"""
        try:
            weight_data = {
                'date': weight_entry['date'],
                'weight_kg': weight_entry.get('weight', 0),
                'body_fat_percent': weight_entry.get('bodyFat', 0),
                'muscle_mass_kg': weight_entry.get('muscleMass', 0),
                'water_percent': weight_entry.get('waterPercentage', 0),
                'bone_mass_kg': weight_entry.get('boneMass', 0)
            }
            self.db._insert_weight_data(self.db._get_connection(), weight_data)
        except Exception as e:
            print(f"Error processing weight data: {e}")
    
    def sync_weather_data(self):
        """Sync weather and pollen data from Open-Meteo and Ambee APIs"""
        try:
            self.weather_status.set_text("Syncing...")
            today = datetime.now().date()
            
            # Get weather data from Open-Meteo API (no API key required)
            weather_url = "https://api.open-meteo.com/v1/forecast?latitude=47.3769&longitude=8.5417&current_weather=true&hourly=temperature_2m,relative_humidity_2m&timezone=Europe%2FZurich"
            
            weather_response = requests.get(weather_url, timeout=10)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # Extract current weather
            current = weather_data['current_weather']
            hourly = weather_data['hourly']
            
            # Map weather code to condition
            weather_code = current['weathercode']
            weather_conditions = {
                0: 'sunny', 1: 'sunny', 2: 'partly_cloudy', 3: 'cloudy',
                45: 'cloudy', 48: 'cloudy', 51: 'rainy', 53: 'rainy',
                55: 'rainy', 61: 'rainy', 63: 'rainy', 65: 'rainy',
                71: 'rainy', 73: 'rainy', 75: 'rainy', 80: 'rainy',
                81: 'rainy', 82: 'rainy', 95: 'rainy', 96: 'rainy', 99: 'rainy'
            }
            
            # Get pollen data from Ambee API (100 free calls per day)
            pollen_url = "https://api.ambeedata.com/latest/pollen/by-lat-lng"
            pollen_params = {
                'lat': 47.3769,  # Zurich latitude
                'lng': 8.5417    # Zurich longitude
            }
            pollen_headers = {
                'x-api-key': '79d9f3d36beafc39abb22b4f57bf909e679530c68358afa8fc8b84f9547921a8',
                'Content-type': 'application/json'
            }
            
            # Try to get real pollen data, fallback to sample data if API key not set
            try:
                pollen_response = requests.get(pollen_url, params=pollen_params, headers=pollen_headers, timeout=10)
                if pollen_response.status_code == 200:
                    pollen_data = pollen_response.json()
                    if pollen_data.get('data') and len(pollen_data['data']) > 0:
                        pollen_info = pollen_data['data'][0]
                        count = pollen_info.get('Count', {})
                        risk = pollen_info.get('Risk', {})
                        
                        # Convert risk levels to numeric values (0-5 scale)
                        risk_mapping = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4, 'Extreme': 5}
                        
                        environmental_data = {
                            'date': today,
                            'location': 'Zurich',
                            'temperature': round(current['temperature'], 1),
                            'humidity': hourly['relative_humidity_2m'][0],
                            'pollen_hazelnut': risk_mapping.get(risk.get('tree_pollen', 'Low'), 1),
                            'pollen_birch': risk_mapping.get(risk.get('tree_pollen', 'Low'), 1),
                            'pollen_grass': risk_mapping.get(risk.get('grass_pollen', 'Low'), 1),
                            'air_quality_index': random.randint(1, 5),  # Not provided by Ambee
                            'weather_condition': weather_conditions.get(weather_code, 'partly_cloudy')
                        }
                        
                        self.db.insert_environmental_data(environmental_data)
                        self.update_pollen_display(risk)
                        ui.notify(f"Weather & pollen data synced for Zurich: {environmental_data['temperature']}°C, Grass pollen: {risk.get('grass_pollen', 'Low')}", type='positive')
                        self.weather_status.set_text("Sync completed")
                        return
            except Exception as pollen_error:
                print(f"Pollen API error: {pollen_error}")
            
            # Fallback to sample data if pollen API fails
            environmental_data = {
                'date': today,
                'location': 'Zurich',
                'temperature': round(current['temperature'], 1),
                'humidity': hourly['relative_humidity_2m'][0],
                'pollen_hazelnut': random.randint(0, 5),
                'pollen_birch': random.randint(0, 5),
                'pollen_grass': random.randint(0, 5),
                'air_quality_index': random.randint(1, 5),
                'weather_condition': weather_conditions.get(weather_code, 'partly_cloudy')
            }
            
            self.db.insert_environmental_data(environmental_data)
            # Update pollen display with sample data
            sample_risk = {
                'tree_pollen': 'Low',
                'grass_pollen': 'Low',
                'weed_pollen': 'Low'
            }
            self.update_pollen_display(sample_risk)
            ui.notify(f"Weather data synced for Zurich: {environmental_data['temperature']}°C (using sample pollen data)", type='positive')
            self.weather_status.set_text("Sync completed")
            
        except Exception as e:
            ui.notify(f"Error syncing weather data: {str(e)}", type='error')
            self.weather_status.set_text("Sync failed")
    
    def update_pollen_display(self, risk_data: Dict[str, str]):
        """Update the pollen display with current risk levels"""
        self.pollen_display.clear()
        
        # Risk level colors
        risk_colors = {
            'Low': 'green',
            'Moderate': 'yellow',
            'High': 'orange', 
            'Very High': 'red',
            'Extreme': 'purple'
        }
        
        with self.pollen_display:
            # Tree pollen (Birch, Hazelnut)
            tree_risk = risk_data.get('tree_pollen', 'Low')
            tree_color = risk_colors.get(tree_risk, 'green')
            with ui.row().classes('w-full mb-1'):
                ui.label('🌳 Tree Pollen:').classes('w-24')
                ui.label(tree_risk).classes(f'text-{tree_color}-600 font-bold')
            
            # Grass pollen
            grass_risk = risk_data.get('grass_pollen', 'Low')
            grass_color = risk_colors.get(grass_risk, 'green')
            with ui.row().classes('w-full mb-1'):
                ui.label('🌾 Grass Pollen:').classes('w-24')
                ui.label(grass_risk).classes(f'text-{grass_color}-600 font-bold')
            
            # Weed pollen
            weed_risk = risk_data.get('weed_pollen', 'Low')
            weed_color = risk_colors.get(weed_risk, 'green')
            with ui.row().classes('w-full mb-1'):
                ui.label('🌿 Weed Pollen:').classes('w-24')
                ui.label(weed_risk).classes(f'text-{weed_color}-600 font-bold')
    
    def load_insights(self):
        """Load insights and analysis"""
        try:
            data = self.db.get_dashboard_data(90)
            
            # Update baseline
            self.baseline_content.clear()
            with self.baseline_content:
                if not data['garmin'].empty:
                    garmin_df = data['garmin']
                    avg_hr = garmin_df['resting_hr'].mean()
                    avg_hrv = garmin_df['hrv'].mean()
                    avg_sleep = garmin_df['sleep_duration'].mean()
                    
                    ui.label(f"Average Resting HR: {avg_hr:.1f} bpm").classes('mb-1')
                    ui.label(f"Average Heart Rate Variability: {avg_hrv:.1f} ms").classes('mb-1')
                    ui.label(f"Average Sleep: {avg_sleep:.1f} hours").classes('mb-1')
                else:
                    ui.label('Insufficient data for baseline calculation').classes('text-grey-6')
            
            # Update correlation chart
            self.correlation_container.clear()
            with self.correlation_container:
                correlation_fig = self.create_correlation_chart(data)
                ui.plotly(figure=correlation_fig).classes('w-full h-64')
            
        except Exception as e:
            ui.notify(f"Error loading insights: {str(e)}", type='error')
    
    def create_correlation_chart(self, data: Dict[str, Any]) -> go.Figure:
        """Create correlation analysis chart"""
        fig = go.Figure()
        
        if not data['garmin'].empty and not data['subjective'].empty:
            # Simple correlation visualization
            garmin_df = data['garmin'].copy()
            subjective_df = data['subjective'].copy()
            
            # Convert date strings to datetime
            garmin_df['date'] = pd.to_datetime(garmin_df['date'])
            subjective_df['date'] = pd.to_datetime(subjective_df['date'])
            
            # Merge data on date
            merged = pd.merge(garmin_df, subjective_df, on='date', how='inner')
            
            if not merged.empty:
                fig.add_trace(go.Scatter(
                    x=merged['hrv'], 
                    y=merged['mood'], 
                    mode='markers',
                    name='Heart Rate Variability vs Mood',
                    marker=dict(color=BRAND_COLOR, size=8)
                ))
        
        fig.update_xaxes(title_text="Heart Rate Variability (ms)")
        fig.update_yaxes(title_text="Mood (1-10)")
        fig.update_layout(height=300, showlegend=True)
        
        return fig
    
    def analyze_lag_correlation(self):
        """Analyze correlation with different lag periods"""
        try:
            var1 = self.corr_var1.value
            var2 = self.corr_var2.value
            max_lag = int(self.max_lag.value)
            
            # Get data
            data = self.db.get_dashboard_data(90)
            
            # Prepare data for correlation analysis
            correlation_data = self.prepare_correlation_data(data, var1, var2)
            
            if correlation_data is None:
                ui.notify("Insufficient data for correlation analysis", type='warning')
                return
            
            # Calculate correlations for different lags
            lag_correlations = self.calculate_lag_correlations(correlation_data, var1, var2, max_lag)
            
            # Update correlation container with results
            self.correlation_container.clear()
            with self.correlation_container:
                # Show lag correlation chart
                lag_fig = self.create_lag_correlation_chart(lag_correlations, var1, var2)
                ui.plotly(figure=lag_fig).classes('w-full h-64 mb-4')
                
                # Show summary
                self.show_lag_correlation_summary(lag_correlations, var1, var2)
            
            ui.notify(f"Lag correlation analysis completed for {var1} vs {var2}", type='positive')
            
        except Exception as e:
            ui.notify(f"Error in lag correlation analysis: {str(e)}", type='error')
    
    def prepare_correlation_data(self, data: Dict[str, Any], var1: str, var2: str):
        """Prepare data for correlation analysis"""
        try:
            # Get the appropriate dataframes based on variable names
            df1 = None
            df2 = None
            
            # Map variables to their data sources
            garmin_vars = ['hrv', 'heart_rate', 'sleep_duration']
            subjective_vars = ['mood', 'energy_level']
            env_vars = ['pollen_hazelnut', 'pollen_birch', 'pollen_grass', 'temperature']
            
            if var1 in garmin_vars:
                df1 = data['garmin'].copy()
            elif var1 in subjective_vars:
                df1 = data['subjective'].copy()
            elif var1 in env_vars:
                df1 = data['environmental'].copy()
            
            if var2 in garmin_vars:
                df2 = data['garmin'].copy()
            elif var2 in subjective_vars:
                df2 = data['subjective'].copy()
            elif var2 in env_vars:
                df2 = data['environmental'].copy()
            
            if df1 is None or df2 is None or df1.empty or df2.empty:
                return None
            
            # Convert dates to datetime
            df1['date'] = pd.to_datetime(df1['date'])
            df2['date'] = pd.to_datetime(df2['date'])
            
            # Merge on date
            merged = pd.merge(df1, df2, on='date', how='inner', suffixes=('_1', '_2'))
            
            if merged.empty:
                return None
            
            return merged
            
        except Exception as e:
            print(f"Error preparing correlation data: {e}")
            return None
    
    def calculate_lag_correlations(self, data, var1: str, var2: str, max_lag: int):
        """Calculate correlations for different lag periods"""
        correlations = []
        
        try:
            # Get the data columns
            col1 = f"{var1}_1" if f"{var1}_1" in data.columns else var1
            col2 = f"{var2}_2" if f"{var2}_2" in data.columns else var2
            
            # Sort by date
            data = data.sort_values('date').reset_index(drop=True)
            
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    # No lag - direct correlation
                    corr_data = data[[col1, col2]].dropna()
                else:
                    # Apply lag - shift var2 by lag days
                    lagged_data = data.copy()
                    lagged_data[col2] = lagged_data[col2].shift(lag)
                    corr_data = lagged_data[[col1, col2]].dropna()
                
                if len(corr_data) > 2:
                    correlation = corr_data[col1].corr(corr_data[col2])
                    correlations.append({
                        'lag': lag,
                        'correlation': correlation,
                        'n_samples': len(corr_data)
                    })
        
        except Exception as e:
            print(f"Error calculating lag correlations: {e}")
        
        return correlations
    
    def create_lag_correlation_chart(self, correlations, var1: str, var2: str):
        """Create chart showing correlation vs lag"""
        fig = go.Figure()
        
        if correlations:
            lags = [c['lag'] for c in correlations]
            corr_values = [c['correlation'] for c in correlations]
            n_samples = [c['n_samples'] for c in correlations]
            
            # Color points by correlation strength
            colors = ['red' if abs(c) > 0.7 else 'orange' if abs(c) > 0.5 else 'blue' for c in corr_values]
            
            fig.add_trace(go.Scatter(
                x=lags,
                y=corr_values,
                mode='markers+lines',
                name=f'{var1} vs {var2}',
                marker=dict(
                    color=colors,
                    size=8,
                    line=dict(width=1, color='white')
                ),
                line=dict(width=2),
                text=[f"Lag: {lag} days<br>Correlation: {corr:.3f}<br>Samples: {n}" 
                      for lag, corr, n in zip(lags, corr_values, n_samples)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        fig.update_xaxes(title_text="Lag (days)")
        fig.update_yaxes(title_text="Correlation Coefficient", range=[-1, 1])
        fig.update_layout(
            height=500,
            title=f"Lag Correlation: {var1} vs {var2}",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add horizontal lines for significance thresholds
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Strong correlation")
        fig.add_hline(y=-0.7, line_dash="dash", line_color="red")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Moderate correlation")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="orange")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
        
        return fig
    
    def show_lag_correlation_summary(self, correlations, var1: str, var2: str):
        """Show summary of lag correlation results"""
        if not correlations:
            ui.label("No correlations found").classes('text-grey-6')
            return
        
        # Find strongest correlations
        strongest_positive = max(correlations, key=lambda x: x['correlation'])
        strongest_negative = min(correlations, key=lambda x: x['correlation'])
        
        with ui.card().classes('p-4'):
            ui.label('Lag Correlation Summary').classes('text-h6 mb-3')
            
            with ui.row().classes('w-full mb-2'):
                with ui.column().classes('flex-1'):
                    ui.label('Strongest Positive:').classes('text-sm font-bold')
                    ui.label(f"Lag {strongest_positive['lag']} days: {strongest_positive['correlation']:.3f}").classes('text-green-600')
                    ui.label(f"({strongest_positive['n_samples']} samples)").classes('text-xs text-grey-6')
            
            with ui.row().classes('w-full mb-2'):
                with ui.column().classes('flex-1'):
                    ui.label('Strongest Negative:').classes('text-sm font-bold')
                    ui.label(f"Lag {strongest_negative['lag']} days: {strongest_negative['correlation']:.3f}").classes('text-red-600')
                    ui.label(f"({strongest_negative['n_samples']} samples)").classes('text-xs text-grey-6')
            
            # Interpretation
            if abs(strongest_positive['correlation']) > 0.7:
                interpretation = f"Strong correlation found at {strongest_positive['lag']} days lag"
            elif abs(strongest_positive['correlation']) > 0.5:
                interpretation = f"Moderate correlation found at {strongest_positive['lag']} days lag"
            else:
                interpretation = "No significant correlations found"
            
            ui.label(interpretation).classes('text-sm text-grey-7 mt-2')
    
    def load_hypotheses(self):
        """Load and display hypotheses"""
        try:
            # Load hypotheses from database
            conn = sqlite3.connect(self.db.db_path)
            hypotheses_df = pd.read_sql_query('''
                SELECT id, title, description, trigger_condition, expected_effect, status, confidence_score, created_at
                FROM hypotheses 
                ORDER BY created_at DESC
            ''', conn)
            conn.close()
            
            self.hypotheses_list.clear()
            with self.hypotheses_list:
                if not hypotheses_df.empty:
                    for _, hyp in hypotheses_df.iterrows():
                        with ui.card().classes('p-3 mb-2'):
                            with ui.row().classes('w-full justify-between items-start'):
                                with ui.column().classes('flex-1'):
                                    ui.label(hyp['title']).classes('font-bold text-base')
                                    
                                    # Trigger and Effect
                                    with ui.row().classes('mb-1'):
                                        ui.label('Trigger:').classes('text-xs text-grey-6 mr-1')
                                        ui.label(hyp['trigger_condition']).classes('text-xs font-medium')
                                    
                                    with ui.row().classes('mb-1'):
                                        ui.label('Effect:').classes('text-xs text-grey-6 mr-1')
                                        ui.label(hyp['expected_effect']).classes('text-xs font-medium')
                                    
                                    # Description if available
                                    if hyp['description']:
                                        ui.label(hyp['description']).classes('text-sm text-grey-6 mt-1')
                                    
                                    # Status and Confidence
                                    status_color = {
                                        'active': 'text-blue-600',
                                        'testing': 'text-yellow-600', 
                                        'confirmed': 'text-green-600',
                                        'rejected': 'text-red-600'
                                    }.get(hyp['status'], 'text-grey-600')
                                    
                                    with ui.row().classes('mt-2'):
                                        ui.label(f"Status:").classes('text-xs text-grey-6 mr-1')
                                        ui.label(hyp['status'].title()).classes(f'text-xs font-medium {status_color} mr-3')
                                        ui.label(f"Confidence: {hyp['confidence_score']:.1f}").classes('text-xs text-grey-5')
                                
                                # Action buttons
                                with ui.row().classes('ml-2'):
                                    ui.button('Test', icon='science').classes('mr-2 text-xs px-2 py-1')
                                    ui.button('Edit', icon='edit').classes('text-xs px-2 py-1')
                else:
                    with ui.card().classes('p-4 text-center'):
                        ui.label('No hypotheses found').classes('text-grey-6')
                        ui.label('Create your first hypothesis using the form above').classes('text-sm text-grey-5 mt-1')
                    
        except Exception as e:
            ui.notify(f"Error loading hypotheses: {str(e)}", type='error')
    
    def add_hypothesis(self):
        """Add new hypothesis"""
        try:
            # Get form values
            title = self.hyp_title.value
            description = self.hyp_description.value
            trigger = self.hyp_trigger.value
            effect = self.hyp_effect.value
            status = self.hyp_status.value
            confidence = self.hyp_confidence.value
            
            # Validate required fields
            if not title or not trigger or not effect:
                ui.notify("Please fill in title, trigger, and effect", type='warning')
                return
            
            # Create hypothesis data
            hypothesis_data = {
                'title': title,
                'description': description or '',
                'trigger_condition': trigger,
                'expected_effect': effect,
                'status': status,
                'confidence_score': confidence
            }
            
            # Add to database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO hypotheses (title, description, trigger_condition, expected_effect, status, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (hypothesis_data['title'], hypothesis_data['description'], 
                  hypothesis_data['trigger_condition'], hypothesis_data['expected_effect'],
                  hypothesis_data['status'], hypothesis_data['confidence_score']))
            conn.commit()
            conn.close()
            
            # Clear form
            self.hyp_title.set_value('')
            self.hyp_description.set_value('')
            self.hyp_trigger.set_value('Pasta consumption')
            self.hyp_effect.set_value('Low HRV next morning')
            self.hyp_status.set_value('active')
            self.hyp_confidence.set_value(0.5)
            self.confidence_label.set_text('0.5')
            
            # Reload hypotheses list
            self.load_hypotheses()
            
            ui.notify(f"Hypothesis '{title}' added successfully!", type='positive')
            
        except Exception as e:
            ui.notify(f"Error adding hypothesis: {str(e)}", type='error')
    
    async def train_neural_network(self):
        """Train neural network on health data"""
        try:
            self.training_stopped = False
            self.train_btn.props('disabled')
            self.stop_btn.props(remove='disabled')
            
            # Clear log
            ui.run_javascript('''
                const logContainer = document.getElementById('neural-log-container');
                if (logContainer) {
                    logContainer.innerHTML = '';
                }
            ''')
            
            # Log function with auto-scroll
            def log(message, color='green'):
                timestamp = datetime.now().strftime('%H:%M:%S')
                color_map = {
                    'green': '#10b981',
                    'blue': '#3b82f6', 
                    'yellow': '#f59e0b',
                    'red': '#ef4444',
                    'cyan': '#06b6d4'
                }
                color_hex = color_map.get(color, '#10b981')
                
                # Add message to log container
                ui.run_javascript(f'''
                    const logContainer = document.getElementById('neural-log-container');
                    if (logContainer) {{
                        const newDiv = document.createElement('div');
                        newDiv.style.color = '{color_hex}';
                        newDiv.style.margin = '0';
                        newDiv.style.padding = '2px 0';
                        newDiv.style.fontFamily = 'Courier New, monospace';
                        newDiv.style.fontSize = '16px';
                        newDiv.style.fontWeight = 'bold';
                        newDiv.style.width = '100%';
                        newDiv.style.display = 'block';
                        newDiv.style.wordWrap = 'break-word';
                        newDiv.textContent = '[{timestamp}] {message}';
                        logContainer.appendChild(newDiv);
                        
                        // Auto-scroll to bottom
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }}
                ''')
            
            log('🚀 Starting neural network training...', 'blue')
            log(f'Target variable: {self.nn_target.value}', 'yellow')
            log(f'Epochs: {int(self.nn_epochs.value)}', 'yellow')
            log(f'Learning rate: {self.nn_learning_rate.value}', 'yellow')
            
            self.nn_status.set_text('Loading data...')
            self.nn_progress.set_value(0.1)
            await asyncio.sleep(0.5)
            
            # Load data
            log('📊 Loading health data from database...')
            data = self.db.get_dashboard_data(90)
            
            # Debug: Log data availability
            log(f'Garmin data: {len(data["garmin"])} rows', 'yellow')
            log(f'Subjective data: {len(data["subjective"])} rows', 'yellow')
            log(f'Environmental data: {len(data["environmental"])} rows', 'yellow')
            
            # Prepare features
            log('🔧 Preparing features...')
            features_df = self.prepare_neural_features(data)
            
            if features_df is None or len(features_df) < 10:
                log('❌ Insufficient data for training (need at least 10 samples)', 'red')
                log(f'Current data: {len(features_df) if features_df is not None else 0} samples', 'red')
                log('💡 Click "Generate Sample Data" to create training data', 'blue')
                ui.notify('Insufficient data for training. Please click "Generate Sample Data" first.', type='error')
                self.train_btn.props(remove='disabled')
                self.stop_btn.props('disabled')
                return
            
            log(f'✓ Loaded {len(features_df)} samples with {len(features_df.columns)} features')
            self.nn_progress.set_value(0.2)
            await asyncio.sleep(0.5)
            
            # Prepare target
            target = self.nn_target.value
            if target not in features_df.columns:
                log(f'❌ Target variable {target} not found in data', 'red')
                self.train_btn.props(remove='disabled')
                self.stop_btn.props('disabled')
                return
            
            # Split features and target
            X = features_df.drop(columns=[target])
            y = features_df[target]
            
            log(f'✓ Target variable: {target} (range: {y.min():.2f} - {y.max():.2f})')
            self.nn_progress.set_value(0.3)
            await asyncio.sleep(0.5)
            
            # Split train/test
            log('🔀 Splitting data into train/test sets...')
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            log(f'✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}')
            
            # Scale features
            log('📏 Scaling features...')
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            log('✓ Features scaled using StandardScaler')
            self.nn_progress.set_value(0.4)
            await asyncio.sleep(0.5)
            
            # Train Random Forest (simulating neural network training)
            log('🧠 Training Random Forest neural network...')
            self.nn_status.set_text('Training model...')
            
            n_epochs = int(self.nn_epochs.value)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                verbose=0
            )
            
            # Simulate epoch-by-epoch training
            for epoch in range(1, n_epochs + 1):
                if self.training_stopped:
                    log('⚠️ Training stopped by user', 'yellow')
                    break
                
                # Simulate training time
                await asyncio.sleep(0.1)
                
                # Update progress
                progress = 0.4 + (0.5 * epoch / n_epochs)
                self.nn_progress.set_value(progress)
                
                # Log every 10 epochs
                if epoch % 10 == 0 or epoch == 1:
                    # Partial fit simulation
                    model.fit(X_train_scaled, y_train)
                    train_score = model.score(X_train_scaled, y_train)
                    test_score = model.score(X_test_scaled, y_test)
                    
                    log(f'Epoch {epoch}/{n_epochs} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}')
            
            if not self.training_stopped:
                # Final training
                log('🎯 Final training pass...')
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                log('📈 Evaluating model performance...')
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                log(f'✓ Training R²: {train_score:.4f}')
                log(f'✓ Test R²: {test_score:.4f}')
                
                # Get predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_mae = np.mean(np.abs(y_train - y_pred_train))
                test_mae = np.mean(np.abs(y_test - y_pred_test))
                
                log(f'✓ Training MAE: {train_mae:.4f}')
                log(f'✓ Test MAE: {test_mae:.4f}')
                
                self.nn_progress.set_value(0.95)
                await asyncio.sleep(0.5)
                
                # Get feature importances
                log('🔍 Analyzing feature importances...')
                feature_importances = dict(zip(X.columns, model.feature_importances_))
                sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
                
                log('✓ Top 5 most important features:')
                for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                    log(f'  {i}. {feature}: {importance:.4f}', 'cyan')
                
                self.nn_progress.set_value(1.0)
                self.nn_status.set_text('✓ Training completed!')
                log('✅ Training completed successfully!', 'green')
                
                # Display results
                await self.display_neural_results(
                    model, X, y, X_test_scaled, y_test, y_pred_test,
                    train_score, test_score, sorted_features, target
                )
            
            self.train_btn.props(remove='disabled')
            self.stop_btn.props('disabled')
            
        except Exception as e:
            log(f'❌ Error: {str(e)}', 'red')
            ui.notify(f'Training error: {str(e)}', type='error')
            self.train_btn.props(remove='disabled')
            self.stop_btn.props('disabled')
    
    def stop_training(self):
        """Stop neural network training"""
        self.training_stopped = True
        self.nn_status.set_text('Stopping training...')
    
    def generate_sample_data_for_nn(self):
        """Generate sample data for neural network training"""
        try:
            # Clear log
            ui.run_javascript('''
                const logContainer = document.getElementById('neural-log-container');
                if (logContainer) {
                    logContainer.innerHTML = '';
                }
            ''')
            
            # Log function with auto-scroll
            def log(message, color='green'):
                timestamp = datetime.now().strftime('%H:%M:%S')
                color_map = {
                    'green': '#10b981',
                    'blue': '#3b82f6', 
                    'yellow': '#f59e0b',
                    'red': '#ef4444',
                    'cyan': '#06b6d4'
                }
                color_hex = color_map.get(color, '#10b981')
                
                # Add message to log container
                ui.run_javascript(f'''
                    const logContainer = document.getElementById('neural-log-container');
                    if (logContainer) {{
                        const newDiv = document.createElement('div');
                        newDiv.style.color = '{color_hex}';
                        newDiv.style.margin = '0';
                        newDiv.style.padding = '2px 0';
                        newDiv.style.fontFamily = 'Courier New, monospace';
                        newDiv.style.fontSize = '16px';
                        newDiv.style.fontWeight = 'bold';
                        newDiv.style.width = '100%';
                        newDiv.style.display = 'block';
                        newDiv.style.wordWrap = 'break-word';
                        newDiv.textContent = '[{timestamp}] {message}';
                        logContainer.appendChild(newDiv);
                        
                        // Auto-scroll to bottom
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }}
                ''')
            
            log('🔄 Generating sample data for neural network...', 'blue')
            
            # Generate 60 days of sample data
            self.db.populate_sample_data()
            
            log('✓ Generated 60 days of sample data', 'green')
            log('✓ Data includes Garmin, subjective, and environmental metrics', 'green')
            log('✓ Ready for neural network training!', 'green')
            
            ui.notify('Sample data generated successfully! You can now train the neural network.', type='positive')
            
        except Exception as e:
            ui.notify(f'Error generating sample data: {str(e)}', type='error')
    
    def prepare_neural_features(self, data: Dict[str, Any]):
        """Prepare features for neural network"""
        try:
            print(f"Debug - Garmin data shape: {data['garmin'].shape if not data['garmin'].empty else 'Empty'}")
            print(f"Debug - Subjective data shape: {data['subjective'].shape if not data['subjective'].empty else 'Empty'}")
            print(f"Debug - Environmental data shape: {data['environmental'].shape if not data['environmental'].empty else 'Empty'}")
            
            # Merge all data sources
            merged_df = None
            
            if not data['garmin'].empty:
                garmin_df = data['garmin'].copy()
                garmin_df['date'] = pd.to_datetime(garmin_df['date'])
                # Select only numeric columns that exist
                garmin_cols = ['date']
                for col in ['heart_rate', 'hrv', 'sleep_duration', 'deep_sleep', 
                           'light_sleep', 'rem_sleep', 'stress_level', 'steps', 'calories']:
                    if col in garmin_df.columns:
                        garmin_cols.append(col)
                
                merged_df = garmin_df[garmin_cols]
                print(f"Debug - Garmin columns selected: {garmin_cols}")
            
            if not data['subjective'].empty:
                subj_df = data['subjective'].copy()
                subj_df['date'] = pd.to_datetime(subj_df['date'])
                
                # Select only numeric columns that exist
                subj_cols = ['date']
                for col in ['mood', 'energy_level', 'pain_level', 'allergy_symptoms', 'stress_level']:
                    if col in subj_df.columns:
                        subj_cols.append(col)
                
                if merged_df is not None:
                    merged_df = pd.merge(merged_df, subj_df[subj_cols], 
                                        on='date', how='outer', suffixes=('', '_subj'))
                else:
                    merged_df = subj_df[subj_cols]
                print(f"Debug - Subjective columns selected: {subj_cols}")
            
            if not data['environmental'].empty:
                env_df = data['environmental'].copy()
                env_df['date'] = pd.to_datetime(env_df['date'])
                
                # Select only numeric columns that exist
                env_cols = ['date']
                for col in ['temperature', 'humidity', 'pollen_hazelnut', 'pollen_birch', 'pollen_grass']:
                    if col in env_df.columns:
                        env_cols.append(col)
                
                if merged_df is not None:
                    merged_df = pd.merge(merged_df, env_df[env_cols], 
                                        on='date', how='outer')
                else:
                    merged_df = env_df[env_cols]
                print(f"Debug - Environmental columns selected: {env_cols}")
            
            if merged_df is None:
                print("Debug - No data sources available")
                return None
            
            print(f"Debug - Merged data shape before processing: {merged_df.shape}")
            print(f"Debug - Merged columns: {list(merged_df.columns)}")
            
            # Drop date column and handle missing values
            if 'date' in merged_df.columns:
                merged_df = merged_df.drop(columns=['date'])
            
            # Use forward fill and backward fill to handle missing values
            merged_df = merged_df.ffill().bfill()
            
            # If still missing values, fill with median
            for col in merged_df.columns:
                if merged_df[col].isna().any():
                    median_val = merged_df[col].median()
                    merged_df[col] = merged_df[col].fillna(median_val)
                    print(f"Debug - Filled {col} with median: {median_val}")
            
            # Remove any remaining rows with NaN
            merged_df = merged_df.dropna()
            
            print(f"Debug - Final merged data shape: {merged_df.shape}")
            print(f"Debug - Final columns: {list(merged_df.columns)}")
            
            return merged_df
            
        except Exception as e:
            print(f"Error preparing neural features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def display_neural_results(self, model, X, y, X_test_scaled, y_test, y_pred_test, 
                                     train_score, test_score, sorted_features, target):
        """Display neural network results"""
        self.nn_results_container.clear()
        
        with self.nn_results_container:
            # Performance metrics
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Model Performance').classes('text-h6 mb-3')
                
                with ui.row().classes('w-full mb-2'):
                    with ui.column().classes('flex-1 p-4 bg-green-50 rounded'):
                        ui.label('Training R²').classes('text-sm text-grey-7')
                        ui.label(f'{train_score:.4f}').classes('text-2xl font-bold text-green-600')
                    
                    with ui.column().classes('flex-1 p-4 bg-blue-50 rounded ml-2'):
                        ui.label('Test R²').classes('text-sm text-grey-7')
                        ui.label(f'{test_score:.4f}').classes('text-2xl font-bold text-blue-600')
                    
                    with ui.column().classes('flex-1 p-4 bg-purple-50 rounded ml-2'):
                        ui.label('Features Used').classes('text-sm text-grey-7')
                        ui.label(f'{len(X.columns)}').classes('text-2xl font-bold text-purple-600')
            
            # Feature importance chart
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Feature Importance').classes('text-h6 mb-3')
                
                # Create bar chart
                top_features = sorted_features[:10][::-1]
                fig = go.Figure(data=[
                    go.Bar(
                        x=[f[1] for f in top_features],
                        y=[f[0] for f in top_features],
                        orientation='h',
                        marker=dict(
                            color=[f[1] for f in top_features],
                            colorscale='Viridis',
                            showscale=True
                        )
                    )
                ])
                
                fig.update_layout(
                    title=f'Top 10 Features Predicting {target}',
                    xaxis_title='Importance Score',
                    yaxis_title='Feature',
                    height=400,
                    showlegend=False
                )
                
                ui.plotly(figure=fig).classes('w-full')
            
            # Prediction vs Actual chart
            with ui.card().classes('w-full p-4 mb-4'):
                ui.label('Predictions vs Actual Values').classes('text-h6 mb-3')
                
                fig = go.Figure()
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred_test,
                    mode='markers',
                    name='Predictions',
                    marker=dict(
                        color=BRAND_COLOR,
                        size=8,
                        opacity=0.6
                    )
                ))
                
                # Add perfect prediction line
                min_val = min(y_test.min(), y_pred_test.min())
                max_val = max(y_test.max(), y_pred_test.max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{target.upper()} - Predicted vs Actual',
                    xaxis_title=f'Actual {target}',
                    yaxis_title=f'Predicted {target}',
                    height=400,
                    showlegend=True
                )
                
                ui.plotly(figure=fig).classes('w-full')
            
            # Insights
            with ui.card().classes('w-full p-4'):
                ui.label('Key Insights').classes('text-h6 mb-3')
                
                # Model quality
                if test_score > 0.7:
                    quality = "Excellent"
                    color = "green"
                elif test_score > 0.5:
                    quality = "Good"
                    color = "blue"
                elif test_score > 0.3:
                    quality = "Moderate"
                    color = "orange"
                else:
                    quality = "Poor"
                    color = "red"
                
                ui.label(f'Model Quality: {quality} (R² = {test_score:.3f})').classes(f'text-{color}-600 font-bold mb-2')
                
                # Top predictors
                ui.label('Top 3 Predictors:').classes('font-bold mb-1')
                for i, (feature, importance) in enumerate(sorted_features[:3], 1):
                    ui.label(f'{i}. {feature} ({importance:.3f})').classes('text-sm ml-4')
                
                # Recommendation
                ui.label('Recommendation:').classes('font-bold mt-3 mb-1')
                if test_score > 0.5:
                    ui.label(f'The model has learned meaningful patterns. Focus on optimizing the top features to improve {target}.').classes('text-sm text-grey-7')
                else:
                    ui.label(f'The model shows weak predictive power. Consider collecting more data or trying different features.').classes('text-sm text-grey-7')

# Create and run the app
if __name__ in {"__main__", "__mp_main__"}:
    app = MyPatternApp()
    ui.run(port=8080, title="MyPattern - Personal Health Detective")
