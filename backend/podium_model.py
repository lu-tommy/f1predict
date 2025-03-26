"""
Specialized F1 Podium Prediction Model
Focuses specifically on accurately predicting podium positions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("FastF1 not available. Install with: pip install fastf1")

# Try to import from existing modules
try:
    from f1predict import (
        load_session_safely, get_qualifying_data, get_race_data,
        get_driver_team_mapping, improved_predict_race_winner
    )
    from track_database import get_track_data, get_optimized_track_weights
except ImportError as e:
    print(f"Error importing from existing modules: {e}")
    print("Using simplified implementations")
    # Simplified implementations would go here if needed

class PodiumPredictionModel:
    """
    Specialized model focused on accurately predicting podium positions
    Uses a combination of ML and expert-driven features
    """
    def __init__(self, model_dir="podium_model"):
        """Initialize the podium prediction model"""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Track historical data
        self.driver_track_affinity = {}  # Driver performance at specific tracks
        self.driver_form = {}  # Recent form/momentum
        self.weather_performance = {}  # Performance in different weather conditions
        
        # Cache for fastf1 data
        if FASTF1_AVAILABLE:
            cache_dir = 'f1_cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            fastf1.Cache.enable_cache(cache_dir)
    
    def build_driver_track_affinity(self, years_back=3):
        """
        Build a database of driver performances at specific tracks
        
        Parameters:
        - years_back: How many years of historical data to use
        """
        print("\nBuilding driver-track affinity database...")
        
        current_year = datetime.now().year
        
        # Initialize tracking
        self.driver_track_affinity = {}
        tracks_processed = 0
        years_processed = []
        
        # Process multiple years of data
        for year in range(current_year - years_back, current_year):
            try:
                print(f"Processing {year} season data...")
                
                # Get race schedule for the year
                schedule = fastf1.get_event_schedule(year)
                
                for idx, event in schedule.iterrows():
                    race_name = event['EventName']
                    
                    try:
                        # Load race session
                        race_session = load_session_safely(year, race_name, 'R')
                        
                        if race_session is None:
                            continue
                        
                        # Get qualifying data for comparison
                        quali_session = load_session_safely(year, race_name, 'Q')
                        
                        if quali_session is None:
                            continue
                        
                        # Get race results
                        race_data = get_race_data(race_session)
                        quali_data, _ = get_qualifying_data(quali_session)
                        
                        # Process each driver's performance
                        for driver, data in race_data.items():
                            if driver not in self.driver_track_affinity:
                                self.driver_track_affinity[driver] = {}
                            
                            if race_name not in self.driver_track_affinity[driver]:
                                self.driver_track_affinity[driver][race_name] = []
                            
                            # Calculate performance metrics
                            race_position = data.get('position')
                            quali_position = quali_data.get(driver, {}).get('position')
                            
                            if race_position is None or quali_position is None:
                                continue
                            
                            # Calculate podium performance (1 if podium, 0 if not)
                            podium_result = 1 if race_position <= 3 else 0
                            
                            # Position improvement from qualifying
                            position_improvement = quali_position - race_position
                            
                            # Store performance metrics
                            performance = {
                                'year': year,
                                'race_position': race_position,
                                'quali_position': quali_position,
                                'position_improvement': position_improvement,
                                'podium_result': podium_result
                            }
                            
                            self.driver_track_affinity[driver][race_name].append(performance)
                        
                        tracks_processed += 1
                        print(f"Processed {race_name} {year}")
                    
                    except Exception as e:
                        print(f"Error processing {race_name} {year}: {e}")
                        continue
                
                years_processed.append(year)
                
            except Exception as e:
                print(f"Error processing {year} season: {e}")
                continue
        
        print(f"\nDriver-track affinity database built.")
        print(f"Processed {tracks_processed} races across {len(years_processed)} seasons.")
        print(f"Tracking {len(self.driver_track_affinity)} drivers.")
        
        # Save the data
        self.save_driver_track_affinity()
    
    def save_driver_track_affinity(self):
        """Save driver-track affinity data to file"""
        try:
            filename = os.path.join(self.model_dir, 'driver_track_affinity.json')
            with open(filename, 'w') as f:
                json.dump(self.driver_track_affinity, f, indent=2)
            print(f"Saved driver-track affinity data to {filename}")
        except Exception as e:
            print(f"Error saving driver-track affinity data: {e}")
    
    def load_driver_track_affinity(self):
        """Load driver-track affinity data from file"""
        try:
            filename = os.path.join(self.model_dir, 'driver_track_affinity.json')
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.driver_track_affinity = json.load(f)
                print(f"Loaded driver-track affinity data from {filename}")
                return True
            return False
        except Exception as e:
            print(f"Error loading driver-track affinity data: {e}")
            return False
    
    def calculate_driver_track_affinity(self, driver, race_name):
        """
        Calculate a driver's affinity score for a specific track
        
        Parameters:
        - driver: Driver code
        - race_name: Name of the race/track
        
        Returns:
        - affinity_score: 0-1 score of driver's affinity for this track
        - podium_probability: Estimated probability of podium finish
        """
        # Default values if no data available
        affinity_score = 0.5
        podium_probability = 0.1  # Base 10% probability
        
        # Check if we have data for this driver and track
        if driver not in self.driver_track_affinity:
            return affinity_score, podium_probability
        
        if race_name not in self.driver_track_affinity[driver]:
            # Try to find similar tracks
            similar_tracks = self.find_similar_tracks(race_name)
            
            # Calculate average performances at similar tracks
            if similar_tracks:
                similar_track_performances = []
                similar_track_podiums = []
                
                for track in similar_tracks:
                    if track in self.driver_track_affinity[driver]:
                        track_data = self.driver_track_affinity[driver][track]
                        
                        # Calculate average position
                        positions = [data['race_position'] for data in track_data]
                        if positions:
                            avg_position = sum(positions) / len(positions)
                            similar_track_performances.append(avg_position)
                        
                        # Calculate podium rate
                        podiums = [data['podium_result'] for data in track_data]
                        if podiums:
                            podium_rate = sum(podiums) / len(podiums)
                            similar_track_podiums.append(podium_rate)
                
                # Calculate affinity from similar tracks
                if similar_track_performances:
                    # Convert to 0-1 score (1 = 1st position, 0 = 20th position)
                    avg_similar_position = sum(similar_track_performances) / len(similar_track_performances)
                    affinity_score = max(0, 1 - (avg_similar_position / 20))
                
                # Calculate podium probability from similar tracks
                if similar_track_podiums:
                    podium_probability = sum(similar_track_podiums) / len(similar_track_podiums)
            
            return affinity_score, podium_probability
        
        # We have historical data for this driver at this track
        track_data = self.driver_track_affinity[driver][race_name]
        
        # Calculate performance metrics
        positions = [data['race_position'] for data in track_data]
        podiums = [data['podium_result'] for data in track_data]
        
        if not positions:
            return affinity_score, podium_probability
        
        # Calculate average position and podium rate with more weight on recent performances
        weighted_positions = 0
        weighted_podiums = 0
        total_weight = 0
        
        current_year = datetime.now().year
        
        for i, data in enumerate(track_data):
            year = data['year']
            # More recent = higher weight (2x weight per year)
            weight = 2 ** (year - (current_year - 3))
            weighted_positions += data['race_position'] * weight
            weighted_podiums += data['podium_result'] * weight
            total_weight += weight
        
        avg_position = weighted_positions / total_weight if total_weight > 0 else None
        podium_rate = weighted_podiums / total_weight if total_weight > 0 else None
        
        # Convert position to affinity score (0-1)
        if avg_position is not None:
            # 1 = 1st position, 0 = 20th position
            affinity_score = max(0, 1 - (avg_position / 20))
        
        # Use historical podium rate as podium probability
        if podium_rate is not None:
            podium_probability = podium_rate
        
        return affinity_score, podium_probability
    
    def find_similar_tracks(self, race_name):
        """
        Find tracks that are similar to the given race
        
        Parameters:
        - race_name: Name of the race/track
        
        Returns:
        - similar_tracks: List of similar track names
        """
        # Get track data
        track_data = get_track_data(race_name)
        
        if not track_data:
            return []
        
        # Get track type and characteristics
        track_type = track_data.get('track_type', 'unknown')
        
        # Find tracks with the same type
        similar_tracks = []
        
        # Import all predefined tracks
        from track_database import TRACK_DATABASE
        
        for track, data in TRACK_DATABASE.items():
            if data.get('track_type') == track_type and track != race_name:
                # Calculate similarity score based on characteristics
                similarity = self.calculate_track_similarity(track_data, data)
                
                if similarity > 0.7:  # Only include highly similar tracks
                    similar_tracks.append(track)
        
        return similar_tracks
    
    def calculate_track_similarity(self, track1_data, track2_data):
        """
        Calculate similarity between two tracks
        
        Parameters:
        - track1_data: Data for first track
        - track2_data: Data for second track
        
        Returns:
        - similarity: 0-1 similarity score
        """
        # Key characteristics to compare
        characteristics = [
            'overtaking_difficulty',
            'tire_degradation',
            'qualifying_importance',
            'start_importance',
            'dirty_air_impact'
        ]
        
        # Calculate similarity based on characteristics
        diff_sum = 0
        comparisons = 0
        
        for char in characteristics:
            if char in track1_data and char in track2_data:
                diff = abs(track1_data[char] - track2_data[char])
                diff_sum += diff
                comparisons += 1
        
        if comparisons == 0:
            return 0
        
        # Convert to similarity score (0-1)
        avg_diff = diff_sum / comparisons
        similarity = 1 - avg_diff
        
        return similarity
    
    def build_driver_form(self, num_races=5):
        """
        Build a database of driver recent form/momentum
        
        Parameters:
        - num_races: Number of recent races to consider
        """
        print("\nBuilding driver form database...")
        
        current_year = datetime.now().year
        
        # Initialize tracking
        self.driver_form = {}
        
        try:
            # Get current season schedule
            schedule = fastf1.get_event_schedule(current_year)
            
            # Find current race index
            current_date = datetime.now()
            current_race_idx = None
            
            for i, event in schedule.iterrows():
                race_date = event['EventDate']
                
                if race_date > current_date:
                    current_race_idx = i - 1  # Previous race
                    break
            
            if current_race_idx is None:
                current_race_idx = len(schedule) - 1  # Last race of the season
            
            # Get previous races in reverse order (most recent first)
            recent_races = []
            
            for i in range(current_race_idx, max(-1, current_race_idx - num_races), -1):
                if i >= 0 and i < len(schedule):
                    recent_races.append(schedule.iloc[i]['EventName'])
            
            print(f"Analyzing form based on these recent races: {recent_races}")
            
            # Process each recent race
            for i, race_name in enumerate(recent_races):
                # Weight factor (most recent races have higher weight)
                weight = (num_races - i) / num_races
                
                try:
                    # Load race session
                    race_session = load_session_safely(current_year, race_name, 'R')
                    
                    if race_session is None:
                        continue
                    
                    # Get race results
                    race_data = get_race_data(race_session)
                    
                    # Process each driver's performance
                    for driver, data in race_data.items():
                        if driver not in self.driver_form:
                            self.driver_form[driver] = {
                                'momentum_score': 0,
                                'podium_rate': 0,
                                'recent_races': []
                            }
                        
                        race_position = data.get('position')
                        
                        if race_position is None:
                            continue
                        
                        # Calculate position score (0-1)
                        position_score = max(0, 1 - (race_position / 20))
                        
                        # Calculate podium result
                        podium_result = 1 if race_position <= 3 else 0
                        
                        # Add weighted result to momentum score
                        self.driver_form[driver]['momentum_score'] += position_score * weight
                        
                        # Track podium results
                        self.driver_form[driver]['recent_races'].append({
                            'race': race_name,
                            'position': race_position,
                            'podium': podium_result,
                            'weight': weight
                        })
                
                except Exception as e:
                    print(f"Error processing {race_name} for form calculation: {e}")
                    continue
            
            # Normalize momentum scores and calculate podium rates
            for driver in self.driver_form:
                # Normalize momentum score to 0-1 range
                weight_sum = sum(race['weight'] for race in self.driver_form[driver]['recent_races'])
                
                if weight_sum > 0:
                    self.driver_form[driver]['momentum_score'] /= weight_sum
                
                # Calculate weighted podium rate
                if self.driver_form[driver]['recent_races']:
                    weighted_podiums = sum(race['podium'] * race['weight'] for race in self.driver_form[driver]['recent_races'])
                    self.driver_form[driver]['podium_rate'] = weighted_podiums / weight_sum
            
            print(f"\nDriver form database built.")
            print(f"Tracking form for {len(self.driver_form)} drivers.")
            
            # Save the data
            self.save_driver_form()
        
        except Exception as e:
            print(f"Error building driver form database: {e}")
    
    def save_driver_form(self):
        """Save driver form data to file"""
        try:
            filename = os.path.join(self.model_dir, 'driver_form.json')
            with open(filename, 'w') as f:
                json.dump(self.driver_form, f, indent=2)
            print(f"Saved driver form data to {filename}")
        except Exception as e:
            print(f"Error saving driver form data: {e}")
    
    def load_driver_form(self):
        """Load driver form data from file"""
        try:
            filename = os.path.join(self.model_dir, 'driver_form.json')
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.driver_form = json.load(f)
                print(f"Loaded driver form data from {filename}")
                return True
            return False
        except Exception as e:
            print(f"Error loading driver form data: {e}")
            return False
    
    def calculate_driver_form_factor(self, driver):
        """
        Calculate a driver's current form factor
        
        Parameters:
        - driver: Driver code
        
        Returns:
        - form_factor: 0-1 score of driver's current form
        - recent_podium_rate: Rate of recent podium finishes
        """
        # Default values if no data available
        form_factor = 0.5
        recent_podium_rate = 0.1  # Base 10% podium rate
        
        # Check if we have form data for this driver
        if driver not in self.driver_form:
            return form_factor, recent_podium_rate
        
        # Get form data
        form_data = self.driver_form[driver]
        
        # Return calculated values
        form_factor = form_data.get('momentum_score', form_factor)
        recent_podium_rate = form_data.get('podium_rate', recent_podium_rate)
        
        return form_factor, recent_podium_rate
    
    def build_weather_performance(self, years_back=2):
        """
        Build a database of driver performances in different weather conditions
        
        Parameters:
        - years_back: How many years of historical data to use
        """
        print("\nBuilding weather performance database...")
        
        current_year = datetime.now().year
        
        # Initialize tracking
        self.weather_performance = {}
        races_processed = 0
        years_processed = []
        
        # Process multiple years of data
        for year in range(current_year - years_back, current_year):
            try:
                print(f"Processing {year} weather performance data...")
                
                # Get race schedule for the year
                schedule = fastf1.get_event_schedule(year)
                
                for idx, event in schedule.iterrows():
                    race_name = event['EventName']
                    
                    try:
                        # Load race session
                        race_session = load_session_safely(year, race_name, 'R')
                        
                        if race_session is None:
                            continue
                        
                        # Get race results
                        race_data = get_race_data(race_session)
                        
                        # Get weather data if available
                        weather_data = None
                        weather_condition = 'dry'  # Default
                        
                        if hasattr(race_session, 'weather_data') and race_session.weather_data is not None:
                            weather_data = race_session.weather_data
                            
                            if not weather_data.empty:
                                if 'Rainfall' in weather_data.columns:
                                    # Check if it was a wet race
                                    rainfall = weather_data['Rainfall'].max()
                                    if rainfall > 0:
                                        weather_condition = 'wet'
                        
                        # Process each driver's performance
                        for driver, data in race_data.items():
                            if driver not in self.weather_performance:
                                self.weather_performance[driver] = {
                                    'dry': {'races': [], 'podiums': 0, 'avg_position': 0},
                                    'wet': {'races': [], 'podiums': 0, 'avg_position': 0}
                                }
                            
                            race_position = data.get('position')
                            
                            if race_position is None:
                                continue
                            
                            # Record performance in appropriate weather condition
                            self.weather_performance[driver][weather_condition]['races'].append({
                                'year': year,
                                'race': race_name,
                                'position': race_position
                            })
                            
                            # Update podium count
                            if race_position <= 3:
                                self.weather_performance[driver][weather_condition]['podiums'] += 1
                        
                        races_processed += 1
                        print(f"Processed {race_name} {year} - Weather: {weather_condition}")
                    
                    except Exception as e:
                        print(f"Error processing {race_name} {year} weather data: {e}")
                        continue
                
                years_processed.append(year)
                
            except Exception as e:
                print(f"Error processing {year} season weather data: {e}")
                continue
        
        # Calculate average positions
        for driver in self.weather_performance:
            for condition in ['dry', 'wet']:
                races = self.weather_performance[driver][condition]['races']
                if races:
                    positions = [race['position'] for race in races]
                    self.weather_performance[driver][condition]['avg_position'] = sum(positions) / len(positions)
        
        print(f"\nWeather performance database built.")
        print(f"Processed {races_processed} races across {len(years_processed)} seasons.")
        print(f"Tracking weather performance for {len(self.weather_performance)} drivers.")
        
        # Save the data
        self.save_weather_performance()
    
    def save_weather_performance(self):
        """Save weather performance data to file"""
        try:
            filename = os.path.join(self.model_dir, 'weather_performance.json')
            with open(filename, 'w') as f:
                json.dump(self.weather_performance, f, indent=2)
            print(f"Saved weather performance data to {filename}")
        except Exception as e:
            print(f"Error saving weather performance data: {e}")
    
    def load_weather_performance(self):
        """Load weather performance data from file"""
        try:
            filename = os.path.join(self.model_dir, 'weather_performance.json')
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.weather_performance = json.load(f)
                print(f"Loaded weather performance data from {filename}")
                return True
            return False
        except Exception as e:
            print(f"Error loading weather performance data: {e}")
            return False
    
    def get_weather_performance_factor(self, driver, expected_weather='dry'):
        """
        Get a driver's performance factor for specific weather conditions
        
        Parameters:
        - driver: Driver code
        - expected_weather: Expected weather conditions ('dry' or 'wet')
        
        Returns:
        - performance_factor: 0-1 score of driver's performance in these conditions
        - podium_probability: Probability of podium in these conditions
        """
        # Default values if no data available
        performance_factor = 0.5
        podium_probability = 0.1  # Base 10% probability
        
        # Check if we have data for this driver
        if driver not in self.weather_performance:
            return performance_factor, podium_probability
        
        # Get weather performance data
        weather_data = self.weather_performance[driver][expected_weather]
        
        # Calculate performance factor based on average position
        avg_position = weather_data.get('avg_position')
        if avg_position:
            # Convert to 0-1 score (1 = 1st position, 0 = 20th position)
            performance_factor = max(0, 1 - (avg_position / 20))
        
        # Calculate podium probability
        races = len(weather_data.get('races', []))
        podiums = weather_data.get('podiums', 0)
        
        if races > 0:
            podium_probability = podiums / races
        
        return performance_factor, podium_probability
    
    def get_weather_forecast(self, race_name, year=None):
        """
        Get weather forecast for a race
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race (default: current year)
        
        Returns:
        - weather_condition: Expected weather condition ('dry' or 'wet')
        - precipitation_probability: Probability of precipitation
        """
        # Default values
        weather_condition = 'dry'
        precipitation_probability = 0.0
        
        if year is None:
            year = datetime.now().year
        
        # In a real implementation, this would fetch weather forecasts
        # from an API. For now, we'll return default values.
        
        return weather_condition, precipitation_probability
    
    def extract_podium_features(self, race_name, year=None):
        """
        Extract features specifically for podium prediction
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race (default: current year)
        
        Returns:
        - podium_features: DataFrame with features for podium prediction
        """
        if year is None:
            year = datetime.now().year
        
        print(f"\nExtracting podium features for {race_name} {year}...")
        
        # Load qualifying session
        quali_session = load_session_safely(year, race_name, 'Q')
        
        if quali_session is None:
            print(f"Qualifying data not available for {race_name} {year}")
            return None
        
        # Get qualifying data
        quali_data, sector_data = get_qualifying_data(quali_session)
        
        # Get driver-team mapping
        driver_teams = get_driver_team_mapping(quali_session)
        
        # Get track data
        track_data = get_track_data(race_name)
        
        # Get weather forecast
        expected_weather, precipitation_prob = self.get_weather_forecast(race_name, year)
        
        # Prepare feature records
        feature_records = []
        
        for driver, quali_info in quali_data.items():
            # Skip if incomplete data
            if 'position' not in quali_info:
                continue
            
            # Base features from qualifying
            quali_position = quali_info['position']
            
            # Create feature record
            record = {
                'driver': driver,
                'race_name': race_name,
                'year': year,
                'team': driver_teams.get(driver, 'Unknown'),
                'quali_position': quali_position,
                
                # Start position categories (important for podium chances)
                'front_row': 1 if quali_position <= 2 else 0,
                'top3_start': 1 if quali_position <= 3 else 0,
                'top5_start': 1 if quali_position <= 5 else 0,
                'top10_start': 1 if quali_position <= 10 else 0,
                
                # Track-specific features
                'track_type': track_data.get('track_type', 'unknown') if track_data else 'unknown',
                'overtaking_difficulty': track_data.get('overtaking_difficulty', 0.6) if track_data else 0.6,
                'qualifying_importance': track_data.get('qualifying_importance', 0.7) if track_data else 0.7,
                
                # Weather features
                'wet_race_expected': 1 if expected_weather == 'wet' else 0,
                'precipitation_probability': precipitation_prob
            }
            
            # Add driver-track affinity
            track_affinity, track_podium_prob = self.calculate_driver_track_affinity(driver, race_name)
            record['track_affinity'] = track_affinity
            record['historical_track_podium_rate'] = track_podium_prob
            
            # Add driver form factor
            form_factor, recent_podium_rate = self.calculate_driver_form_factor(driver)
            record['form_factor'] = form_factor
            record['recent_podium_rate'] = recent_podium_rate
            
            # Add weather performance
            weather_perf, weather_podium_prob = self.get_weather_performance_factor(driver, expected_weather)
            record['weather_performance'] = weather_perf
            record['weather_podium_probability'] = weather_podium_prob
            
            # Add to records
            feature_records.append(record)
        
        # Convert to DataFrame
        if not feature_records:
            print("No feature records extracted")
            return None
        
        podium_features = pd.DataFrame(feature_records)
        print(f"Extracted features for {len(podium_features)} drivers")
        
        return podium_features
    
    def train_podium_model(self, years_back=2, races_to_include=None):
        """
        Train a specialized model for podium prediction
        
        Parameters:
        - years_back: How many years of historical data to use
        - races_to_include: List of specific races to include (None = all races)
        
        Returns:
        - True if successful, False otherwise
        """
        print("\nTraining specialized podium prediction model...")
        
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Cannot train model.")
            return False
        
        # Ensure we have the necessary databases
        if not self.driver_track_affinity:
            loaded = self.load_driver_track_affinity()
            if not loaded:
                print("Building driver-track affinity database...")
                self.build_driver_track_affinity(years_back=years_back)
        
        if not self.driver_form:
            loaded = self.load_driver_form()
            if not loaded:
                print("Building driver form database...")
                self.build_driver_form()
        
        if not self.weather_performance:
            loaded = self.load_weather_performance()
            if not loaded:
                print("Building weather performance database...")
                self.build_weather_performance(years_back=years_back)
        
        # Collect training data
        training_data = []
        current_year = datetime.now().year
        
        for year in range(current_year - years_back, current_year):
            try:
                print(f"Collecting training data from {year}...")
                
                # Get race schedule for the year
                schedule = fastf1.get_event_schedule(year)
                
                if races_to_include:
                    # Filter to specified races
                    races = [race for race in schedule['EventName'].tolist() if race in races_to_include]
                else:
                    races = schedule['EventName'].tolist()
                
                for race_name in races:
                    try:
                        # Get features for this race
                        race_features = self.extract_podium_features(race_name, year)
                        
                        if race_features is None:
                            continue
                        
                        # Get actual race results for labels
                        race_session = load_session_safely(year, race_name, 'R')
                        
                        if race_session is None:
                            continue
                        
                        race_data = get_race_data(race_session)
                        
                        # Add podium labels (1 if podium, 0 if not)
                        podium_labels = []
                        
                        for _, row in race_features.iterrows():
                            driver = row['driver']
                            
                            if driver in race_data:
                                position = race_data[driver].get('position')
                                podium_result = 1 if position is not None and position <= 3 else 0
                                podium_labels.append(podium_result)
                            else:
                                # Driver did not finish or not in results
                                podium_labels.append(0)
                        
                        race_features['podium_result'] = podium_labels
                        
                        # Add to training data
                        training_data.append(race_features)
                        
                        print(f"Added {race_name} {year} to training data")
                    
                    except Exception as e:
                        print(f"Error processing {race_name} {year} for training: {e}")
                        continue
            
            except Exception as e:
                print(f"Error processing {year} season for training: {e}")
                continue
        
        # Combine all training data
        if not training_data:
            print("No training data collected")
            return False
        
        training_df = pd.concat(training_data, ignore_index=True)
        print(f"Collected {len(training_df)} samples for training")
        
        # Prepare features and labels
        X = training_df.drop(['driver', 'race_name', 'year', 'team', 'track_type', 'podium_result'], axis=1)
        y = training_df['podium_result']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define specialized objective function for podium positions
        def podium_focused_objective(y_true, y_pred):
            # Convert predictions to probabilities
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
            
            # Calculate gradient and hessian
            grad = y_pred - y_true
            hess = y_pred * (1.0 - y_pred)
            
            # Apply higher weight to positive examples (podium finishes)
            pos_weight = 3.0  # Weight for podium positions
            
            grad = np.where(y_true > 0, grad * pos_weight, grad)
            hess = np.where(y_true > 0, hess * pos_weight, hess)
            
            return grad, hess
        
        # Train model with cross-validation for best parameters
        if len(X_train) > 50:  # Only if we have enough data
            print("Performing parameter tuning...")
            param_grid = {
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            }
            
            grid_search = GridSearchCV(
                estimator=xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    use_label_encoder=False,
                    random_state=42
                ),
                param_grid=param_grid,
                scoring='roc_auc',
                cv=3,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Train final model with best parameters
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42,
                **best_params
            )
        else:
            # Not enough data for grid search, use default parameters
            print("Using default parameters (not enough data for tuning)")
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42
            )
        
        # Train the model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=10,
            verbose=True
        )
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\nPodium Prediction Model Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save model
        self.save_model()
        
        return True
    
    def save_model(self):
        """Save trained model and components to disk"""
        try:
            # Save model
            model_path = os.path.join(self.model_dir, 'podium_model.json')
            self.model.save_model(model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            from joblib import dump
            dump(self.scaler, scaler_path)
            
            # Save feature names
            feature_path = os.path.join(self.model_dir, 'feature_names.json')
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f)
            
            print(f"Model and components saved to {self.model_dir}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load model and components from disk"""
        try:
            # Check if model exists
            model_path = os.path.join(self.model_dir, 'podium_model.json')
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
            
            # Load model
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
            if os.path.exists(scaler_path):
                from joblib import load
                self.scaler = load(scaler_path)
            
            # Load feature names
            feature_path = os.path.join(self.model_dir, 'feature_names.json')
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
            
            print(f"Model and components loaded from {self.model_dir}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_podium(self, race_name, year=None):
        """
        Predict podium finishers for a race
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race (default: current year)
        
        Returns:
        - podium_predictions: List of drivers predicted to finish on podium
        - all_predictions: Dictionary with podium probabilities for all drivers
        """
        if year is None:
            year = datetime.now().year
        
        print(f"\nPredicting podium for {race_name} {year}...")
        
        # Extract features
        features_df = self.extract_podium_features(race_name, year)
        
        if features_df is None:
            print("Could not extract features")
            return None, {}
        
        # Check if model is loaded
        if self.model is None:
            loaded = self.load_model()
            if not loaded:
                print("Could not load model. Using alternative prediction method.")
                return self.predict_podium_fallback(race_name, year, features_df)
        
        # Prepare features for prediction
        X = features_df.drop(['driver', 'race_name', 'year', 'team', 'track_type'], axis=1)
        
        # Ensure we have all required features
        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        # Keep only features known to the model
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        podium_probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Combine with driver info
        predictions = {}
        for i, (_, row) in enumerate(features_df.iterrows()):
            driver = row['driver']
            predictions[driver] = float(podium_probs[i])
        
        # Sort by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 as predicted podium
        podium_predictions = [driver for driver, _ in sorted_predictions[:3]]
        
        return podium_predictions, predictions
    
    def predict_podium_fallback(self, race_name, year, features_df=None):
        """
        Fallback method for podium prediction when model is not available
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race
        - features_df: Features DataFrame (if already extracted)
        
        Returns:
        - podium_predictions: List of drivers predicted to finish on podium
        - all_predictions: Dictionary with podium probabilities for all drivers
        """
        print("Using fallback method for podium prediction...")
        
        # If features not provided, extract them
        if features_df is None:
            features_df = self.extract_podium_features(race_name, year)
            
            if features_df is None:
                print("Could not extract features")
                return None, {}
        
        # Calculate podium scores based on features
        predictions = {}
        
        for _, row in features_df.iterrows():
            driver = row['driver']
            
            # Calculate score based on various factors
            score = 0.0
            
            # Qualifying position (most important)
            quali_weight = row['qualifying_importance']  # Track-specific importance
            quali_pos = row['quali_position']
            quali_score = max(0, 1 - ((quali_pos - 1) / 20))  # 1st: 1.0, 20th: 0.0
            score += quali_score * quali_weight * 0.4
            
            # Front row bonus
            if row['front_row'] == 1:
                score += 0.15
            
            # Driver's track affinity
            score += row['track_affinity'] * 0.15
            
            # Historical podium rate at this track
            score += row['historical_track_podium_rate'] * 0.1
            
            # Recent form
            score += row['form_factor'] * 0.15
            
            # Weather performance (if applicable)
            if row['wet_race_expected'] == 1:
                score += row['weather_performance'] * 0.1
            
            # Scale to 0-1 range
            predictions[driver] = min(1.0, max(0.0, score))
        
        # Sort by score
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 3 as predicted podium
        podium_predictions = [driver for driver, _ in sorted_predictions[:3]]
        
        return podium_predictions, predictions
    
    def visualize_podium_prediction(self, race_name, year, predictions):
        """
        Create visualization of podium predictions
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race
        - predictions: Dictionary of predicted podium probabilities
        
        Returns:
        - figure: Matplotlib figure
        """
        # Sort predictions by probability
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Get top drivers (maximum 10 for visibility)
        top_n = min(10, len(sorted_predictions))
        top_drivers = [driver for driver, _ in sorted_predictions[:top_n]]
        probs = [predictions[driver] for driver in top_drivers]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create bar colors (top 3 in special colors)
        colors = ['gold', 'silver', '#CD7F32'] + ['steelblue'] * (top_n - 3)
        
        plt.bar(top_drivers, probs, color=colors)
        plt.xlabel('Driver')
        plt.ylabel('Podium Probability')
        plt.title(f'F1 Podium Prediction: {race_name} {year}')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add probability values
        for i, prob in enumerate(probs):
            plt.text(i, prob + 0.02, f"{prob:.3f}", ha='center')
        
        plt.tight_layout()
        
        return plt.gcf()

def predict_race_with_podium_focus(race_name, year=None):
    """
    Predict race results with special focus on accurate podium prediction
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    
    Returns:
    - podium_predictions: List of drivers predicted to finish on podium
    - position_predictions: Dictionary with predicted positions for all drivers
    """
    if year is None:
        year = datetime.now().year
    
    print(f"\nPredicting race results with podium focus: {race_name} {year}")
    
    # 1. Get standard prediction
    print("\nStep 1: Getting baseline prediction...")
    try:
        optimized_weights = get_optimized_track_weights(race_name, year)
        _, baseline_predictions = improved_predict_race_winner(race_name, optimized_weights)
    except Exception as e:
        print(f"Error in baseline prediction: {e}")
        print("Using default weights")
        _, baseline_predictions = improved_predict_race_winner(race_name)
    
    # 2. Get specialized podium prediction
    print("\nStep 2: Getting specialized podium prediction...")
    podium_model = PodiumPredictionModel()
    
    # Check if model is already trained
    if not podium_model.load_model():
        print("Training podium model...")
        podium_model.train_podium_model(years_back=2)
    
    # Make podium prediction
    podium_drivers, podium_probabilities = podium_model.predict_podium(race_name, year)
    
    if not podium_drivers:
        print("Could not get podium prediction. Using baseline only.")
        # Return baseline prediction
        baseline_sorted = sorted(baseline_predictions.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in baseline_sorted[:3]], baseline_predictions
    
    # 3. Create ensemble prediction
    print("\nStep 3: Creating ensemble prediction...")
    ensemble_predictions = {}
    
    # Get all drivers from both predictions
    all_drivers = set(baseline_predictions.keys()) | set(podium_probabilities.keys())
    
    for driver in all_drivers:
        # Get baseline position score (0-1)
        baseline_score = baseline_predictions.get(driver, 0)
        
        # Get podium probability
        podium_prob = podium_probabilities.get(driver, 0)
        
        # Calculate ensemble score with weights:
        # - Higher weight for podium model for top positions
        # - Higher weight for baseline model for lower positions
        if baseline_score > 0.7:  # Likely top position
            ensemble_score = (baseline_score * 0.4) + (podium_prob * 0.6)
        else:
            ensemble_score = (baseline_score * 0.7) + (podium_prob * 0.3)
        
        ensemble_predictions[driver] = ensemble_score
    
    # Sort ensemble predictions
    sorted_ensemble = sorted(ensemble_predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Get predicted podium from ensemble
    predicted_podium = [driver for driver, _ in sorted_ensemble[:3]]
    
    print("\nEnsemble Prediction - Podium:")
    for i, driver in enumerate(predicted_podium):
        print(f"{i+1}. {driver} (Score: {ensemble_predictions[driver]:.3f})")
    
    # Create visualization
    try:
        podium_model.visualize_podium_prediction(race_name, year, ensemble_predictions)
        plt.savefig(f"podium_prediction_{race_name.replace(' ', '_')}_{year}.png", dpi=300)
        print(f"Saved visualization to podium_prediction_{race_name.replace(' ', '_')}_{year}.png")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    return predicted_podium, ensemble_predictions

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Podium Prediction')
    parser.add_argument('--race', type=str, help='Race name')
    parser.add_argument('--year', type=int, default=datetime.now().year, help='Year')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--build-data', action='store_true', help='Build databases')
    
    args = parser.parse_args()
    
    podium_model = PodiumPredictionModel()
    
    if args.build_data:
        # Build all databases
        podium_model.build_driver_track_affinity()
        podium_model.build_driver_form()
        podium_model.build_weather_performance()
    
    if args.train:
        # Train model
        podium_model.train_podium_model()
    
    if args.race:
        # Predict podium
        predict_race_with_podium_focus(args.race, args.year)