"""
Weather Integration Module for F1 Prediction
Enhances prediction accuracy by incorporating weather data and driver performance in different conditions
"""

import os
import numpy as np
import pandas as pd
import json
import requests
from datetime import datetime, timedelta
import fastf1

# Initialize a cache for weather data
WEATHER_CACHE = {}
WEATHER_CACHE_FILE = 'weather_cache.json'

# Driver weather performance data
DRIVER_WEATHER_PERFORMANCE = {}
DRIVER_WEATHER_FILE = 'driver_weather_performance.json'

# API Key for weather data (replace with your key in production)
WEATHER_API_KEY = 'YOUR_WEATHER_API_KEY'  # For WeatherAPI.com or similar service

def load_weather_cache():
    """Load weather cache from file if it exists"""
    global WEATHER_CACHE
    
    if os.path.exists(WEATHER_CACHE_FILE):
        try:
            with open(WEATHER_CACHE_FILE, 'r') as f:
                WEATHER_CACHE = json.load(f)
            print(f"Loaded weather cache with {len(WEATHER_CACHE)} entries")
            return True
        except Exception as e:
            print(f"Error loading weather cache: {e}")
    
    return False

def save_weather_cache():
    """Save weather cache to file"""
    try:
        with open(WEATHER_CACHE_FILE, 'w') as f:
            json.dump(WEATHER_CACHE, f)
        print(f"Saved weather cache with {len(WEATHER_CACHE)} entries")
        return True
    except Exception as e:
        print(f"Error saving weather cache: {e}")
        return False

def load_driver_weather_performance():
    """Load driver weather performance data from file if it exists"""
    global DRIVER_WEATHER_PERFORMANCE
    
    if os.path.exists(DRIVER_WEATHER_FILE):
        try:
            with open(DRIVER_WEATHER_FILE, 'r') as f:
                DRIVER_WEATHER_PERFORMANCE = json.load(f)
            print(f"Loaded driver weather performance data for {len(DRIVER_WEATHER_PERFORMANCE)} drivers")
            return True
        except Exception as e:
            print(f"Error loading driver weather performance data: {e}")
    
    return False

def save_driver_weather_performance():
    """Save driver weather performance data to file"""
    try:
        with open(DRIVER_WEATHER_FILE, 'w') as f:
            json.dump(DRIVER_WEATHER_PERFORMANCE, f)
        print(f"Saved weather performance data for {len(DRIVER_WEATHER_PERFORMANCE)} drivers")
        return True
    except Exception as e:
        print(f"Error saving driver weather performance data: {e}")
        return False

def get_track_coordinates(race_name):
    """
    Get geographical coordinates for a race track
    
    Parameters:
    - race_name: Name of the race/track
    
    Returns:
    - latitude, longitude: Coordinates for the track
    """
    # Define known track coordinates
    track_coordinates = {
        'Monaco Grand Prix': (43.7347, 7.4206),
        'British Grand Prix': (52.0706, -1.0174),
        'Italian Grand Prix': (45.6156, 9.2812),
        'Belgian Grand Prix': (50.4372, 5.9705),
        'Australian Grand Prix': (-37.8497, 144.9680),
        'Spanish Grand Prix': (41.5638, 2.2585),
        'Austrian Grand Prix': (47.2197, 14.7647),
        'Hungarian Grand Prix': (47.5830, 19.2526),
        'Dutch Grand Prix': (52.3888, 4.5408),
        'Singapore Grand Prix': (1.2914, 103.8644),
        'Japanese Grand Prix': (34.8431, 136.5407),
        'United States Grand Prix': (30.1328, -97.6411),
        'Mexico City Grand Prix': (19.4042, -99.0907),
        'Brazilian Grand Prix': (-23.7036, -46.6997),
        'Las Vegas Grand Prix': (36.1147, -115.1728),
        'Qatar Grand Prix': (25.4882, 51.4530),
        'Abu Dhabi Grand Prix': (24.4672, 54.6031),
        'Saudi Arabian Grand Prix': (21.6319, 39.1044),
        'Bahrain Grand Prix': (26.0370, 50.5112),
        'Canadian Grand Prix': (45.5016, -73.5222),
        'Miami Grand Prix': (25.9581, -80.2389),
        'Emilia Romagna Grand Prix': (44.3440, 11.7167),
        'Azerbaijan Grand Prix': (40.3724, 49.8533),
    }
    
    # Check for exact match
    if race_name in track_coordinates:
        return track_coordinates[race_name]
    
    # Check for partial match
    for track, coords in track_coordinates.items():
        if race_name.lower() in track.lower() or track.lower() in race_name.lower():
            return coords
    
    # Default coordinates if track not found
    print(f"Track coordinates not found for {race_name}, using default")
    return (0, 0)

def get_race_date(race_name, year):
    """
    Get the date of a race
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race
    
    Returns:
    - race_date: Date of the race or None if not found
    """
    try:
        # Get the event schedule
        schedule = fastf1.get_event_schedule(year)
        
        # Find the race by name
        for idx, event in schedule.iterrows():
            if event['EventName'] == race_name:
                return event['EventDate']
        
        print(f"Race not found in {year} schedule: {race_name}")
        return None
    except Exception as e:
        print(f"Error getting race date: {e}")
        return None

def get_weather_forecast(race_name, year=None):
    """
    Get weather forecast for a race
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    
    Returns:
    - forecast: Dictionary with weather forecast data
    """
    if year is None:
        year = datetime.now().year
    
    # Check if forecast is in cache
    cache_key = f"{race_name}_{year}"
    if cache_key in WEATHER_CACHE:
        # Check if forecast is recent (within 6 hours for upcoming races)
        cached_forecast = WEATHER_CACHE[cache_key]
        last_updated = datetime.fromisoformat(cached_forecast['last_updated'])
        
        race_date = get_race_date(race_name, year)
        
        # If race is in the future and cache is recent, use cached data
        if race_date and race_date > datetime.now() and (datetime.now() - last_updated).total_seconds() < 21600:
            print(f"Using cached weather forecast for {race_name} {year}")
            return cached_forecast
    
    # Get track coordinates
    lat, lon = get_track_coordinates(race_name)
    
    if lat == 0 and lon == 0:
        # Could not get coordinates
        print(f"Cannot get weather forecast for {race_name}: missing coordinates")
        return None
    
    # Get race date
    race_date = get_race_date(race_name, year)
    
    if race_date is None:
        print(f"Cannot get weather forecast for {race_name}: race date not found")
        return None
    
    # Check if race is in the past or future
    is_future_race = race_date > datetime.now()
    
    # For future races, get actual forecast
    if is_future_race:
        forecast = get_actual_forecast(lat, lon, race_date)
    else:
        # For past races, get historical weather
        forecast = get_historical_weather(lat, lon, race_date)
    
    if forecast:
        # Add to cache
        forecast['last_updated'] = datetime.now().isoformat()
        WEATHER_CACHE[cache_key] = forecast
        save_weather_cache()
    
    return forecast

def get_actual_forecast(lat, lon, race_date):
    """Get weather forecast for upcoming race"""
    # In a real implementation, this would use a weather API
    # For this example, we'll return a mock forecast
    
    # For production use an actual API like WeatherAPI.com:
    # url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={lat},{lon}&days=10&aqi=no&alerts=no"
    # response = requests.get(url)
    # data = response.json()
    
    # Mock forecast
    days_until_race = (race_date - datetime.now()).days
    
    # Cannot forecast too far in advance
    if days_until_race > 10:
        return {
            'forecast_available': False,
            'too_far_in_advance': True,
            'race_date': race_date.isoformat()
        }
    
    # Generate mock forecast
    rain_probability = np.random.randint(0, 100)
    is_wet = rain_probability > 70
    
    return {
        'forecast_available': True,
        'race_date': race_date.isoformat(),
        'weather_condition': 'wet' if is_wet else 'dry',
        'precipitation_probability': rain_probability / 100,
        'temperature_c': np.random.randint(15, 35),
        'wind_kph': np.random.randint(0, 30),
        'confidence': 'high' if days_until_race <= 3 else 'medium' if days_until_race <= 7 else 'low'
    }

def get_historical_weather(lat, lon, race_date):
    """Get historical weather for past race"""
    # In a real implementation, this would use a weather API
    # For this example, we'll extract weather from FastF1 if available
    
    year = race_date.year
    race_name = None
    
    # Find race name from schedule
    try:
        schedule = fastf1.get_event_schedule(year)
        for idx, event in schedule.iterrows():
            if event['EventDate'].date() == race_date.date():
                race_name = event['EventName']
                break
    except Exception as e:
        print(f"Error finding race name: {e}")
    
    if race_name:
        # Try to get weather from race session
        try:
            from f1predict import load_session_safely
            
            race_session = load_session_safely(year, race_name, 'R')
            
            if race_session and hasattr(race_session, 'weather_data') and race_session.weather_data is not None:
                weather_data = race_session.weather_data
                
                if not weather_data.empty:
                    # Determine if it was wet
                    is_wet = False
                    
                    if 'Rainfall' in weather_data.columns:
                        rainfall = weather_data['Rainfall'].max()
                        is_wet = rainfall > 0
                    
                    # Get average temperatures
                    avg_air_temp = weather_data['AirTemp'].mean() if 'AirTemp' in weather_data.columns else None
                    avg_track_temp = weather_data['TrackTemp'].mean() if 'TrackTemp' in weather_data.columns else None
                    
                    return {
                        'forecast_available': True,
                        'historical_data': True,
                        'race_date': race_date.isoformat(),
                        'weather_condition': 'wet' if is_wet else 'dry',
                        'precipitation_occurred': is_wet,
                        'average_air_temp_c': float(avg_air_temp) if avg_air_temp is not None else None,
                        'average_track_temp_c': float(avg_track_temp) if avg_track_temp is not None else None
                    }
        except Exception as e:
            print(f"Error getting weather from FastF1: {e}")
    
    # Fallback to a mock historical record
    return {
        'forecast_available': True,
        'historical_data': True,
        'race_date': race_date.isoformat(),
        'weather_condition': 'wet' if np.random.random() < 0.3 else 'dry',  # 30% chance of wet race
        'precipitation_occurred': np.random.random() < 0.3
    }

def build_driver_weather_performance(years_back=3):
    """
    Build a database of driver performances in different weather conditions
    
    Parameters:
    - years_back: How many years of historical data to use
    
    Returns:
    - Dictionary of driver weather performance data
    """
    print("\nBuilding driver weather performance database...")
    
    global DRIVER_WEATHER_PERFORMANCE
    DRIVER_WEATHER_PERFORMANCE = {}
    
    current_year = datetime.now().year
    races_processed = 0
    
    # Try to import necessary functions
    try:
        from f1predict import load_session_safely, get_race_data, get_driver_team_mapping
    except ImportError as e:
        print(f"Error importing necessary functions: {e}")
        return DRIVER_WEATHER_PERFORMANCE
    
    # Process data for multiple years
    for year in range(current_year - years_back, current_year):
        try:
            # Get race schedule for the year
            schedule = fastf1.get_event_schedule(year)
            
            for idx, event in schedule.iterrows():
                race_name = event['EventName']
                race_date = event['EventDate']
                
                # Skip future races
                if race_date > datetime.now():
                    continue
                
                try:
                    # Load race session
                    race_session = load_session_safely(year, race_name, 'R')
                    
                    if race_session is None:
                        continue
                    
                    # Determine weather condition
                    weather_condition = 'dry'  # Default
                    
                    if hasattr(race_session, 'weather_data') and race_session.weather_data is not None:
                        weather_data = race_session.weather_data
                        
                        if not weather_data.empty and 'Rainfall' in weather_data.columns:
                            rainfall = weather_data['Rainfall'].max()
                            if rainfall > 0:
                                weather_condition = 'wet'
                    
                    # Get race results
                    race_data = get_race_data(race_session)
                    
                    # Process each driver's performance
                    for driver, data in race_data.items():
                        if driver not in DRIVER_WEATHER_PERFORMANCE:
                            DRIVER_WEATHER_PERFORMANCE[driver] = {
                                'dry': {'races': [], 'podiums': 0, 'non_finishes': 0},
                                'wet': {'races': [], 'podiums': 0, 'non_finishes': 0}
                            }
                        
                        race_position = data.get('position')
                        
                        # Process race result
                        race_result = {
                            'year': year,
                            'race': race_name,
                            'position': race_position
                        }
                        
                        DRIVER_WEATHER_PERFORMANCE[driver][weather_condition]['races'].append(race_result)
                        
                        # Update podium count
                        if race_position is not None and race_position <= 3:
                            DRIVER_WEATHER_PERFORMANCE[driver][weather_condition]['podiums'] += 1
                        
                        # Update non-finish count
                        if race_position is None or race_position > 20:  # No position or classified very low
                            DRIVER_WEATHER_PERFORMANCE[driver][weather_condition]['non_finishes'] += 1
                    
                    races_processed += 1
                    print(f"Processed {race_name} {year} - Weather: {weather_condition}")
                
                except Exception as e:
                    print(f"Error processing {race_name} {year}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error processing {year} season: {e}")
            continue
    
    # Calculate performance metrics
    for driver in DRIVER_WEATHER_PERFORMANCE:
        for condition in ['dry', 'wet']:
            data = DRIVER_WEATHER_PERFORMANCE[driver][condition]
            races = data['races']
            
            if races:
                # Calculate average position
                positions = [r['position'] for r in races if r['position'] is not None]
                if positions:
                    data['avg_position'] = sum(positions) / len(positions)
                else:
                    data['avg_position'] = None
                
                # Calculate podium rate
                data['podium_rate'] = data['podiums'] / len(races)
                
                # Calculate reliability rate
                data['reliability_rate'] = 1 - (data['non_finishes'] / len(races))
                
                # Calculate form - weighted by recency
                weighted_positions = 0
                weighted_count = 0
                current_year = datetime.now().year
                
                for race in races:
                    position = race.get('position')
                    year = race.get('year', 0)
                    
                    if position is not None:
                        # More recent races have higher weight
                        weight = 2 ** (year - (current_year - 3))
                        weighted_positions += position * weight
                        weighted_count += weight
                
                if weighted_count > 0:
                    data['weighted_avg_position'] = weighted_positions / weighted_count
                else:
                    data['weighted_avg_position'] = None
    
    print(f"\nDriver weather performance database built.")
    print(f"Processed {races_processed} races.")
    print(f"Tracking {len(DRIVER_WEATHER_PERFORMANCE)} drivers.")
    
    # Save the data
    save_driver_weather_performance()
    
    return DRIVER_WEATHER_PERFORMANCE

def get_driver_weather_adjustment(driver, weather_condition='dry'):
    """
    Get driver-specific adjustment factor based on weather performance
    
    Parameters:
    - driver: Driver code
    - weather_condition: Expected weather condition (dry or wet)
    
    Returns:
    - adjustment: Score adjustment factor (+/- value)
    - podium_factor: Factor to adjust podium probability
    """
    # Default values
    adjustment = 0.0
    podium_factor = 1.0
    
    # Load driver weather performance if not already loaded
    if not DRIVER_WEATHER_PERFORMANCE:
        loaded = load_driver_weather_performance()
        if not loaded:
            print("Building driver weather performance database...")
            build_driver_weather_performance()
    
    # Check if we have data for this driver
    if driver not in DRIVER_WEATHER_PERFORMANCE:
        return adjustment, podium_factor
    
    # Get driver's performance data
    driver_data = DRIVER_WEATHER_PERFORMANCE[driver]
    
    # Check if we have data for this weather condition
    if weather_condition not in driver_data:
        return adjustment, podium_factor
    
    # Get performance in this condition
    condition_data = driver_data[weather_condition]
    
    # Check if we have enough races for statistical significance
    if len(condition_data.get('races', [])) < 2:
        return adjustment, podium_factor
    
    # Get other condition for comparison
    other_condition = 'dry' if weather_condition == 'wet' else 'wet'
    other_condition_data = driver_data[other_condition]
    
    # Compare performance across conditions
    if condition_data.get('avg_position') is not None and other_condition_data.get('avg_position') is not None:
        # Calculate position difference
        pos_diff = other_condition_data['avg_position'] - condition_data['avg_position']
        
        # Positive difference means driver performs better in current condition
        normalized_diff = pos_diff / 20  # Normalize by max grid positions
        
        # Apply adjustment based on performance difference
        adjustment = normalized_diff * 0.15  # Scale factor (adjust as needed)
    
    # Compare podium rates
    condition_podium_rate = condition_data.get('podium_rate', 0)
    other_podium_rate = other_condition_data.get('podium_rate', 0)
    
    if condition_podium_rate > 0 and other_podium_rate > 0:
        podium_factor = condition_podium_rate / other_podium_rate
        
        # Limit extreme values
        podium_factor = min(max(podium_factor, 0.5), 2.0)
    
    return adjustment, podium_factor

def apply_weather_adjustments(predictions, race_name, year=None):
    """
    Apply weather-based adjustments to prediction scores
    
    Parameters:
    - predictions: Dictionary of predicted scores by driver
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    
    Returns:
    - adjusted_predictions: Dictionary of weather-adjusted prediction scores
    """
    if year is None:
        year = datetime.now().year
    
    print(f"\nApplying weather adjustments for {race_name} {year}...")
    
    # Get weather forecast
    forecast = get_weather_forecast(race_name, year)
    
    if not forecast or not forecast.get('forecast_available', False):
        print("No weather forecast available, skipping adjustments")
        return predictions
    
    # Get expected weather condition
    weather_condition = forecast.get('weather_condition', 'dry')
    
    print(f"Expected weather condition: {weather_condition}")
    
    # Apply adjustments to each driver
    adjusted_predictions = {}
    
    for driver, score in predictions.items():
        # Get weather adjustment for this driver
        adjustment, podium_factor = get_driver_weather_adjustment(driver, weather_condition)
        
        # Apply adjustment
        adjusted_score = score + adjustment
        
        # Ensure score stays in valid range
        adjusted_score = min(1.0, max(0.0, adjusted_score))
        
        adjusted_predictions[driver] = adjusted_score
        
        # Print significant adjustments
        if abs(adjustment) > 0.05:
            direction = "up" if adjustment > 0 else "down"
            print(f"  {driver}: adjusted {direction} by {abs(adjustment):.3f} based on {weather_condition} performance")
    
    return adjusted_predictions

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Weather Integration')
    parser.add_argument('--race', type=str, help='Race name')
    parser.add_argument('--year', type=int, default=datetime.now().year, help='Year')
    parser.add_argument('--build-data', action='store_true', help='Build weather performance database')
    
    args = parser.parse_args()
    
    if args.build_data:
        build_driver_weather_performance()
    
    if args.race:
        # Get weather forecast
        forecast = get_weather_forecast(args.race, args.year)
        
        if forecast:
            print(f"Weather forecast for {args.race} {args.year}:")
            for key, value in forecast.items():
                print(f"  {key}: {value}")
        
        # Get sample predictions
        try:
            from f1predict import improved_predict_race_winner
            
            winner, predictions = improved_predict_race_winner(args.race)
            
            # Apply weather adjustments
            adjusted_predictions = apply_weather_adjustments(predictions, args.race, args.year)
            
            # Compare top 5 before and after adjustment
            print("\nPredictions before weather adjustments (top 5):")
            for i, (driver, score) in enumerate(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"{i+1}. {driver}: {score:.3f}")
            
            print("\nPredictions after weather adjustments (top 5):")
            for i, (driver, score) in enumerate(sorted(adjusted_predictions.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"{i+1}. {driver}: {score:.3f}")
        
        except ImportError:
            print("Could not import prediction functions")