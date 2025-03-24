"""
F1 Track Database - Pre-defined track characteristics to avoid track data warnings
Works alongside the F1 prediction system to provide track data without analysis
"""
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime

# Dictionary of known F1 tracks with their characteristics
TRACK_DATABASE = {
    # Power circuits
    "Italian Grand Prix": {
        "track_type": "power",
        "overtaking_difficulty": 0.35,
        "tire_degradation": 0.50,
        "qualifying_importance": 0.55,
        "start_importance": 0.65,
        "dirty_air_impact": 0.40,
        "drs_effectiveness": 0.85,
        "top_speed_importance": 0.90,
        "track_length": 5.793,
        "corners": 11,
        "average_deg_per_lap": 0.06
    },
    "Belgian Grand Prix": {
        "track_type": "power",
        "overtaking_difficulty": 0.40,
        "tire_degradation": 0.60,
        "qualifying_importance": 0.60,
        "start_importance": 0.70,
        "dirty_air_impact": 0.50,
        "drs_effectiveness": 0.80,
        "top_speed_importance": 0.85,
        "track_length": 7.004,
        "corners": 19,
        "average_deg_per_lap": 0.07
    },
    "Azerbaijan Grand Prix": {
        "track_type": "hybrid",
        "overtaking_difficulty": 0.45,
        "tire_degradation": 0.55,
        "qualifying_importance": 0.65,
        "start_importance": 0.70,
        "dirty_air_impact": 0.55,
        "drs_effectiveness": 0.85,
        "top_speed_importance": 0.80,
        "track_length": 6.003,
        "corners": 20,
        "average_deg_per_lap": 0.05
    },
    
    # Street circuits
    "Monaco Grand Prix": {
        "track_type": "street",
        "overtaking_difficulty": 0.95,
        "tire_degradation": 0.30,
        "qualifying_importance": 0.95,
        "start_importance": 0.90,
        "dirty_air_impact": 0.90,
        "drs_effectiveness": 0.30,
        "top_speed_importance": 0.30,
        "track_length": 3.337,
        "corners": 19,
        "average_deg_per_lap": 0.03
    },
    "Singapore Grand Prix": {
        "track_type": "street",
        "overtaking_difficulty": 0.85,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.85,
        "start_importance": 0.80,
        "dirty_air_impact": 0.85,
        "drs_effectiveness": 0.40,
        "top_speed_importance": 0.40,
        "track_length": 4.940,
        "corners": 23,
        "average_deg_per_lap": 0.08
    },
    "Las Vegas Grand Prix": {
        "track_type": "street",
        "overtaking_difficulty": 0.60,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.75,
        "start_importance": 0.75,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.70,
        "top_speed_importance": 0.85,
        "track_length": 6.201,
        "corners": 17,
        "average_deg_per_lap": 0.07
    },
    
    # Technical circuits
    "Hungarian Grand Prix": {
        "track_type": "technical",
        "overtaking_difficulty": 0.75,
        "tire_degradation": 0.70,
        "qualifying_importance": 0.80,
        "start_importance": 0.85,
        "dirty_air_impact": 0.80,
        "drs_effectiveness": 0.50,
        "top_speed_importance": 0.40,
        "track_length": 4.381,
        "corners": 14,
        "average_deg_per_lap": 0.08
    },
    "Spanish Grand Prix": {
        "track_type": "technical",
        "overtaking_difficulty": 0.70,
        "tire_degradation": 0.75,
        "qualifying_importance": 0.75,
        "start_importance": 0.80,
        "dirty_air_impact": 0.80,
        "drs_effectiveness": 0.60,
        "top_speed_importance": 0.45,
        "track_length": 4.675,
        "corners": 16,
        "average_deg_per_lap": 0.09
    },
    
    # Balanced circuits
    "British Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.55,
        "tire_degradation": 0.70,
        "qualifying_importance": 0.70,
        "start_importance": 0.75,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.65,
        "top_speed_importance": 0.65,
        "track_length": 5.891,
        "corners": 18,
        "average_deg_per_lap": 0.08
    },
    "Austrian Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.50,
        "tire_degradation": 0.75,
        "qualifying_importance": 0.65,
        "start_importance": 0.70,
        "dirty_air_impact": 0.60,
        "drs_effectiveness": 0.70,
        "top_speed_importance": 0.70,
        "track_length": 4.318,
        "corners": 10,
        "average_deg_per_lap": 0.09
    },
    "United States Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.50,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.65,
        "start_importance": 0.70,
        "dirty_air_impact": 0.60,
        "drs_effectiveness": 0.70,
        "top_speed_importance": 0.65,
        "track_length": 5.513,
        "corners": 20,
        "average_deg_per_lap": 0.07
    },
    
    # Other circuits
    "Abu Dhabi Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.65,
        "tire_degradation": 0.60,
        "qualifying_importance": 0.75,
        "start_importance": 0.75,
        "dirty_air_impact": 0.70,
        "drs_effectiveness": 0.65,
        "top_speed_importance": 0.60,
        "track_length": 5.281,
        "corners": 16,
        "average_deg_per_lap": 0.06
    },
    "Bahrain Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.45,
        "tire_degradation": 0.75,
        "qualifying_importance": 0.60,
        "start_importance": 0.70,
        "dirty_air_impact": 0.55,
        "drs_effectiveness": 0.75,
        "top_speed_importance": 0.65,
        "track_length": 5.412,
        "corners": 15,
        "average_deg_per_lap": 0.09
    },
    "Saudi Arabian Grand Prix": {
        "track_type": "power",
        "overtaking_difficulty": 0.55,
        "tire_degradation": 0.55,
        "qualifying_importance": 0.75,
        "start_importance": 0.70,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.75,
        "top_speed_importance": 0.80,
        "track_length": 6.174,
        "corners": 27,
        "average_deg_per_lap": 0.05
    },
    "Australian Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.60,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.70,
        "start_importance": 0.75,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.65,
        "top_speed_importance": 0.60,
        "track_length": 5.278,
        "corners": 14,
        "average_deg_per_lap": 0.07
    },
    "Japanese Grand Prix": {
        "track_type": "technical",
        "overtaking_difficulty": 0.65,
        "tire_degradation": 0.75,
        "qualifying_importance": 0.75,
        "start_importance": 0.70,
        "dirty_air_impact": 0.75,
        "drs_effectiveness": 0.55,
        "top_speed_importance": 0.55,
        "track_length": 5.807,
        "corners": 18,
        "average_deg_per_lap": 0.08
    },
    "Chinese Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.50,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.65,
        "start_importance": 0.75,
        "dirty_air_impact": 0.60,
        "drs_effectiveness": 0.70,
        "top_speed_importance": 0.60,
        "track_length": 5.451,
        "corners": 16,
        "average_deg_per_lap": 0.07
    },
    "Mexican Grand Prix": {
        "track_type": "power",
        "overtaking_difficulty": 0.60,
        "tire_degradation": 0.70,
        "qualifying_importance": 0.70,
        "start_importance": 0.75,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.75,
        "top_speed_importance": 0.75,
        "track_length": 4.304,
        "corners": 17,
        "average_deg_per_lap": 0.08
    },
    "Mexico City Grand Prix": {
        "track_type": "power",
        "overtaking_difficulty": 0.60,
        "tire_degradation": 0.70,
        "qualifying_importance": 0.70,
        "start_importance": 0.75,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.75,
        "top_speed_importance": 0.75,
        "track_length": 4.304,
        "corners": 17,
        "average_deg_per_lap": 0.08
    },
    "Dutch Grand Prix": {
        "track_type": "technical",
        "overtaking_difficulty": 0.70,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.80,
        "start_importance": 0.75,
        "dirty_air_impact": 0.75,
        "drs_effectiveness": 0.55,
        "top_speed_importance": 0.50,
        "track_length": 4.259,
        "corners": 14,
        "average_deg_per_lap": 0.07
    },
    "Qatar Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.55,
        "tire_degradation": 0.80,
        "qualifying_importance": 0.70,
        "start_importance": 0.70,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.70,
        "top_speed_importance": 0.65,
        "track_length": 5.419,
        "corners": 16,
        "average_deg_per_lap": 0.10
    },
    "São Paulo Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.45,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.65,
        "start_importance": 0.75,
        "dirty_air_impact": 0.60,
        "drs_effectiveness": 0.75,
        "top_speed_importance": 0.70,
        "track_length": 4.309,
        "corners": 15,
        "average_deg_per_lap": 0.07
    },
    "Brazilian Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.45,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.65,
        "start_importance": 0.75,
        "dirty_air_impact": 0.60,
        "drs_effectiveness": 0.75,
        "top_speed_importance": 0.70,
        "track_length": 4.309,
        "corners": 15,
        "average_deg_per_lap": 0.07
    },
    "Canadian Grand Prix": {
        "track_type": "power",
        "overtaking_difficulty": 0.45,
        "tire_degradation": 0.60,
        "qualifying_importance": 0.65,
        "start_importance": 0.70,
        "dirty_air_impact": 0.55,
        "drs_effectiveness": 0.80,
        "top_speed_importance": 0.75,
        "track_length": 4.361,
        "corners": 14,
        "average_deg_per_lap": 0.06
    },
    "Miami Grand Prix": {
        "track_type": "balanced",
        "overtaking_difficulty": 0.55,
        "tire_degradation": 0.65,
        "qualifying_importance": 0.70,
        "start_importance": 0.75,
        "dirty_air_impact": 0.65,
        "drs_effectiveness": 0.70,
        "top_speed_importance": 0.65,
        "track_length": 5.412,
        "corners": 19,
        "average_deg_per_lap": 0.07
    },
    "Emilia Romagna Grand Prix": {
        "track_type": "technical",
        "overtaking_difficulty": 0.70,
        "tire_degradation": 0.60,
        "qualifying_importance": 0.80,
        "start_importance": 0.75,
        "dirty_air_impact": 0.75,
        "drs_effectiveness": 0.60,
        "top_speed_importance": 0.60,
        "track_length": 4.909,
        "corners": 19,
        "average_deg_per_lap": 0.06
    }
}

def get_track_data(race_name, year=None):
    """
    Get track characteristics from the database
    
    Parameters:
    - race_name: Name of the race (e.g., 'Monaco Grand Prix')
    - year: Year of the race (optional, default is None)
    
    Returns:
    - Dictionary of track data or None if not found
    """
    # Normalize race name (remove year if present)
    race_name = race_name.split(' Grand Prix')[0].strip() + ' Grand Prix'
    
    # Look for exact match first
    if race_name in TRACK_DATABASE:
        return TRACK_DATABASE[race_name]
    
    # Try common variations
    if 'Mexico City' in race_name:
        if 'Mexican Grand Prix' in TRACK_DATABASE:
            return TRACK_DATABASE['Mexican Grand Prix']
    
    if 'São Paulo' in race_name or 'Sao Paulo' in race_name:
        if 'Brazilian Grand Prix' in TRACK_DATABASE:
            return TRACK_DATABASE['Brazilian Grand Prix']
    
    # Fuzzy matching for partial names
    for db_name, data in TRACK_DATABASE.items():
        if race_name.split(' ')[0].lower() in db_name.lower():
            return data
    
    return None

def generate_optimized_weights(track_data):
    """
    Generate optimized prediction weights based on track characteristics
    
    Parameters:
    - track_data: Dictionary of track characteristics
    
    Returns:
    - Dictionary of optimized weights for prediction
    """
    if not track_data:
        # Default weights if no track data is available
        return {
            'quali_weight': 0.4,
            'sector_performance_weight': 0.10,
            'tire_management_weight': 0.10,
            'race_start_weight': 0.08,
            'overtaking_ability_weight': 0.06,
            'team_strategy_weight': 0.08,
            'starting_position_weight': 0.18,
            'team_dynamics_weight': 0.12,
            'dirty_air_weight': 0.12
        }
    
    # Extract track characteristics
    track_type = track_data.get('track_type', 'balanced')
    qualifying_importance = track_data.get('qualifying_importance', 0.7)
    tire_degradation = track_data.get('tire_degradation', 0.6)
    start_importance = track_data.get('start_importance', 0.7)
    dirty_air_impact = track_data.get('dirty_air_impact', 0.6)
    overtaking_difficulty = track_data.get('overtaking_difficulty', 0.6)
    
    # Calculate weights based on track characteristics
    weights = {
        'quali_weight': 0.4 * qualifying_importance / 0.7,
        'sector_performance_weight': 0.10,
        'tire_management_weight': 0.10 * tire_degradation / 0.6,
        'race_start_weight': 0.08 * start_importance / 0.7,
        'overtaking_ability_weight': 0.06 * (1 - overtaking_difficulty),
        'team_strategy_weight': 0.08 * (tire_degradation / 0.6),
        'starting_position_weight': 0.18 * (qualifying_importance / 0.7),
        'team_dynamics_weight': 0.12,
        'dirty_air_weight': 0.12 * dirty_air_impact / 0.6
    }
    
    # Track type specific adjustments
    if track_type == 'street':
        weights['quali_weight'] *= 1.2
        weights['starting_position_weight'] *= 1.1
        weights['overtaking_ability_weight'] *= 0.8
    elif track_type == 'power':
        weights['overtaking_ability_weight'] *= 1.2
        weights['quali_weight'] *= 0.9
    elif track_type == 'technical':
        weights['tire_management_weight'] *= 1.1
        weights['team_strategy_weight'] *= 1.1
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    return weights

def save_track_data(race_name, year=None):
    """
    Save track data to track_analysis directory to prevent warnings
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    
    Returns:
    - Path to saved file or None if not saved
    """
    if year is None:
        year = datetime.now().year
    
    # Get track data
    track_data = get_track_data(race_name)
    
    if not track_data:
        return None
    
    # Ensure directory exists
    output_dir = 'track_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Add additional metadata
    save_data = track_data.copy()
    save_data.update({
        'race_name': race_name,
        'year': year,
        'source': 'track_database.py',
        'date_generated': datetime.now().strftime('%Y-%m-%d')
    })
    
    # Convert any numeric types to standard Python types
    for key, value in save_data.items():
        if isinstance(value, (pd.Series, pd.DataFrame)):
            save_data[key] = value.to_dict()
        elif isinstance(value, (np.int32, np.int64)):
            save_data[key] = int(value)
        elif isinstance(value, (np.float32, np.float64)):
            save_data[key] = float(value)
    
    # Save to file
    filename = f"{output_dir}/{year}_{race_name.replace(' ', '_')}_data.json"
    try:
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=4)
        return filename
    except Exception as e:
        print(f"Error saving track data for {race_name}: {e}")
        return None

def setup_tracks_for_season(year=None):
    """
    Set up track data files for all races in a season
    
    Parameters:
    - year: Year of the season (default: current year)
    
    Returns:
    - Number of track data files created
    """
    if year is None:
        year = datetime.now().year
    
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(year)
        races = schedule['EventName'].tolist()
        
        count = 0
        for race in races:
            filename = save_track_data(race, year)
            if filename:
                print(f"Created track data file: {filename}")
                count += 1
        
        return count
    except ImportError:
        # If FastF1 is not available, use all tracks in the database
        count = 0
        for race in TRACK_DATABASE.keys():
            filename = save_track_data(race, year)
            if filename:
                print(f"Created track data file: {filename}")
                count += 1
        
        return count

# Monkey-patch for get_optimized_track_weights
def get_optimized_track_weights(race_name, year=None, load_track_data_func=None):
    """
    Get optimized feature weights based on track characteristics
    
    Parameters:
    - race_name: Name of the race/circuit
    - year: Year to use for track data
    - load_track_data_func: Custom function to load track data if provided
    
    Returns:
    - weights: Dictionary of optimized feature weights
    """
    # First try custom load function if provided
    if load_track_data_func is not None:
        track_data = load_track_data_func(race_name, year)
        if track_data is not None:
            return generate_optimized_weights(track_data)
    
    # Then try database
    track_data = get_track_data(race_name, year)
    if track_data is not None:
        return generate_optimized_weights(track_data)
    
    # Default weights if no track data is available
    print(f"No track data available for {race_name}, using default weights")
    return generate_optimized_weights(None)

if __name__ == "__main__":
    # Run setup function when script is executed directly
    print("Setting up track data files for current season...")
    count = setup_tracks_for_season()
    print(f"Created {count} track data files")
    
    # Test getting optimized weights
    test_race = "Monaco Grand Prix"
    weights = get_optimized_track_weights(test_race)
    print(f"\nOptimized weights for {test_race}:")
    for key, value in weights.items():
        print(f"  {key}: {value:.3f}")