import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

def analyze_track_characteristics(race_name, year=None, load_session_function=None, 
                                 get_qualifying_data_function=None, get_race_data_function=None):
    """
    Analyze historical track characteristics to improve prediction model
    
    Parameters:
    - race_name: Name of the race/circuit
    - year: Year to analyze (defaults to current year - 1)
    - load_session_function: Function to load race session
    - get_qualifying_data_function: Function to extract qualifying data
    - get_race_data_function: Function to extract race data
    
    Returns:
    - track_data: Dictionary of track-specific metrics and characteristics
    """
    # Set year if not provided
    if year is None:
        year = datetime.now().year - 1  # Previous year
    
    print(f"Analyzing {race_name} track characteristics based on {year} data...")
    
    # Check for required functions
    if load_session_function is None:
        print("Error: load_session_function is required")
        return None
        
    if get_qualifying_data_function is None or get_race_data_function is None:
        print("Error: data extraction functions are required")
        return None
    
    # Store track-specific data
    track_data = {
        'race_name': race_name,
        'year': year,
        'overtaking_metrics': {},
        'qualifying_importance': None,
        'tire_degradation': None,
        'start_importance': None,
        'dirty_air_impact': None,
        'track_type': None,
        'track_characteristics': {}
    }
    
    # Load sessions
    quali_session = load_session_function(year, race_name, 'Q')
    race_session = load_session_function(year, race_name, 'R')
    
    if race_session is None:
        print(f"Race data for {race_name} {year} not available.")
        return None
    
    # Process race data
    try:
        # 1. Analyze overtaking
        overtaking_metrics = calculate_overtaking_metrics(race_session)
        if overtaking_metrics:
            track_data['overtaking_metrics'] = overtaking_metrics
            
            # Derive overtaking difficulty (higher = more difficult)
            if 'total_overtakes' in overtaking_metrics:
                total_overtakes = overtaking_metrics['total_overtakes']
                # Scale: >50 overtakes = easy (0.3), <10 overtakes = difficult (0.9)
                track_data['overtaking_difficulty'] = max(0.3, min(0.9, 0.9 - (total_overtakes / 70) * 0.6))
                print(f"Estimated overtaking difficulty: {track_data['overtaking_difficulty']:.2f}")
        
        # 2. Analyze qualifying importance by comparing quali vs race positions
        if quali_session is not None:
            quali_importance = calculate_qualifying_importance(
                quali_session, race_session, get_qualifying_data_function, get_race_data_function
            )
            track_data['qualifying_importance'] = quali_importance
            print(f"Qualifying position importance: {quali_importance:.2f}")
        
        # 3. Analyze tire degradation
        tire_deg = analyze_tire_degradation(race_session)
        if tire_deg is not None:
            track_data['tire_degradation'] = tire_deg
            print(f"Tire degradation factor: {tire_deg:.2f}")
        
        # 4. Analyze race start importance
        start_importance = analyze_start_importance(
            quali_session, race_session, get_qualifying_data_function
        )
        if start_importance is not None:
            track_data['start_importance'] = start_importance
            print(f"Start importance factor: {start_importance:.2f}")
        
        # 5. Estimate dirty air impact
        dirty_air = estimate_dirty_air_impact(race_session)
        if dirty_air is not None:
            track_data['dirty_air_impact'] = dirty_air
            print(f"Dirty air impact factor: {dirty_air:.2f}")
        
        # 6. Determine track type
        track_type = determine_track_type(race_name, race_session)
        track_data['track_type'] = track_type
        print(f"Track type classification: {track_type}")
        
        # 7. Additional track characteristics
        track_characteristics = extract_track_characteristics(race_session)
        if track_characteristics:
            track_data['track_characteristics'] = track_characteristics
        
        # 8. Visualize track analysis
        visualize_track_analysis(track_data, race_name, year)
        
        # 9. Save track data
        save_track_data(track_data)
        
        return track_data
    
    except Exception as e:
        print(f"Error analyzing track characteristics: {e}")
        return None

def calculate_overtaking_metrics(race_session):
    """Calculate overtaking metrics based on race session data"""
    try:
        # Get all laps
        laps = race_session.laps
        
        if laps.empty:
            return None
        
        # Initialize counters
        overtake_count = 0
        position_changes = 0
        
        # Get all drivers
        drivers = race_session.drivers
        
        # Track position changes by lap
        for driver in drivers:
            try:
                driver_laps = laps.pick_drivers(driver)
                
                # Skip if no laps
                if driver_laps.empty:
                    continue
                
                # Sort by lap number
                driver_laps = driver_laps.sort_values('LapNumber')
                
                # Check for position changes across laps
                if 'Position' in driver_laps.columns:
                    prev_position = None
                    
                    for _, lap in driver_laps.iterrows():
                        if pd.notna(lap['Position']):
                            current_position = lap['Position']
                            
                            if prev_position is not None and current_position != prev_position:
                                position_changes += 1
                                
                                # Count only improvements as overtakes
                                if current_position < prev_position:
                                    overtake_count += 1
                            
                            prev_position = current_position
            
            except Exception as e:
                print(f"Error processing driver {driver}: {e}")
                continue
        
        # Calculate pit stop count to adjust overtaking metrics
        pit_stops = 0
        for driver in drivers:
            try:
                driver_laps = laps.pick_drivers(driver)
                pit_stops += len(driver_laps[~driver_laps['PitInTime'].isna()])
            except Exception:
                continue
        
        # Adjust overtake count to exclude pit-related position changes
        adjusted_overtakes = max(0, overtake_count - pit_stops)
        
        # Estimate DRS effectiveness by analyzing speed changes
        drs_effectiveness = None
        try:
            if 'DRS' in laps.columns:
                drs_laps = laps[laps['DRS'] > 0]
                non_drs_laps = laps[laps['DRS'] == 0]
                
                if not drs_laps.empty and not non_drs_laps.empty and 'SpeedST' in laps.columns:
                    drs_speed = drs_laps['SpeedST'].mean()
                    non_drs_speed = non_drs_laps['SpeedST'].mean()
                    
                    if pd.notna(drs_speed) and pd.notna(non_drs_speed) and non_drs_speed > 0:
                        drs_effectiveness = (drs_speed / non_drs_speed) - 1
        except Exception:
            drs_effectiveness = None
        
        return {
            'total_overtakes': adjusted_overtakes,
            'position_changes': position_changes,
            'pit_stops': pit_stops,
            'drs_effectiveness': drs_effectiveness
        }
    
    except Exception as e:
        print(f"Error calculating overtaking metrics: {e}")
        return None

def calculate_qualifying_importance(quali_session, race_session, get_qualifying_data_function, get_race_data_function):
    """Calculate the importance of qualifying position for race results"""
    try:
        # Get qualifying data
        quali_data, _ = get_qualifying_data_function(quali_session)
        
        # Get race results
        race_data = get_race_data_function(race_session)
        
        # Check if we have both datasets
        if not quali_data or not race_data:
            return 0.7  # Default value
        
        # Calculate correlation between qualifying and race positions
        common_drivers = set(quali_data.keys()) & set(race_data.keys())
        
        if len(common_drivers) < 3:
            return 0.7  # Not enough data
        
        quali_positions = []
        race_positions = []
        
        for driver in common_drivers:
            if 'position' in quali_data[driver] and 'position' in race_data[driver]:
                quali_positions.append(quali_data[driver]['position'])
                race_positions.append(race_data[driver]['position'])
        
        if len(quali_positions) < 3:
            return 0.7  # Not enough valid positions
        
        # Calculate Spearman rank correlation
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(quali_positions, race_positions)
        
        # Convert correlation to importance factor (0.5-0.95)
        # Higher correlation = qualifying more important
        importance = 0.5 + (correlation * 0.45)
        
        # Extra importance for front row
        # Check if pole position finished in top 3
        pole_driver = None
        for driver, data in quali_data.items():
            if data.get('position') == 1:
                pole_driver = driver
                break
        
        if pole_driver and pole_driver in race_data:
            pole_finish = race_data[pole_driver].get('position')
            if pole_finish is not None and pole_finish <= 3:
                importance += 0.05  # Bonus for pole conversion
        
        return min(0.95, max(0.5, importance))
    
    except Exception as e:
        print(f"Error calculating qualifying importance: {e}")
        return 0.7  # Default value

def analyze_tire_degradation(race_session):
    """Analyze tire degradation factor based on race pace drop-off"""
    try:
        # Get all laps
        laps = race_session.laps
        
        if laps.empty:
            return None
        
        # Track degradation by compound
        compound_degradation = {}
        
        # Process each driver's stints
        for driver in race_session.drivers:
            driver_laps = laps.pick_drivers(driver)
            
            # Skip if no laps
            if driver_laps.empty:
                continue
            
            # Organize laps by stint
            stints = []
            current_stint = []
            
            # Sort by lap number
            sorted_laps = driver_laps.sort_values('LapNumber')
            
            for i, lap in sorted_laps.iterrows():
                if len(current_stint) == 0:
                    current_stint.append(lap)
                else:
                    # Check if this is a consecutive lap
                    prev_lap = current_stint[-1]
                    if lap['LapNumber'] == prev_lap['LapNumber'] + 1:
                        current_stint.append(lap)
                    else:
                        # Non-consecutive lap, new stint
                        if len(current_stint) > 5:  # Only consider substantial stints
                            stints.append(current_stint)
                        current_stint = [lap]
            
            # Add last stint if substantial
            if len(current_stint) > 5:
                stints.append(current_stint)
            
            # Analyze each stint
            for stint in stints:
                if len(stint) < 6:
                    continue  # Skip short stints
                
                # Get compound if available
                compound = stint[0].get('Compound', 'Unknown')
                
                # Extract lap times and numbers
                lap_numbers = [lap['LapNumber'] for lap in stint]
                lap_times = [lap['LapTime'].total_seconds() if pd.notna(lap['LapTime']) else None for lap in stint]
                
                # Remove None values
                valid_indices = [i for i, t in enumerate(lap_times) if t is not None]
                lap_numbers = [lap_numbers[i] for i in valid_indices]
                lap_times = [lap_times[i] for i in valid_indices]
                
                # Skip if not enough valid laps
                if len(lap_times) < 5:
                    continue
                
                # Skip first lap of stint (often not representative)
                lap_numbers = lap_numbers[1:]
                lap_times = lap_times[1:]
                
                # Calculate linear degradation trend
                try:
                    # Simple linear regression
                    x = np.array(lap_numbers)
                    y = np.array(lap_times)
                    
                    # Normalize lap times by first lap
                    first_lap_time = y[0]
                    y_norm = y / first_lap_time
                    
                    A = np.vstack([x, np.ones(len(x))]).T
                    m, c = np.linalg.lstsq(A, y_norm, rcond=None)[0]
                    
                    # m = slope = degradation per lap (as percentage)
                    degradation_percentage = m * 100
                    
                    # Store degradation by compound
                    if compound not in compound_degradation:
                        compound_degradation[compound] = []
                    
                    compound_degradation[compound].append(degradation_percentage)
                
                except Exception:
                    continue
        
        # Calculate average degradation across compounds
        if not compound_degradation:
            return 0.6  # Default value
        
        # Calculate weighted average (softer compounds have higher weight)
        compound_weights = {'soft': 1.0, 'medium': 0.8, 'hard': 0.6, 'Unknown': 0.7}
        
        total_weight = 0
        weighted_sum = 0
        
        for compound, degradations in compound_degradation.items():
            if not degradations:
                continue
            
            avg_degradation = sum(degradations) / len(degradations)
            compound_lower = compound.lower()
            
            weight = 0.7  # Default weight
            for key, val in compound_weights.items():
                if key.lower() in compound_lower:
                    weight = val
                    break
            
            weighted_sum += avg_degradation * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.6  # Default value
        
        avg_degradation = weighted_sum / total_weight
        
        # Convert to degradation factor (0.3-0.9)
        # Higher degradation = higher factor
        # Typical degradation might be 0.05-0.5% per lap
        degradation_factor = 0.3 + min(0.6, abs(avg_degradation) * 1.5)
        
        return degradation_factor
    
    except Exception as e:
        print(f"Error analyzing tire degradation: {e}")
        return 0.6  # Default value

def analyze_start_importance(quali_session, race_session, get_qualifying_data_function):
    """Analyze importance of race start based on position changes on lap 1"""
    try:
        if quali_session is None or race_session is None:
            return 0.7  # Default value
        
        # Get qualifying positions
        quali_data, _ = get_qualifying_data_function(quali_session)
        
        # Get all laps from race
        laps = race_session.laps
        
        if laps.empty:
            return 0.7  # Default value
        
        # Calculate position changes on lap 1
        lap1_changes = []
        
        for driver in race_session.drivers:
            try:
                driver_abbr = race_session.get_driver(driver)['Abbreviation']
                
                # Skip if driver not in qualifying
                if driver_abbr not in quali_data:
                    continue
                
                quali_pos = quali_data[driver_abbr]['position']
                
                # Get lap 1 position
                lap1 = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == 1)]
                
                if lap1.empty or 'Position' not in lap1.columns:
                    continue
                
                lap1_pos = lap1['Position'].iloc[0]
                
                if pd.notna(lap1_pos):
                    # Calculate position change
                    position_change = quali_pos - lap1_pos
                    lap1_changes.append(position_change)
            
            except Exception:
                continue
        
        if not lap1_changes:
            return 0.7  # Default value
        
        # Calculate average absolute position change
        avg_change = sum(abs(change) for change in lap1_changes) / len(lap1_changes)
        
        # Convert to start importance factor (0.5-0.9)
        # More position changes = higher importance
        importance = 0.5 + min(0.4, avg_change * 0.1)
        
        return importance
    
    except Exception as e:
        print(f"Error analyzing start importance: {e}")
        return 0.7  # Default value

def estimate_dirty_air_impact(race_session):
    """Estimate dirty air impact based on lap time differentials when following"""
    try:
        # Get all laps
        laps = race_session.laps
        
        if laps.empty:
            return 0.6  # Default value
        
        # Check if we have telemetry data
        if not hasattr(race_session, 'car_data') or race_session.car_data is None:
            # Estimate based on circuit characteristics
            return 0.6  # Default value
        
        # Collect instances of cars following closely
        following_time_diffs = []
        
        # Process each lap
        for lap_number in range(2, int(laps['LapNumber'].max()) + 1):
            # Get laps for this lap number
            lap_data = laps[laps['LapNumber'] == lap_number]
            
            if lap_data.empty:
                continue
            
            # Sort by track position
            if 'Position' in lap_data.columns:
                lap_data = lap_data.sort_values('Position')
            
            # Check consecutive drivers
            for i in range(len(lap_data) - 1):
                try:
                    # Get current and next driver
                    curr_driver = lap_data.iloc[i]['Driver']
                    next_driver = lap_data.iloc[i+1]['Driver']
                    
                    # Get telemetry for both drivers
                    try:
                        curr_telemetry = race_session.car_data[curr_driver].loc[lap_number]
                        next_telemetry = race_session.car_data[next_driver].loc[lap_number]
                    except:
                        continue
                    
                    # Calculate speed difference when following closely
                    if 'Speed' in curr_telemetry.columns and 'Speed' in next_telemetry.columns:
                        # Compare speeds in corners (lower speeds)
                        corner_mask = (curr_telemetry['Speed'] < 150)  # Assuming corners are below 150 km/h
                        
                        if not corner_mask.any():
                            continue
                        
                        # Get average corner speeds
                        curr_corner_speed = curr_telemetry.loc[corner_mask, 'Speed'].mean()
                        next_corner_speed = next_telemetry.loc[corner_mask, 'Speed'].mean()
                        
                        if pd.notna(curr_corner_speed) and pd.notna(next_corner_speed) and curr_corner_speed > 0:
                            # Calculate normalized speed difference
                            speed_diff = (curr_corner_speed - next_corner_speed) / curr_corner_speed
                            following_time_diffs.append(speed_diff)
                
                except Exception:
                    continue
        
        # If not enough data, estimate from track width
        if len(following_time_diffs) < 5:
            return 0.6  # Default value
        
        # Calculate average speed difference
        avg_diff = sum(following_time_diffs) / len(following_time_diffs)
        
        # Convert to dirty air impact factor (0.4-0.9)
        # Higher difference = more dirty air impact
        impact = 0.4 + min(0.5, avg_diff * 10)
        
        return impact
    
    except Exception as e:
        print(f"Error estimating dirty air impact: {e}")
        return 0.6  # Default value

def determine_track_type(race_name, race_session):
    """Determine the type of track based on circuit characteristics"""
    # Track type classification based on name patterns
    race_lower = race_name.lower()
    
    if any(name in race_lower for name in ['monaco', 'singapore', 'baku']):
        return 'street'
    elif any(name in race_lower for name in ['monza', 'spa', 'azerbaijan']):
        return 'power'
    elif any(name in race_lower for name in ['hungar', 'barcelona', 'catalunya']):
        return 'technical'
    elif any(name in race_lower for name in ['silverstone', 'suzuka', 'austin']):
        return 'balanced'
    
    # If we can't determine from name, analyze lap characteristics
    laps = race_session.laps
    
    if laps.empty:
        return 'unknown'
    
    # Check top speed and average speed if available
    try:
        if 'SpeedI2' in laps.columns and 'SpeedFL' in laps.columns:
            # Get max speed over the weekend
            max_speed = laps['SpeedI2'].max()
            avg_speed = laps['SpeedFL'].mean()
            
            if pd.notna(max_speed) and pd.notna(avg_speed):
                if max_speed > 330:  # High top speed
                    return 'power'
                elif avg_speed < 200:  # Low average speed
                    return 'technical'
                else:
                    return 'balanced'
    except:
        pass
    
    return 'unknown'

def extract_track_characteristics(race_session):
    """Extract additional track characteristics for prediction"""
    characteristics = {}
    
    try:
        # Get event information
        event_info = race_session.event
        
        if hasattr(event_info, 'to_dict'):
            event_dict = event_info.to_dict()
            
            # Extract relevant information
            if 'EventFormat' in event_dict:
                characteristics['event_format'] = event_dict['EventFormat']
            
            if 'Country' in event_dict:
                characteristics['country'] = event_dict['Country']
            
            if 'F1ApiSupport' in event_dict:
                characteristics['api_support'] = event_dict['F1ApiSupport']
        
        # Analyze lap data
        laps = race_session.laps
        
        if not laps.empty:
            # Calculate lap time statistics
            if 'LapTime' in laps.columns and not laps['LapTime'].isna().all():
                median_lap = laps['LapTime'].median().total_seconds()
                characteristics['median_lap_time'] = median_lap
            
            # Estimate track length
            if 'PitInTime' in laps.columns and 'Distance' in laps.columns:
                non_pit_laps = laps[laps['PitInTime'].isna()]
                if not non_pit_laps.empty and 'Distance' in non_pit_laps.columns:
                    avg_distance = non_pit_laps['Distance'].mean()
                    if pd.notna(avg_distance):
                        characteristics['track_length'] = avg_distance
            
            # Count safety cars and red flags
            if 'TrackStatus' in laps.columns:
                safety_car_laps = laps[laps['TrackStatus'].str.contains('SC|VSC|RS', na=False)]
                characteristics['safety_car_laps'] = len(safety_car_laps)
        
        # Weather data
        if hasattr(race_session, 'weather_data') and race_session.weather_data is not None:
            weather = race_session.weather_data
            
            if not weather.empty:
                if 'AirTemp' in weather.columns:
                    characteristics['avg_air_temp'] = weather['AirTemp'].mean()
                
                if 'TrackTemp' in weather.columns:
                    characteristics['avg_track_temp'] = weather['TrackTemp'].mean()
                
                if 'Rainfall' in weather.columns:
                    characteristics['rainfall'] = weather['Rainfall'].max() > 0
    
    except Exception as e:
        print(f"Error extracting track characteristics: {e}")
    
    return characteristics

def visualize_track_analysis(track_data, race_name, year):
    """Create visualizations for track analysis"""
    try:
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Set up radar chart
        categories = ['Qualifying\nImportance', 'Tire\nDegradation', 'Start\nImportance', 
                      'Dirty Air\nImpact', 'Overtaking\nDifficulty']
        
        values = [
            track_data.get('qualifying_importance', 0.7),
            track_data.get('tire_degradation', 0.6),
            track_data.get('start_importance', 0.7),
            track_data.get('dirty_air_impact', 0.6),
            track_data.get('overtaking_difficulty', 0.6)
        ]
        
        # Make sure values are between 0 and 1
        values = [min(1, max(0, v)) for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Close the polygon
        angles += angles[:1]  # Close the polygon
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set category labels
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        # Set radial limits
        ax.set_ylim(0, 1)
        
        # Set radial labels
        ax.set_rticks([0.2, 0.4, 0.6, 0.8])
        ax.set_rlabel_position(0)
        
        # Add title
        plt.title(f"{race_name} {year} Track Analysis", size=14, y=1.1)
        
        # Add annotation with track type
        track_type = track_data.get('track_type', 'unknown')
        plt.annotate(f"Track Type: {track_type.capitalize()}", xy=(0.5, 0.02), 
                    xycoords='figure fraction', ha='center', fontsize=12)
        
        # Save the figure
        output_dir = 'track_analysis'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(f"{output_dir}/{year}_{race_name.replace(' ', '_')}_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Saved track analysis visualization to {output_dir}/{year}_{race_name.replace(' ', '_')}_analysis.png")
        
        # Close the figure
        plt.close()
        
        # If we have overtaking data, create a separate visualization
        overtaking_metrics = track_data.get('overtaking_metrics', {})
        if overtaking_metrics and 'total_overtakes' in overtaking_metrics:
            plt.figure(figsize=(10, 6))
            
            overtakes = overtaking_metrics['total_overtakes']
            
            plt.bar(['Total Overtakes'], [overtakes], color='skyblue')
            plt.ylabel('Number of Overtakes')
            plt.title(f"{race_name} {year} Overtaking Analysis")
            
            # Add overtaking difficulty annotation
            difficulty = track_data.get('overtaking_difficulty', 0.6)
            difficulty_label = "Very Difficult" if difficulty > 0.8 else \
                              "Difficult" if difficulty > 0.6 else \
                              "Moderate" if difficulty > 0.4 else "Easy"
            
            plt.annotate(f"Overtaking Difficulty: {difficulty_label} ({difficulty:.2f})", 
                        xy=(0.5, 0.9), xycoords='figure fraction', ha='center')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{year}_{race_name.replace(' ', '_')}_overtaking.png", dpi=300)
            print(f"Saved overtaking analysis to {output_dir}/{year}_{race_name.replace(' ', '_')}_overtaking.png")
            plt.close()
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def save_track_data(track_data):
    """Save track analysis data to file for future reference"""
    # Create directory if it doesn't exist
    output_dir = 'track_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    race_name = track_data['race_name']
    year = track_data['year']
    
    # Convert any non-serializable values
    for key, value in track_data.items():
        if isinstance(value, np.float64):
            track_data[key] = float(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, np.float64):
                    track_data[key][k] = float(v)
    
    # Save to file
    filename = f"{output_dir}/{year}_{race_name.replace(' ', '_')}_data.json"
    try:
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=4)
        print(f"Saved track data to {filename}")
    except Exception as e:
        print(f"Error saving track data: {e}")

def load_track_data(race_name, year=None):
    """Load previously analyzed track data"""
    # Determine year if not provided
    if year is None:
        year = datetime.now().year - 1  # Previous year
    
    output_dir = 'track_analysis'
    filename = f"{output_dir}/{year}_{race_name.replace(' ', '_')}_data.json"
    
    if not os.path.exists(filename):
        print(f"No track data file found for {race_name} {year}")
        return None
    
    try:
        with open(filename, 'r') as f:
            track_data = json.load(f)
        return track_data
    except Exception as e:
        print(f"Error loading track data: {e}")
        return None

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
    # Try to load previously analyzed track data
    current_year = datetime.now().year
    
    # Use provided load function or default to built-in
    track_data_loader = load_track_data_func if load_track_data_func else load_track_data
    
    # Try current year - 1 first, then current year - 2 as fallback
    track_data = track_data_loader(race_name, current_year - 1)
    
    if track_data is None:
        track_data = track_data_loader(race_name, current_year - 2)
    
    # If no track data available, use default weights
    if track_data is None:
        print(f"No track data available for {race_name}, using default weights")
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
    quali_importance = track_data.get('qualifying_importance', 0.7)
    tire_degradation = track_data.get('tire_degradation', 0.6)
    start_importance = track_data.get('start_importance', 0.7)
    dirty_air_impact = track_data.get('dirty_air_impact', 0.6)
    overtaking_difficulty = track_data.get('overtaking_difficulty', 0.6)
    
    # Adjust weights based on track characteristics
    weights = {}
    
    # Base weights
    weights['quali_weight'] = 0.4 * quali_importance / 0.7  # Scale by importance
    weights['sector_performance_weight'] = 0.10
    weights['tire_management_weight'] = 0.10 * tire_degradation / 0.6  # Scale by degradation
    weights['race_start_weight'] = 0.08 * start_importance / 0.7  # Scale by importance
    weights['overtaking_ability_weight'] = 0.06 * (1 - overtaking_difficulty)  # Inverse scale
    weights['team_strategy_weight'] = 0.08 * (tire_degradation / 0.6)  # More important with high deg
    weights['starting_position_weight'] = 0.18 * (quali_importance / 0.7)  # Scale by quali importance
    weights['team_dynamics_weight'] = 0.12
    weights['dirty_air_weight'] = 0.12 * dirty_air_impact / 0.6  # Scale by impact
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Print optimized weights
    print(f"\nOptimized weights for {race_name} based on track characteristics:")
    for k, v in normalized_weights.items():
        print(f"  {k}: {v:.3f}")
    
    return normalized_weights