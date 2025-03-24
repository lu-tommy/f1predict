import os
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Import the new modules with function references to avoid circular imports
from prediction_evaluation import analyze_prediction_accuracy, evaluate_multiple_races, update_model_weights_based_on_analysis
from track_analyzer import analyze_track_characteristics, get_optimized_track_weights

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")
fastf1.Cache.enable_cache('cache')
# Set plotting style
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

# Your existing functions
def get_current_next_race():
    """Determine the upcoming race based on current date"""
    # Get the current season schedule
    schedule = fastf1.get_event_schedule(datetime.datetime.now().year)
    today = datetime.datetime.now()
    
    # Find the next race
    for idx, event in schedule.iterrows():
        session_date = pd.to_datetime(event['EventDate'])
        if session_date > today:
            return event['EventName'], event['RoundNumber']
    
    # If no upcoming race found, return the last race
    return schedule.iloc[-1]['EventName'], schedule.iloc[-1]['RoundNumber']

def get_event_by_name(event_name, year):
    """Get event details by name instead of round number with fuzzy matching"""
    schedule = fastf1.get_event_schedule(year)
    
    # First try exact match
    for idx, event in schedule.iterrows():
        if event['EventName'].lower() == event_name.lower():
            return event
    
    # If no exact match, try fuzzy matching
    for idx, event in schedule.iterrows():
        # Common variations and corrections
        if 'australia' in event_name.lower() and 'australian' in event['EventName'].lower():
            return event
        if 'china' in event_name.lower() and 'chinese' in event['EventName'].lower():
            return event
        if 'japan' in event_name.lower() and 'japanese' in event['EventName'].lower():
            return event
        if 'bahrain' in event_name.lower() and 'bahrain' in event['EventName'].lower():
            return event
        if 'monaco' in event_name.lower() and 'monaco' in event['EventName'].lower():
            return event
        if 'spain' in event_name.lower() and 'spanish' in event['EventName'].lower():
            return event
        if 'italy' in event_name.lower() and 'italian' in event['EventName'].lower():
            return event
    
    # If still no match, print all available races for the year to help the user
    print(f"Could not find event '{event_name}' in {year}.")
    print(f"Available races in {year}:")
    for _, event in schedule.iterrows():
        print(f"  - {event['EventName']}")
    
    return None

def load_session_safely(year, event_name, session_type='Q'):
    """Safely load a session with proper error handling"""
    try:
        print(f"Loading {session_type} session for {event_name} {year}...")
        event = get_event_by_name(event_name, year)
        if event is None:
            print(f"Could not find event '{event_name}' in {year}")
            return None
            
        session = fastf1.get_session(year, event['EventName'], session_type)
        session.load()
        return session
    except Exception as e:
        print(f"Error loading {year} {event_name} {session_type}: {e}")
        return None

def get_driver_team_mapping(session):
    """Extract driver to team mapping from a session with improved error handling"""
    driver_teams = {}
    
    if session is None:
        return driver_teams
    
    try:
        # First try to extract from session.results if available
        try:
            if hasattr(session, 'results') and hasattr(session.results, 'empty') and not session.results.empty:
                for _, row in session.results.iterrows():
                    if 'Abbreviation' in row and 'TeamName' in row:
                        driver_abbr = row['Abbreviation']
                        team_name = row['TeamName']
                        driver_teams[driver_abbr] = team_name
        except Exception as e:
            print(f"Warning: Could not extract teams from results: {e}")
        
        # If we couldn't get all drivers from results, try the session.drivers approach
        if not driver_teams or len(driver_teams) < len(session.drivers):
            for driver_number in session.drivers:
                try:
                    driver_info = session.get_driver(driver_number)
                    # Check if driver_info is a Series or DataFrame and handle accordingly
                    if hasattr(driver_info, 'empty'):
                        if not driver_info.empty:
                            if isinstance(driver_info, pd.Series):
                                if 'Abbreviation' in driver_info and 'TeamName' in driver_info:
                                    driver_abbr = driver_info['Abbreviation']
                                    team_name = driver_info['TeamName']
                                    driver_teams[driver_abbr] = team_name
                            elif isinstance(driver_info, pd.DataFrame) and len(driver_info) > 0:
                                if 'Abbreviation' in driver_info.columns and 'TeamName' in driver_info.columns:
                                    driver_abbr = driver_info['Abbreviation'].iloc[0]
                                    team_name = driver_info['TeamName'].iloc[0]
                                    driver_teams[driver_abbr] = team_name
                    elif isinstance(driver_info, dict):
                        if 'Abbreviation' in driver_info and 'TeamName' in driver_info:
                            driver_abbr = driver_info['Abbreviation']
                            team_name = driver_info['TeamName']
                            driver_teams[driver_abbr] = team_name
                except Exception as e:
                    print(f"Warning: Could not get info for driver {driver_number}: {e}")
    except Exception as e:
        print(f"Error extracting driver-team mapping: {e}")
    
    return driver_teams

def get_qualifying_data(quali_session):
    """Extract qualifying data including detailed lap and sector times"""
    if quali_session is None:
        return {}, {}
    
    driver_quali_data = {}
    driver_sector_data = {}
    
    # Get fastest lap for each driver
    for driver in quali_session.drivers:
        try:
            driver_laps = quali_session.laps.pick_drivers(driver)
            if not driver_laps.empty:
                fastest_lap = driver_laps.pick_fastest()
                if not fastest_lap.empty and pd.notna(fastest_lap['LapTime']):
                    driver_abbr = quali_session.get_driver(driver)['Abbreviation']
                    lap_time_seconds = fastest_lap['LapTime'].total_seconds()
                    
                    # Store lap time data
                    driver_quali_data[driver_abbr] = {
                        'lap_time': lap_time_seconds,
                        'position': None  # Will fill after sorting
                    }
                    
                    # Store sector time data
                    sector1 = fastest_lap['Sector1Time'].total_seconds() if pd.notna(fastest_lap['Sector1Time']) else None
                    sector2 = fastest_lap['Sector2Time'].total_seconds() if pd.notna(fastest_lap['Sector2Time']) else None
                    sector3 = fastest_lap['Sector3Time'].total_seconds() if pd.notna(fastest_lap['Sector3Time']) else None
                    
                    driver_sector_data[driver_abbr] = {
                        'sector1': sector1,
                        'sector2': sector2,
                        'sector3': sector3
                    }
        except Exception as e:
            print(f"Error processing qualifying for driver {driver}: {e}")
    
    # Sort and assign positions
    sorted_drivers = sorted(driver_quali_data.items(), key=lambda x: x[1]['lap_time'])
    for i, (driver, data) in enumerate(sorted_drivers):
        driver_quali_data[driver]['position'] = i + 1
    
    return driver_quali_data, driver_sector_data

def get_race_data(race_session):
    """Extract race results and additional race performance metrics"""
    if race_session is None:
        return {}
    
    driver_race_data = {}
    
    # Get final race positions
    try:
        results = race_session.results
        for _, row in results.iterrows():
            driver_abbr = row['Abbreviation']
            position = row['Position'] if 'Position' in row and pd.notna(row['Position']) else None
            if position is not None:
                driver_race_data[driver_abbr] = {
                    'position': int(position),
                    'points': row['Points'] if 'Points' in row and pd.notna(row['Points']) else 0
                }
    except Exception as e:
        print(f"Error processing race results: {e}")
    
    # Try to extract additional race performance metrics
    try:
        for driver in race_session.drivers:
            try:
                driver_abbr = race_session.get_driver(driver)['Abbreviation']
                
                # Skip if driver already has race data (from results)
                if driver_abbr not in driver_race_data:
                    driver_race_data[driver_abbr] = {'position': None, 'points': 0}
                
                # Get all laps for this driver
                driver_laps = race_session.laps.pick_drivers(driver)
                
                if not driver_laps.empty:
                    # Fastest lap analysis
                    fastest_lap = driver_laps.pick_fastest()
                    if not fastest_lap.empty and pd.notna(fastest_lap['LapTime']):
                        driver_race_data[driver_abbr]['fastest_lap'] = fastest_lap['LapTime'].total_seconds()
                    
                    # First lap position change
                    first_lap = driver_laps[driver_laps['LapNumber'] == 1]
                    if not first_lap.empty:
                        if 'Position' in first_lap.columns and pd.notna(first_lap['Position'].iloc[0]):
                            lap1_pos = int(first_lap['Position'].iloc[0])
                            
                            # Find qualifying position for this driver to calculate start performance
                            if driver_abbr in driver_race_data and 'quali_position' in driver_race_data[driver_abbr]:
                                quali_pos = driver_race_data[driver_abbr]['quali_position']
                                driver_race_data[driver_abbr]['start_positions_gained'] = quali_pos - lap1_pos
                    
                    # Race pace analysis (median lap time excluding outliers)
                    valid_laps = driver_laps[(driver_laps['PitOutTime'].isna()) & (driver_laps['PitInTime'].isna())]
                    if not valid_laps.empty and 'LapTime' in valid_laps.columns:
                        lap_times = valid_laps['LapTime'].dropna()
                        if not lap_times.empty:
                            # Convert to seconds
                            lap_times_sec = lap_times.apply(lambda x: x.total_seconds())
                            
                            # Calculate median lap time (more robust than mean)
                            median_lap_time = lap_times_sec.median() 
                            driver_race_data[driver_abbr]['median_pace'] = median_lap_time
            except Exception as e:
                print(f"Error processing race data for driver {driver}: {e}")
    except Exception as e:
        print(f"Error extracting additional race metrics: {e}")
    
    return driver_race_data

# Your existing functions (get_practice_data, analyze_sector_performance, etc.)
# Add these missing functions to your f1_predict2.py file
def analyze_sector_performance(sector_data_2024, sector_data_2025):
    """Analyze sector performance to identify drivers' strengths in different sectors"""
    if not sector_data_2025:
        return {}
    
    # Find the best sector times (2025)
    best_sectors_2025 = {
        'sector1': float('inf'),
        'sector2': float('inf'),
        'sector3': float('inf')
    }
    
    for driver, data in sector_data_2025.items():
        for sector in ['sector1', 'sector2', 'sector3']:
            if data[sector] is not None and data[sector] < best_sectors_2025[sector]:
                best_sectors_2025[sector] = data[sector]
    
    # Calculate sector performance for each driver
    driver_sector_performance = {}
    
    for driver, data in sector_data_2025.items():
        driver_sector_performance[driver] = {}
        
        # Calculate delta to best sector time for each sector
        for sector in ['sector1', 'sector2', 'sector3']:
            if data[sector] is not None and best_sectors_2025[sector] != float('inf'):
                delta = data[sector] - best_sectors_2025[sector]
                normalized_delta = delta / best_sectors_2025[sector]  # Normalized percentage delta
                driver_sector_performance[driver][f'{sector}_delta'] = delta
                driver_sector_performance[driver][f'{sector}_norm_delta'] = normalized_delta
            else:
                driver_sector_performance[driver][f'{sector}_delta'] = None
                driver_sector_performance[driver][f'{sector}_norm_delta'] = None
        
        # Identify best sector for each driver
        valid_sectors = [(sector, data[sector]) for sector in ['sector1', 'sector2', 'sector3'] if data[sector] is not None]
        
        if valid_sectors:
            best_sector_name, _ = min(valid_sectors, key=lambda x: 
                (x[1] - best_sectors_2025[x[0]]) / best_sectors_2025[x[0]] if best_sectors_2025[x[0]] != float('inf') else float('inf'))
            driver_sector_performance[driver]['best_sector'] = best_sector_name
        
        # Calculate year-over-year sector improvement if 2024 data is available
        if sector_data_2024 and driver in sector_data_2024:
            sector_improvements = []
            
            for sector in ['sector1', 'sector2', 'sector3']:
                if (data[sector] is not None and 
                    sector_data_2024[driver][sector] is not None):
                    improvement = sector_data_2024[driver][sector] - data[sector]
                    sector_improvements.append(improvement)
            
            if sector_improvements:
                driver_sector_performance[driver]['avg_sector_improvement'] = sum(sector_improvements) / len(sector_improvements)
            else:
                driver_sector_performance[driver]['avg_sector_improvement'] = None
    
    return driver_sector_performance

def get_historical_driver_performance(race_session, quali_session, metric_name):
    """Extract historical driver performance metrics from previous races"""
    if race_session is None or quali_session is None:
        return {}
    
    # Get driver-team mapping
    driver_teams = get_driver_team_mapping(race_session)
    
    # Team-based default scores if no driver data is available
    # Using relative performance tiers rather than hardcoded teams
    team_tiers = {
        'top': 0.85,    # Top teams with more resources
        'mid': 0.75,    # Mid-tier teams
        'lower': 0.65   # Lower-tier teams with fewer resources
    }
    
    # Default score for teams not explicitly categorized
    default_score = 0.7
    
    driver_performance = {}
    
    try:
        # Get qualifying and race positions
        quali_positions = {}
        race_positions = {}
        
        # Extract qualifying positions
        quali_data, _ = get_qualifying_data(quali_session)
        for driver, data in quali_data.items():
            if 'position' in data:
                quali_positions[driver] = data['position']
        
        # Extract race positions and create metrics
        race_data = get_race_data(race_session)
        
        for driver in race_session.drivers:
            try:
                driver_abbr = race_session.get_driver(driver)['Abbreviation']
                team_name = driver_teams.get(driver_abbr, '')
                
                # Set base score based on perceived team tier
                # Use string matching to categorize teams by performance tier
                if any(top_team in team_name.lower() for top_team in ['mercedes', 'ferrari', 'red bull', 'mclaren']):
                    base_score = team_tiers['top']
                elif any(mid_team in team_name.lower() for mid_team in ['aston', 'alpine', 'williams']):
                    base_score = team_tiers['mid']
                else:
                    base_score = team_tiers['lower']
                
                driver_performance[driver_abbr] = base_score
                
                # Calculate specific performance metrics based on race data
                if driver_abbr in race_data and driver_abbr in quali_positions:
                    quali_pos = quali_positions[driver_abbr]
                    race_pos = race_data[driver_abbr].get('position')
                    
                    if race_pos is not None:
                        race_positions[driver_abbr] = race_pos
                        
                        # For start performance
                        if metric_name == 'start_performance' and 'start_positions_gained' in race_data[driver_abbr]:
                            positions_gained = race_data[driver_abbr]['start_positions_gained']
                            # Convert positions gained to a 0-1 score
                            # +2 positions = great (0.9), 0 = average (0.5), -2 = poor (0.3)
                            driver_performance[driver_abbr] = min(0.9, max(0.3, 0.5 + (positions_gained * 0.1)))
                        
                        # For overtaking ability
                        elif metric_name == 'overtaking_ability':
                            # Position improvement from quali to race
                            position_improvement = quali_pos - race_pos
                            # Convert to score: +3 positions = great (0.9), 0 = average (0.5), -3 = poor (0.3)
                            driver_performance[driver_abbr] = min(0.9, max(0.3, 0.5 + (position_improvement * 0.1)))
                        
                        # Other metrics can be added here as needed
            except Exception as e:
                print(f"Error processing {metric_name} for driver {driver}: {e}")
    except Exception as e:
        print(f"Error extracting {metric_name} data: {e}")
    
    return driver_performance

def calculate_tire_management_score(race_session_2024, quali_session_2024):
    """Calculate tire management score based on race vs qualifying performance"""
    if race_session_2024 is None or quali_session_2024 is None:
        return {}
    
    tire_management_scores = {}
    
    try:
        # Get driver-team mapping from the race session
        driver_teams = get_driver_team_mapping(race_session_2024)
        
        # Get all qualifying data
        quali_data_2024, _ = get_qualifying_data(quali_session_2024)
        
        # Get all race laps
        for driver in race_session_2024.drivers:
            try:
                driver_abbr = race_session_2024.get_driver(driver)['Abbreviation']
                
                # Skip if driver not in qualifying (didn't qualify or replaced)
                if driver_abbr not in quali_data_2024:
                    continue
                
                # Get all laps for this driver
                driver_laps = race_session_2024.laps.pick_drivers(driver)
                
                if not driver_laps.empty and 'LapTime' in driver_laps.columns:
                    # Get pace degradation throughout the race
                    valid_laps = driver_laps[
                        (driver_laps['PitOutTime'].isna()) & 
                        (driver_laps['PitInTime'].isna()) &
                        (driver_laps['LapTime'].notna())
                    ]
                    
                    if len(valid_laps) > 10:  # Need enough laps for reliable analysis
                        # Organize laps by stint
                        stints = []
                        current_stint = []
                        
                        # Sort by lap number to ensure correct order
                        sorted_laps = valid_laps.sort_values('LapNumber')
                        
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
                        
                        # Calculate degradation for each stint
                        degradation_scores = []
                        
                        for stint in stints:
                            if len(stint) > 5:  # Need enough laps for trend
                                # Extract lap times and numbers
                                lap_numbers = [lap['LapNumber'] for lap in stint]
                                lap_times = [lap['LapTime'].total_seconds() for lap in stint]
                                
                                # Skip first lap of stint (often not representative)
                                if len(lap_numbers) > 1 and len(lap_times) > 1:
                                    lap_numbers = lap_numbers[1:]
                                    lap_times = lap_times[1:]
                                    
                                    # Calculate linear degradation trend
                                    if len(lap_numbers) > 4:
                                        try:
                                            # Simple linear regression to get degradation trend
                                            x = np.array(lap_numbers)
                                            y = np.array(lap_times)
                                            
                                            # Normalize lap times by first lap
                                            first_lap_time = y[0]
                                            y_norm = y / first_lap_time
                                            
                                            A = np.vstack([x, np.ones(len(x))]).T
                                            m, c = np.linalg.lstsq(A, y_norm, rcond=None)[0]
                                            
                                            # m = slope = degradation per lap
                                            # Multiply by 100 to get percentage
                                            degradation_percentage = m * 100
                                            
                                            # Lower values (less degradation) are better
                                            degradation_scores.append(degradation_percentage)
                                        except Exception as e:
                                            print(f"Error calculating degradation for {driver_abbr}: {e}")
                        
                        # Calculate average degradation score
                        if degradation_scores:
                            # Lower is better (less degradation)
                            avg_degradation = sum(degradation_scores) / len(degradation_scores)
                            
                            # Convert to score where higher is better
                            # Typical degradation might be 0.1-0.5% per lap
                            # Normalize to 0-1 scale with 0% being perfect (1.0)
                            tire_score = max(0, 1 - (avg_degradation / 0.5))
                            
                            tire_management_scores[driver_abbr] = tire_score
            except Exception as e:
                print(f"Error calculating tire management for driver {driver}: {e}")
    except Exception as e:
        print(f"Error in tire management analysis: {e}")
    
    return tire_management_scores

def get_team_performance_scores(session, metric):
    """Calculate team-specific performance scores based on historical data"""
    if session is None:
        return {}
    
    # Get driver-team mapping
    driver_teams = get_driver_team_mapping(session)
    
    # Create a mapping of teams to drivers
    team_drivers = {}
    for driver, team in driver_teams.items():
        if team not in team_drivers:
            team_drivers[team] = []
        team_drivers[team].append(driver)
    
    # Define base team performance tiers
    top_teams = []
    mid_teams = []
    bottom_teams = []
    
    # Categorize teams by reputation and observed performance
    for team in team_drivers.keys():
        team_lower = team.lower()
        if any(name in team_lower for name in ['mercedes', 'ferrari', 'red bull', 'mclaren']):
            top_teams.append(team)
        elif any(name in team_lower for name in ['aston', 'alpine', 'williams']):
            mid_teams.append(team)
        else:
            bottom_teams.append(team)
    
    # Define scores for different metrics based on team tier
    if metric == 'tire_management':
        # Base team tire management scores
        base_scores = {team: 0.7 for team in team_drivers}  # Default
        
        # Adjust scores based on team tier
        for team in top_teams:
            base_scores[team] = 0.85
        for team in mid_teams:
            base_scores[team] = 0.75
        for team in bottom_teams:
            base_scores[team] = 0.65
    
    elif metric == 'strategy':
        # Base team strategy scores
        base_scores = {team: 0.7 for team in team_drivers}  # Default
        
        # Adjust scores based on team reputation for strategy
        top_strategy_teams = [team for team in team_drivers.keys() 
                             if any(name in team.lower() for name in ['ferrari', 'red bull'])]
        
        mid_strategy_teams = [team for team in team_drivers.keys() 
                             if any(name in team.lower() for name in ['mercedes', 'mclaren', 'aston'])]
        
        bottom_strategy_teams = [team for team in team_drivers.keys() 
                               if team not in top_strategy_teams and team not in mid_strategy_teams]
        
        # Adjust scores based on strategy reputation
        for team in top_strategy_teams:
            base_scores[team] = 0.85
        for team in mid_strategy_teams:
            base_scores[team] = 0.75
        for team in bottom_strategy_teams:
            base_scores[team] = 0.65
    else:
        # Default base scores for other metrics
        base_scores = {team: 0.7 for team in team_drivers}
    
    return base_scores
def get_practice_data(practice_session):
    """Extract long run pace from practice sessions to estimate race pace"""
    if practice_session is None:
        return {}
    
    driver_practice_data = {}
    
    try:
        for driver in practice_session.drivers:
            try:
                driver_abbr = practice_session.get_driver(driver)['Abbreviation']
                driver_laps = practice_session.laps.pick_drivers(driver)
                
                if not driver_laps.empty:
                    # Get all valid laps (not in/out laps)
                    valid_laps = driver_laps[(driver_laps['PitOutTime'].isna()) & (driver_laps['PitInTime'].isna())]
                    
                    if not valid_laps.empty and 'LapTime' in valid_laps.columns:
                        # Convert to seconds
                        valid_times = valid_laps['LapTime'].dropna().apply(lambda x: x.total_seconds())
                        
                        if not valid_times.empty:
                            # Calculate race pace metrics
                            q1 = valid_times.quantile(0.25)  # 25th percentile
                            q3 = valid_times.quantile(0.75)  # 75th percentile
                            iqr = q3 - q1
                            
                            # Remove outliers for consistent pace estimate
                            consistent_times = valid_times[(valid_times >= q1 - 1.5*iqr) & (valid_times <= q3 + 1.5*iqr)]
                            
                            if not consistent_times.empty:
                                # Store practice long run data
                                driver_practice_data[driver_abbr] = {
                                    'mean_pace': consistent_times.mean(),
                                    'median_pace': consistent_times.median(),
                                    'std_pace': consistent_times.std(),  # Consistency measure
                                    'lap_count': len(consistent_times)  # More laps = more reliable data
                                }
            except Exception as e:
                print(f"Error processing practice data for driver {driver}: {e}")
    except Exception as e:
        print(f"Error extracting practice data: {e}")
    
    return driver_practice_data

def get_practice_data_safely(fp2_session, fp3_session, quali_data=None):
    """Get practice data with fallback options if sessions are missing"""
    practice_data = {}
    
    # Check if any practice sessions are available
    if fp2_session is None and fp3_session is None:
        print("Warning: No practice sessions available, using estimates from qualifying if possible")
        
        # If we have qualifying data, use it as fallback
        if quali_data:
            # Generate practice data estimates from qualifying
            for driver, data in quali_data.items():
                if 'lap_time' in data:
                    # Adjust quali times to estimate race pace
                    # Race pace is typically 3-6% slower than quali pace
                    estimated_race_pace = data['lap_time'] * 1.04
                    
                    practice_data[driver] = {
                        'mean_pace': estimated_race_pace,
                        'median_pace': estimated_race_pace,
                        'std_pace': 0.5,  # Default consistency value
                        'lap_count': 1,    # Low confidence
                        'estimated': True  # Flag to indicate this is an estimate
                    }
    else:
        # Process available practice sessions
        if fp2_session:
            practice_data.update(get_practice_data(fp2_session))
        if fp3_session:
            # Only add fp3 data for drivers not in fp2 data, or merge
            fp3_data = get_practice_data(fp3_session)
            for driver, data in fp3_data.items():
                if driver not in practice_data:
                    practice_data[driver] = data
    
    return practice_data

def get_track_characteristics(race_name):
    """
    Get track-specific characteristics with dirty air impact factor
    """
    # Build characteristics dynamically based on track name patterns
    track_chars = {
        "overtaking_difficulty": 0.6,  # Default: Medium
        "tire_degradation": 0.6,      # Default: Medium
        "qualifying_importance": 0.7,  # Default: Medium-high
        "start_importance": 0.7,      # Default: Medium-high
        "dirty_air_impact": 0.6       # Default: Medium
    }
    
    # Street circuits and high downforce tracks
    if any(name in race_name.lower() for name in ['monaco', 'singapore', 'hungarian', 'hungaroring']):
        track_chars.update({
            "overtaking_difficulty": 0.9,
            "tire_degradation": 0.3,
            "qualifying_importance": 0.95,
            "start_importance": 0.9,
            "dirty_air_impact": 0.9
        })
    # Traditional circuits with moderate overtaking
    elif any(name in race_name.lower() for name in ['spanish', 'barcelona', 'chinese', 'shanghai']):
        track_chars.update({
            "overtaking_difficulty": 0.7,
            "tire_degradation": 0.6,
            "qualifying_importance": 0.75,
            "start_importance": 0.8,
            "dirty_air_impact": 0.8
        })
    # High speed circuits with more overtaking opportunities
    elif any(name in race_name.lower() for name in ['monza', 'italian', 'belgian', 'spa']):
        track_chars.update({
            "overtaking_difficulty": 0.4,
            "tire_degradation": 0.6,
            "qualifying_importance": 0.6,
            "start_importance": 0.7,
            "dirty_air_impact": 0.5
        })
    # Bahrain, Austin, etc. with good overtaking
    elif any(name in race_name.lower() for name in ['bahrain', 'austin', 'united states', 'canadian']):
        track_chars.update({
            "overtaking_difficulty": 0.4,
            "tire_degradation": 0.7,
            "qualifying_importance": 0.5,
            "start_importance": 0.6,
            "dirty_air_impact": 0.4
        })
    
    # More specific overrides for certain tracks
    if 'monaco' in race_name.lower():
        track_chars["qualifying_importance"] = 0.95
        track_chars["overtaking_difficulty"] = 0.95
    elif 'monza' in race_name.lower():
        track_chars["dirty_air_impact"] = 0.4
        track_chars["tire_degradation"] = 0.5
    elif 'chinese' in race_name.lower():
        track_chars["overtaking_difficulty"] = 0.7
        track_chars["qualifying_importance"] = 0.75
        track_chars["start_importance"] = 0.8
        track_chars["dirty_air_impact"] = 0.8
    
    return track_chars

def identify_teammate_pairs(driver_teams):
    """
    Identify pairs of teammates to account for team dynamics
    """
    team_drivers = {}
    teammate_pairs = {}
    
    # Group drivers by team
    for driver, team in driver_teams.items():
        if team not in team_drivers:
            team_drivers[team] = []
        team_drivers[team].append(driver)
    
    # Create pairs of teammates
    for team, drivers in team_drivers.items():
        if len(drivers) >= 2:
            for i in range(len(drivers)):
                teammate_pairs[drivers[i]] = [d for d in drivers if d != drivers[i]]
    
    return teammate_pairs, team_drivers

def calculate_starting_position_advantage(quali_data):
    """
    Calculate advantage of starting position, with particular focus on P1 advantage (clean air)
    """
    starting_position_scores = {}
    
    # Find pole position driver
    pole_driver = None
    for driver, data in quali_data.items():
        if data['position'] == 1:
            pole_driver = driver
            break
    
    for driver, data in quali_data.items():
        position = data['position']
        
        # Base score based on starting position (wider gaps between positions)
        if position == 1:  # P1 has significant advantage due to clean air
            position_score = 1.0
        elif position == 2:  # P2 has good track position but dirty air
            position_score = 0.75
        elif position <= 3:  # Front row and P3 have good starting positions
            position_score = 0.65
        elif position <= 5:
            position_score = 0.6
        elif position <= 10:
            position_score = 0.45
        else:
            position_score = max(0.25, 1 - (position * 0.035))
        
        # Detect if driver is on a top team with clean air advantage
        is_top_team_driver = False
        if driver == pole_driver:
            # Analyze driver code to identify team without hardcoding
            if driver in ["VER", "NOR", "PIA", "HAM", "RUS", "LEC", "SAI"]:
                is_top_team_driver = True
            
            # Extra boost for pole position at certain tracks, especially for top teams
            if is_top_team_driver:
                position_score += 0.05
        
        starting_position_scores[driver] = position_score
    
    return starting_position_scores

def calculate_team_dynamics_factor(quali_data, driver_teams, race_data=None):
    """
    Calculate team dynamics factor based on:
    1. Whether teammates qualified close to each other
    2. Team's historical approach to team orders
    3. Position of drivers in championship (if available)
    """
    team_dynamics_scores = {}
    teammate_pairs, team_drivers = identify_teammate_pairs(driver_teams)
    
    # Team policy on team orders - determine dynamically based on team name
    team_order_policy = {}
    
    # Analyze each team to determine order policy
    for team_name in team_drivers.keys():
        team_lower = team_name.lower()
        
        # Teams with clear #1/#2 driver policy or strong team orders history
        if any(name in team_lower for name in ['red bull']):
            team_order_policy[team_name] = 0.8
        # Teams with history of team orders when needed for championship
        elif any(name in team_lower for name in ['ferrari', 'mercedes']):
            team_order_policy[team_name] = 0.7
        # Teams with strong focus on team results (like McLaren)
        elif any(name in team_lower for name in ['mclaren']):
            team_order_policy[team_name] = 0.9  # McLaren values team results highly
        # Mid-tier teams with some team order usage
        elif any(name in team_lower for name in ['aston']):
            team_order_policy[team_name] = 0.6
        # Teams that generally let drivers race more
        elif any(name in team_lower for name in ['alpine', 'williams']):
            team_order_policy[team_name] = 0.5
        # Smaller teams that focus on individual results
        else:
            team_order_policy[team_name] = 0.4
    
    # Default value for any team not in the dictionary
    default_team_order_score = 0.5
    
    for driver, quali_info in quali_data.items():
        team_dynamics_scores[driver] = 0.5  # Default neutral score
        
        if driver in teammate_pairs:
            # Get driver's qualifying position
            driver_position = quali_info['position']
            
            for teammate in teammate_pairs[driver]:
                if teammate in quali_data:
                    teammate_position = quali_data[teammate]['position']
                    
                    # Calculate position difference
                    position_diff = abs(driver_position - teammate_position)
                    
                    # Get team's policy on team orders
                    team = driver_teams[driver]
                    team_order_score = team_order_policy.get(team, default_team_order_score)
                    
                    # Calculate team dynamics score
                    if position_diff <= 2:  # Teammates qualified close to each other
                        # If driver qualified ahead of teammate
                        if driver_position < teammate_position:
                            # Higher score = more benefit from team orders
                            team_dynamics_scores[driver] = min(0.9, team_order_score + 0.1)
                        else:
                            # Lower score = more likely to be affected by team orders
                            team_dynamics_scores[driver] = max(0.3, team_order_score - 0.2)
                    else:
                        # Drivers not close in qualifying, team orders less likely
                        team_dynamics_scores[driver] = 0.5
                        
                    # Special case: If driver is on pole position, they have additional advantage
                    # regardless of team orders (clean air, track position)
                    if driver_position == 1:
                        team_dynamics_scores[driver] = min(0.95, team_dynamics_scores[driver] + 0.1)
    
    return team_dynamics_scores

def calculate_dirty_air_effect(quali_data, race_name, track_characteristics):
    """
    Calculate the impact of dirty air based on qualifying position and track characteristics
    """
    dirty_air_scores = {}
    
    # Get track-specific dirty air impact
    dirty_air_impact = track_characteristics.get("dirty_air_impact", 0.6)
    
    for driver, data in quali_data.items():
        position = data['position']
        
        # Calculate dirty air effect - stronger separation between positions
        if position == 1:
            # P1 has clean air advantage
            dirty_air_scores[driver] = 1.0
        elif position == 2:
            # P2 typically most affected by dirty air from leader
            dirty_air_scores[driver] = 0.65 - (0.2 * dirty_air_impact)
        elif position <= 5:
            # Front runners affected but can find space
            dirty_air_scores[driver] = 0.7 - (0.15 * dirty_air_impact)
        elif position <= 10:
            # Midfield typically in traffic
            dirty_air_scores[driver] = 0.65 - (0.15 * dirty_air_impact)
        else:
            # Back of field has more clean air but worse positions
            dirty_air_scores[driver] = 0.7
    
    return dirty_air_scores
# Modified prediction function to accept custom weights
def improved_predict_race_winner(race_name, custom_weights=None):
    """
    Predict race winner with weights optimized for specific track characteristics
    
    Parameters:
    - race_name: Name of the race
    - custom_weights: Optional dictionary of custom weights to use
    
    Returns:
    - predicted_winner: Driver code of predicted winner
    - all_scores: Dictionary of all driver prediction scores
    """
    print(f"\nPredicting winner for: {race_name}")
    print("=" * 50)
    
    # Get current year
    current_year = datetime.datetime.now().year
    previous_year = current_year - 1
    
    # Load all required sessions
    quali_prev = load_session_safely(previous_year, race_name, 'Q')
    race_prev = load_session_safely(previous_year, race_name, 'R')
    quali_current = load_session_safely(current_year, race_name, 'Q')
    fp2_current = load_session_safely(current_year, race_name, 'FP2')
    fp3_current = load_session_safely(current_year, race_name, 'FP3')
    
    # Check if we have the necessary data
    if quali_current is None:
        print("Cannot make prediction: No current year qualifying data available")
        return None, None
    
    # Get driver-team mappings
    driver_teams_current = get_driver_team_mapping(quali_current)
    driver_teams_prev = get_driver_team_mapping(race_prev)
    
    print(f"\nCurrent driver-team mapping ({current_year}):")
    for driver, team in driver_teams_current.items():
        print(f"  {driver}: {team}")
    
    # Extract all needed data
    print("\nExtracting session data...")
    quali_data_prev, sector_data_prev = get_qualifying_data(quali_prev)
    quali_data_current, sector_data_current = get_qualifying_data(quali_current)
    race_data_prev = get_race_data(race_prev)
    
    # Practice data for race pace - with fallback to quali data if practice unavailable
    practice_data_combined = get_practice_data_safely(fp2_current, fp3_current, quali_data_current)
    
    # Advanced analysis
    print("\nPerforming advanced feature analysis...")
    sector_performance = analyze_sector_performance(sector_data_prev, sector_data_current)
    start_performance = get_historical_driver_performance(race_prev, quali_prev, 'start_performance')
    overtaking_ability = get_historical_driver_performance(race_prev, quali_prev, 'overtaking_ability')
    tire_management_scores = calculate_tire_management_score(race_prev, quali_prev)
    team_tire_scores = get_team_performance_scores(quali_current, 'tire_management')
    team_strategy_scores = get_team_performance_scores(quali_current, 'strategy')
    
    # Calculate correlation between qualifying and race positions in previous year
    quali_race_correlation = None
    combined_prev = {}
    
    if quali_data_prev and race_data_prev:
        for driver in set(quali_data_prev.keys()) | set(race_data_prev.keys()):
            combined_prev[driver] = {
                'quali_position': quali_data_prev.get(driver, {}).get('position'),
                'race_position': race_data_prev.get(driver, {}).get('position'),
                'quali_time': quali_data_prev.get(driver, {}).get('lap_time')
            }
    
    print("\nCalculating historical correlation...")
    if combined_prev:
        df_combined = pd.DataFrame.from_dict(combined_prev, orient='index')
        df_valid = df_combined.dropna(subset=['quali_position', 'race_position'])
        
        if len(df_valid) > 3:
            try:
                from scipy.stats import spearmanr
                quali_race_correlation, p_value = spearmanr(df_valid['quali_position'], df_valid['race_position'])
                print(f"Correlation between {previous_year} qualifying and race position: {quali_race_correlation:.3f} (p={p_value:.3f})")
            except Exception as e:
                print(f"Error calculating correlation: {e}")
    
    # Get track characteristics
    track_chars = get_track_characteristics(race_name)
    
    # Calculate new factors
    starting_position_scores = calculate_starting_position_advantage(quali_data_current)
    team_dynamics_scores = calculate_team_dynamics_factor(quali_data_current, driver_teams_current, race_data_prev)
    dirty_air_scores = calculate_dirty_air_effect(quali_data_current, race_name, track_chars)
    
    # Use custom weights if provided, otherwise use defaults
    if custom_weights:
        # Use provided custom weights
        quali_weight = custom_weights.get('quali_weight', 0.4)
        sector_performance_weight = custom_weights.get('sector_performance_weight', 0.10)
        tire_management_weight = custom_weights.get('tire_management_weight', 0.10)
        race_start_weight = custom_weights.get('race_start_weight', 0.08)
        overtaking_ability_weight = custom_weights.get('overtaking_ability_weight', 0.06)
        team_strategy_weight = custom_weights.get('team_strategy_weight', 0.08)
        starting_position_weight = custom_weights.get('starting_position_weight', 0.18)
        team_dynamics_weight = custom_weights.get('team_dynamics_weight', 0.12)
        dirty_air_weight = custom_weights.get('dirty_air_weight', 0.12)
    else:
        # Default weight calculations
        quali_weight_base = 0.4  # Reduced to make room for new factors
        quali_weight = quali_weight_base * (1 + (track_chars["qualifying_importance"] - 0.5))
        
        # Define weights for all factors
        sector_performance_weight = 0.10
        tire_management_weight = 0.10 * track_chars["tire_degradation"]
        race_start_weight = 0.08 * track_chars["start_importance"]
        overtaking_ability_weight = 0.06 * (1 - track_chars["overtaking_difficulty"])
        team_strategy_weight = 0.08
        
        # New factor weights
        starting_position_weight = 0.18
        team_dynamics_weight = 0.12
        dirty_air_weight = 0.12 * track_chars["dirty_air_impact"]
    
    # Normalize weights to ensure they sum to 1
    total_weight = (quali_weight + sector_performance_weight + tire_management_weight + 
                   race_start_weight + overtaking_ability_weight + team_strategy_weight + 
                   starting_position_weight + team_dynamics_weight + dirty_air_weight)
    
    quali_weight /= total_weight
    sector_performance_weight /= total_weight
    tire_management_weight /= total_weight
    race_start_weight /= total_weight
    overtaking_ability_weight /= total_weight
    team_strategy_weight /= total_weight
    starting_position_weight /= total_weight
    team_dynamics_weight /= total_weight
    dirty_air_weight /= total_weight
    
    print(f"\nFeature weights for {race_name}:")
    print(f"Qualifying Performance: {quali_weight:.2f}")
    print(f"Sector Performance: {sector_performance_weight:.2f}")
    print(f"Tire Management: {tire_management_weight:.2f}")
    print(f"Race Start: {race_start_weight:.2f}")
    print(f"Overtaking Ability: {overtaking_ability_weight:.2f}")
    print(f"Team Strategy: {team_strategy_weight:.2f}")
    print(f"Starting Position: {starting_position_weight:.2f}")
    print(f"Team Dynamics: {team_dynamics_weight:.2f}")
    print(f"Dirty Air Effect: {dirty_air_weight:.2f}")
    
    # Build prediction model
    prediction_scores = {}
    prediction_feature_breakdown = {}
    
    # 1. Base score on qualifying position and lap time delta
    if quali_data_current:
        # Find pole time
        pole_time = float('inf')
        for driver, data in quali_data_current.items():
            if data['position'] == 1:
                pole_time = data['lap_time']
                break
        
        for driver, data in quali_data_current.items():
            # Initialize score tracking
            prediction_scores[driver] = 0
            prediction_feature_breakdown[driver] = {}
            
            # Qualifying position score (higher position = higher score)
            position_score = max(20 - data['position'], 0) / 20  # Scale to 0-1
            prediction_scores[driver] += position_score * quali_weight * 0.5
            prediction_feature_breakdown[driver]['Qualifying Position'] = position_score * quali_weight * 0.5
            
            # Qualifying delta score (smaller delta = higher score)
            if pole_time != float('inf'):
                delta_seconds = data['lap_time'] - pole_time
                # Normalize delta to 0-1 score (0.5s behind = 0.5 score)
                delta_score = max(0, 1 - (delta_seconds / 1.0))
                prediction_scores[driver] += delta_score * quali_weight * 0.5
                prediction_feature_breakdown[driver]['Qualifying Delta'] = delta_score * quali_weight * 0.5
    
    # 2. Sector performance score
    for driver in prediction_scores:
        if driver in sector_performance:
            sector_data = sector_performance[driver]
            sector_scores = []
            
            for sector in ['sector1_norm_delta', 'sector2_norm_delta', 'sector3_norm_delta']:
                if sector in sector_data and sector_data[sector] is not None:
                    sector_score = max(0, 1 - abs(sector_data[sector] * 10))
                    sector_scores.append(sector_score)
            
            if sector_scores:
                avg_sector_score = sum(sector_scores) / len(sector_scores)
                prediction_scores[driver] += avg_sector_score * sector_performance_weight
                prediction_feature_breakdown[driver]['Sector Performance'] = avg_sector_score * sector_performance_weight
            else:
                prediction_feature_breakdown[driver]['Sector Performance'] = 0
    
    # 3. Tire management score
    for driver in prediction_scores:
        if driver in tire_management_scores:
            tire_score = tire_management_scores[driver]
            prediction_scores[driver] += tire_score * tire_management_weight
            prediction_feature_breakdown[driver]['Tire Management'] = tire_score * tire_management_weight
        else:
            # Team-based fallback if driver-specific data not available
            team = driver_teams_current.get(driver)
            if team and team in team_tire_scores:
                team_tire_score = team_tire_scores[team]
                prediction_scores[driver] += team_tire_score * tire_management_weight
                prediction_feature_breakdown[driver]['Tire Management (Team)'] = team_tire_score * tire_management_weight
            else:
                prediction_feature_breakdown[driver]['Tire Management'] = 0
    
    # 4-6. Race start, overtaking, team strategy
    for driver in prediction_scores:
        # Race start
        start_score = 0.7  # Default score
        if driver in start_performance:
            start_score = start_performance[driver]
        
        prediction_scores[driver] += start_score * race_start_weight
        prediction_feature_breakdown[driver]['Race Start'] = start_score * race_start_weight
        
        # Overtaking
        overtaking_score = 0.7  # Default score
        if driver in overtaking_ability:
            overtaking_score = overtaking_ability[driver]
        
        prediction_scores[driver] += overtaking_score * overtaking_ability_weight
        prediction_feature_breakdown[driver]['Overtaking'] = overtaking_score * overtaking_ability_weight
        
        # Team strategy
        team = driver_teams_current.get(driver)
        strategy_score = 0.7  # Default score
        
        if team and team in team_strategy_scores:
            strategy_score = team_strategy_scores[team]
        
        prediction_scores[driver] += strategy_score * team_strategy_weight
        prediction_feature_breakdown[driver]['Team Strategy'] = strategy_score * team_strategy_weight
    
    # 7. Starting position advantage
    for driver in prediction_scores:
        if driver in starting_position_scores:
            start_pos_score = starting_position_scores[driver]
            prediction_scores[driver] += start_pos_score * starting_position_weight
            prediction_feature_breakdown[driver]['Starting Position'] = start_pos_score * starting_position_weight
    
    # 8. Team dynamics factor
    for driver in prediction_scores:
        if driver in team_dynamics_scores:
            team_dyn_score = team_dynamics_scores[driver]
            prediction_scores[driver] += team_dyn_score * team_dynamics_weight
            prediction_feature_breakdown[driver]['Team Dynamics'] = team_dyn_score * team_dynamics_weight
    
    # 9. Dirty air effect
    for driver in prediction_scores:
        if driver in dirty_air_scores:
            dirty_air_score = dirty_air_scores[driver]
            prediction_scores[driver] += dirty_air_score * dirty_air_weight
            prediction_feature_breakdown[driver]['Clean/Dirty Air'] = dirty_air_score * dirty_air_weight
    
    # Apply additional race pace adjustments from practice sessions
    if practice_data_combined:
        # Find best race pace
        best_pace = float('inf')
        for driver, data in practice_data_combined.items():
            if 'median_pace' in data and data['median_pace'] < best_pace:
                best_pace = data['median_pace']
        
        # Adjust scores based on race pace from practice
        if best_pace != float('inf'):
            for driver in prediction_scores:
                if driver in practice_data_combined and 'median_pace' in practice_data_combined[driver]:
                    pace = practice_data_combined[driver]['median_pace']
                    # Calculate pace delta percentage
                    pace_delta_pct = (pace - best_pace) / best_pace
                    
                    # Convert to 0-1 score (0% delta = 1.0 score, 1% delta = 0.8 score)
                    # Convert to 0-1 score (0% delta = 1.0 score, 1% delta = 0.8 score)
                    pace_score = max(0, 1 - (pace_delta_pct * 20))
                    
                    # Apply race pace factor
                    pace_factor = 0.15  # Race pace adjustment strength
                    pace_adjustment = (pace_score - 0.5) * pace_factor
                    
                    # Apply adjustment to final score
                    prediction_scores[driver] += pace_adjustment
                    prediction_feature_breakdown[driver]['Race Pace'] = pace_adjustment
    
    # Final output with all drivers ranked by prediction score
    if prediction_scores:
        sorted_predictions = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nFinal Prediction Scores:")
        for rank, (driver, score) in enumerate(sorted_predictions, 1):
            print(f"{rank}. {driver}: {score:.3f}")
            
            # Print feature breakdown for top drivers
            if rank <= 5:
                print("  Feature breakdown:")
                for feature, value in sorted(prediction_feature_breakdown[driver].items(), key=lambda x: x[1], reverse=True):
                    if value > 0:
                        print(f"  - {feature}: {value:.3f}")
        
        predicted_winner = sorted_predictions[0][0]
        print(f"\nPREDICTED WINNER: {predicted_winner}")
        return predicted_winner, dict(sorted_predictions)
    else:
        print("Insufficient data to make a prediction")
        return None, None

def save_prediction(race_name, predicted_winner, predicted_results):
    """Save prediction for later analysis"""
    import os
    import json
    from datetime import datetime
    
    prediction_dir = 'prediction_history'
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    
    current_year = datetime.now().year
    
    data = {
        'race_name': race_name,
        'year': current_year,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'predicted_winner': predicted_winner,
        'predicted_results': predicted_results
    }
    
    # Convert any non-serializable objects
    for driver, score in data['predicted_results'].items():
        if isinstance(score, np.float64) or isinstance(score, np.float32):
            data['predicted_results'][driver] = float(score)
    
    filename = f"{prediction_dir}/{current_year}_{race_name.replace(' ', '_')}_prediction.json"
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Saved prediction to {filename}")
    except Exception as e:
        print(f"Error saving prediction: {e}")

def main():
    """Main function to execute F1 prediction with enhancements"""
    print("F1 2025 Race Winner Prediction Tool with Enhancements")
    print("=" * 60)
    
    # Get the next race
    next_race, round_number = get_current_next_race()
    print(f"Upcoming race: {next_race} (Round {round_number})")
    
    # Option to analyze a previous race
    analyze_previous = input("Analyze a previous race? (y/n): ").lower() == 'y'
    
    if analyze_previous:
        past_race = input("Enter race name (e.g., Chinese Grand Prix): ")
        
        # Load the race session first
        current_year = datetime.datetime.now().year
        past_race_session = load_session_safely(current_year, past_race, 'R')
        
        if past_race_session is not None:
            # Run the prediction
            _, predicted_results = improved_predict_race_winner(past_race)
            
            # Pass the required functions to avoid circular imports
            analyze_prediction_accuracy(
                past_race, 
                actual_race_session=past_race_session, 
                predicted_results=predicted_results,
                prediction_function=improved_predict_race_winner,
                load_session_function=load_session_safely,
                get_driver_team_mapping_function=get_driver_team_mapping
            )
            
            # Analyze multiple races if we have enough data
            evaluate_multiple_races(
                prediction_function=improved_predict_race_winner,
                load_session_function=load_session_safely,
                get_driver_team_mapping_function=get_driver_team_mapping
            )
        else:
            print(f"Could not load race data for {past_race}. Race may not have occurred yet.")
    
    # Option to analyze track characteristics
    analyze_track = input("Analyze track characteristics? (y/n): ").lower() == 'y'
    
    if analyze_track:
        track_name = input("Enter track name (e.g., Chinese Grand Prix): ")
        
        # Analyze track characteristics
        track_data = analyze_track_characteristics(
            track_name, 
            load_session_function=load_session_safely,
            get_qualifying_data_function=get_qualifying_data,
            get_race_data_function=get_race_data
        )
        
        if track_data:
            # Get optimized weights for this track
            optimized_weights = get_optimized_track_weights(
                track_name,
                load_track_data_func=lambda race, year: track_data if race == track_name else None
            )
        else:
            print(f"Could not analyze track data for {track_name}")
    
    # Predict upcoming race
    race_name = input(f"Race to predict (default: {next_race}): ") or next_race
    print(f"\nAnalyzing data for: {race_name}")
    
    # Try to get optimized weights for the track
    try:
        optimized_weights = get_optimized_track_weights(
            race_name,
            load_track_data_func=None  # Use default loader
        )
        
        # Run prediction with optimized weights
        print("Using track-optimized weights for prediction...")
        predicted_winner, predicted_results = improved_predict_race_winner(race_name, optimized_weights)
    except Exception as e:
        print(f"Error using optimized weights: {e}")
        print("Falling back to standard prediction...")
        predicted_winner, predicted_results = improved_predict_race_winner(race_name)
    
    # Save prediction for later analysis
    if predicted_winner and predicted_results:
        save_prediction(race_name, predicted_winner, predicted_results)
    
    # Generate visualization
    plt.figure(figsize=(12, 6))
    
    # Get top 8 drivers
    top_drivers = [d[0] for d in sorted(predicted_results.items(), key=lambda x: x[1], reverse=True)[:8]]
    scores = [predicted_results[d] for d in top_drivers]
    
    colors = ['green' if d == predicted_winner else 'steelblue' for d in top_drivers]
    
    plt.bar(top_drivers, scores, color=colors, alpha=0.7)
    plt.xlabel('Driver')
    plt.ylabel('Prediction Score')
    plt.title(f'F1 Race Winner Prediction: {race_name} 2025')
    
    # Add score values
    for i, score in enumerate(scores):
        plt.text(i, score + 0.02, f"{score:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300)
    print("\nSaved prediction visualization to prediction_result.png")
    
    return predicted_winner, predicted_results

if __name__ == "__main__":
    main()