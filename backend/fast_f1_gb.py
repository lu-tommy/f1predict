"""
F1 Race Predictor - Gradient Boosting with FastF1 Data and Track Database Integration
Optimized for limited data scenarios with pre-defined track characteristics
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load

# Suppress warnings
warnings.filterwarnings("ignore")

# Handle imports with fallbacks for various environments
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = False  # Set to False to prioritize XGBoost for limited data
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import FastF1
try:
    import fastf1
    import fastf1.plotting
    from fastf1.core import Session
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("FastF1 not available. Install with: pip install fastf1")

# Import track database for pre-defined track characteristics
try:
    import track_database
    from track_database import (
        get_track_data, 
        get_optimized_track_weights, 
        setup_tracks_for_season
    )
    TRACK_DB_AVAILABLE = True
    print("Successfully imported track database")
except ImportError:
    TRACK_DB_AVAILABLE = False
    print("Warning: track_database.py not found. Some features will be limited.")

# Import from existing modules if available, otherwise define minimal versions
try:
    from f1predict import (
        load_session_safely, get_qualifying_data, get_race_data,
        get_driver_team_mapping, improved_predict_race_winner
    )
    # Success
    print("Successfully imported functions from f1_predict2")
except ImportError:
    # Define fallback functions
    print("Using fallback functions - f1_predict2 module not found")
    
    def load_session_safely(year, event_name, session_type='Q'):
        """Safely load a session with robust error handling"""
        if not FASTF1_AVAILABLE:
            print(f"FastF1 not available. Cannot load {year} {event_name} {session_type}")
            return None
            
        try:
            print(f"Loading {session_type} session for {event_name} {year}...")
            
            # Try to find event in schedule
            schedule = fastf1.get_event_schedule(year)
            event = None
            
            # Exact match first
            for idx, e in schedule.iterrows():
                if e['EventName'].lower() == event_name.lower():
                    event = e
                    break
            
            # Fuzzy matching if needed
            if event is None:
                for idx, e in schedule.iterrows():
                    # Common variations and simplifications
                    if any(name in event_name.lower() for name in ['australia', 'melbourne']) and 'australia' in e['EventName'].lower():
                        event = e
                        break
                    if 'spain' in event_name.lower() and 'spain' in e['EventName'].lower():
                        event = e
                        break
                    # Add more common variations as needed
                    
            if event is None:
                print(f"Could not find event '{event_name}' in {year}")
                return None
                
            # Load session
            session = fastf1.get_session(year, event['EventName'], session_type)
            session.load()
            return session
        except Exception as e:
            print(f"Error loading {year} {event_name} {session_type}: {e}")
            return None
    
    def get_driver_team_mapping(session):
        """Extract driver to team mapping from session data"""
        driver_teams = {}
        
        if session is None:
            return driver_teams
        
        try:
            # Try multiple approaches to maximize data recovery
            
            # 1. From session results
            if hasattr(session, 'results') and not session.results.empty:
                for _, row in session.results.iterrows():
                    if 'Abbreviation' in row and 'TeamName' in row:
                        driver_teams[row['Abbreviation']] = row['TeamName']
            
            # 2. From driver info
            if len(driver_teams) < len(session.drivers):
                for driver in session.drivers:
                    try:
                        driver_info = session.get_driver(driver)
                        if isinstance(driver_info, pd.Series):
                            if 'Abbreviation' in driver_info and 'TeamName' in driver_info:
                                driver_teams[driver_info['Abbreviation']] = driver_info['TeamName']
                    except:
                        pass
            
            # 3. From laps data
            if hasattr(session, 'laps') and not session.laps.empty:
                if 'Team' in session.laps.columns and 'Driver' in session.laps.columns:
                    for _, lap in session.laps.iterrows():
                        if pd.notna(lap['Team']) and pd.notna(lap['Driver']):
                            driver_teams[lap['Driver']] = lap['Team']
                            
        except Exception as e:
            print(f"Error extracting driver-team mapping: {e}")
        
        return driver_teams
    
    def get_qualifying_data(quali_session):
        """Extract qualifying data including lap and sector times"""
        if quali_session is None:
            return {}, {}
        
        driver_quali_data = {}
        driver_sector_data = {}
        
        try:
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
                    print(f"Error processing qualifying data for driver {driver}: {e}")
            
            # Sort and assign positions
            sorted_drivers = sorted(driver_quali_data.items(), key=lambda x: x[1]['lap_time'])
            for i, (driver, data) in enumerate(sorted_drivers):
                driver_quali_data[driver]['position'] = i + 1
                
        except Exception as e:
            print(f"Error processing qualifying data: {e}")
        
        return driver_quali_data, driver_sector_data
    
    def get_race_data(race_session):
        """Extract race results and performance metrics"""
        if race_session is None:
            return {}
        
        driver_race_data = {}
        
        try:
            # Get final race positions
            if hasattr(race_session, 'results') and not race_session.results.empty:
                for _, row in race_session.results.iterrows():
                    if 'Abbreviation' in row and 'Position' in row:
                        driver_abbr = row['Abbreviation']
                        position = row['Position'] if pd.notna(row['Position']) else None
                        
                        if position is not None:
                            driver_race_data[driver_abbr] = {
                                'position': int(position),
                                'points': row['Points'] if 'Points' in row and pd.notna(row['Points']) else 0
                            }
            
            # Extract additional race metrics
            if hasattr(race_session, 'laps') and not race_session.laps.empty:
                for driver in race_session.drivers:
                    try:
                        driver_abbr = race_session.get_driver(driver)['Abbreviation']
                        
                        # Skip if driver already has race data from results
                        if driver_abbr not in driver_race_data:
                            driver_race_data[driver_abbr] = {'position': None, 'points': 0}
                        
                        # Get laps for this driver
                        driver_laps = race_session.laps.pick_drivers(driver)
                        
                        if not driver_laps.empty:
                            # Fastest lap
                            fastest_lap = driver_laps.pick_fastest()
                            if not fastest_lap.empty and pd.notna(fastest_lap['LapTime']):
                                driver_race_data[driver_abbr]['fastest_lap'] = fastest_lap['LapTime'].total_seconds()
                            
                            # Race pace (median non-pit lap)
                            non_pit_laps = driver_laps[(driver_laps['PitInTime'].isna()) & (driver_laps['PitOutTime'].isna())]
                            if not non_pit_laps.empty:
                                lap_times = non_pit_laps['LapTime'].dropna()
                                if not lap_times.empty:
                                    median_time = lap_times.median().total_seconds()
                                    driver_race_data[driver_abbr]['median_pace'] = median_time
                    except Exception as e:
                        print(f"Error processing race data for driver {driver}: {e}")
                        
        except Exception as e:
            print(f"Error processing race data: {e}")
        
        return driver_race_data
    
    def calculate_starting_position_advantage(quali_data):
        """Calculate advantage of starting position"""
        starting_position_scores = {}
        
        for driver, data in quali_data.items():
            position = data['position']
            
            # Calculate advantage factor based on position
            if position == 1:  # Pole
                score = 1.0
            elif position == 2:  # Front row
                score = 0.8
            elif position <= 3:
                score = 0.7
            elif position <= 5:
                score = 0.6
            elif position <= 10:
                score = 0.5
            else:
                score = max(0.3, 1 - (position * 0.03))
                
            starting_position_scores[driver] = score
            
        return starting_position_scores

# Function for calculating starting position advantage (if not imported from f1_predict2)
if 'calculate_starting_position_advantage' not in globals():
    def calculate_starting_position_advantage(quali_data):
        """Calculate advantage of starting position"""
        starting_position_scores = {}
        
        for driver, data in quali_data.items():
            position = data['position']
            
            # Calculate advantage factor based on position
            if position == 1:  # Pole
                score = 1.0
            elif position == 2:  # Front row
                score = 0.8
            elif position <= 3:
                score = 0.7
            elif position <= 5:
                score = 0.6
            elif position <= 10:
                score = 0.5
            else:
                score = max(0.3, 1 - (position * 0.03))
                
            starting_position_scores[driver] = score
            
        return starting_position_scores

class F1GradientBoostModel:
    """
    F1 Race Prediction Model using Gradient Boosting
    Uses track_database for track characteristics if available
    """
    def __init__(self, model_dir="ml_models"):
        """Initialize the model"""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize model components
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.historical_data = None
        
        # Track metrics
        self.metrics = {}
        
        # Setup track database if available
        if TRACK_DB_AVAILABLE:
            # Check if track data files exist
            track_dir = 'track_analysis'
            if not os.path.exists(track_dir) or not os.listdir(track_dir):
                print("Setting up track data files for current season...")
                setup_tracks_for_season()
        
    def gather_historical_data(self, years=None, races=None, max_races=10):
        """
        Gather historical race data from FastF1
        
        Parameters:
        - years: List of years to collect data from (default: current and previous year)
        - races: List of specific races to collect (default: all available)
        - max_races: Maximum number of races to process (avoid overloading API)
        
        Returns:
        - DataFrame with historical data
        """
        print("Gathering historical race data using FastF1...")
        
        # Set default years if not provided
        if years is None:
            current_year = datetime.now().year
            years = [current_year, current_year - 1]
        
        # Initialize data storage
        all_race_data = []
        race_count = 0
        
        # Process each year
        for year in years:
            print(f"\nProcessing data from {year}...")
            
            # Get available races for this year
            try:
                schedule = fastf1.get_event_schedule(year)
                available_races = schedule['EventName'].tolist()
                
                # Filter races if specified
                if races is not None:
                    available_races = [r for r in available_races if any(name.lower() in r.lower() for name in races)]
                
                print(f"Found {len(available_races)} races for {year}")
                
                # Process each race
                for race_name in available_races:
                    if race_count >= max_races:
                        print(f"Reached maximum race limit ({max_races}). Stopping data collection.")
                        break
                    
                    try:
                        print(f"\nProcessing {race_name} {year}...")
                        
                        # Load quali and race sessions
                        quali_session = load_session_safely(year, race_name, 'Q')
                        race_session = load_session_safely(year, race_name, 'R')
                        
                        if quali_session is None or race_session is None:
                            print(f"Missing session data for {race_name}. Skipping.")
                            continue
                        
                        # Get driver-team mapping
                        driver_teams = get_driver_team_mapping(race_session)
                        
                        # Get qualifying data
                        quali_data, sector_data = get_qualifying_data(quali_session)
                        
                        # Get race results
                        race_data = get_race_data(race_session)
                        
                        # Get track data from track database (if available)
                        track_chars = None
                        if TRACK_DB_AVAILABLE:
                            track_chars = get_track_data(race_name)
                        
                        # Process each driver that has both quali and race data
                        for driver in set(quali_data.keys()) & set(race_data.keys()):
                            # Skip if race position is missing
                            if 'position' not in race_data[driver]:
                                continue
                            
                            # Create feature record
                            driver_record = {
                                'year': year,
                                'race_name': race_name,
                                'driver': driver,
                                'team': driver_teams.get(driver, 'Unknown'),
                                
                                # Target variables
                                'race_position': race_data[driver]['position'],
                                
                                # Basic features
                                'quali_position': quali_data[driver]['position'],
                                'quali_time': quali_data[driver]['lap_time'],
                            }
                            
                            # Add sector times if available
                            if driver in sector_data:
                                for sector in ['sector1', 'sector2', 'sector3']:
                                    if sector in sector_data[driver] and sector_data[driver][sector] is not None:
                                        driver_record[f'{sector}_time'] = sector_data[driver][sector]
                            
                            # Add additional race metrics if available
                            for metric in ['fastest_lap', 'median_pace']:
                                if metric in race_data[driver]:
                                    driver_record[metric] = race_data[driver][metric]
                            
                            # Add track-specific features if available
                            if track_chars:
                                driver_record['track_type'] = track_chars.get('track_type', 'unknown')
                                driver_record['overtaking_difficulty'] = track_chars.get('overtaking_difficulty', 0.6)
                                driver_record['qualifying_importance'] = track_chars.get('qualifying_importance', 0.7)
                                driver_record['tire_degradation'] = track_chars.get('tire_degradation', 0.6)
                            
                            # Add to dataset
                            all_race_data.append(driver_record)
                        
                        race_count += 1
                        print(f"Processed {race_name} {year} successfully")
                        
                    except Exception as e:
                        print(f"Error processing {race_name} {year}: {e}")
                        continue
            
            except Exception as e:
                print(f"Error accessing race schedule for {year}: {e}")
                continue
                
        # Convert to DataFrame
        if not all_race_data:
            print("No historical data collected!")
            return None
            
        df = pd.DataFrame(all_race_data)
        print(f"\nSuccessfully collected {len(df)} driver-race records across {df['race_name'].nunique()} races")
        
        # Fill missing values with median
        for col in df.columns:
            if col not in ['year', 'race_name', 'driver', 'team', 'track_type']:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
        
        # Store the data
        self.historical_data = df
        
        # Save raw data
        try:
            df.to_csv(os.path.join(self.model_dir, 'historical_data.csv'), index=False)
            print(f"Saved raw historical data to {os.path.join(self.model_dir, 'historical_data.csv')}")
        except Exception as e:
            print(f"Error saving historical data: {e}")
        
        return df
    
    def engineer_features(self, df):
        """
        Engineer features for model training
        
        Parameters:
        - df: DataFrame with raw historical data
        
        Returns:
        - DataFrame with engineered features
        """
        if df is None or len(df) == 0:
            print("No data provided for feature engineering")
            return None
        
        print("\nEngineering features...")
        
        # Create a copy to avoid modifying the original
        df_engineered = df.copy()
        
        # 1. Calculate position improvement (quali → race)
        df_engineered['position_improvement'] = df_engineered['quali_position'] - df_engineered['race_position']
        
        # 2. Calculate team performance metrics
        team_race_results = df_engineered.groupby(['year', 'race_name', 'team'])['race_position'].mean().reset_index()
        team_race_results.rename(columns={'race_position': 'team_avg_position'}, inplace=True)
        
        # Merge team performance back to driver data
        df_engineered = pd.merge(
            df_engineered,
            team_race_results,
            on=['year', 'race_name', 'team'],
            how='left'
        )
        
        # 3. Calculate driver vs team performance
        df_engineered['driver_vs_team'] = df_engineered['team_avg_position'] - df_engineered['race_position']
        
        # 4. Create start position indicators for track-specific advantages
        df_engineered['front_row_start'] = (df_engineered['quali_position'] <= 2).astype(int)
        df_engineered['top_three_start'] = (df_engineered['quali_position'] <= 3).astype(int)
        
        # 5. One-hot encode teams (if enough data)
        if len(df_engineered) > 30:  # Only if we have enough data
            team_counts = df_engineered['team'].value_counts()
            common_teams = team_counts[team_counts > 1].index.tolist()
            
            for team in common_teams:
                df_engineered[f'team_{team.replace(" ", "_")}'] = (df_engineered['team'] == team).astype(int)
        
        # 6. Add track type indicators if not already present
        if 'track_type' in df_engineered.columns:
            # Encode track type if it's not already encoded
            track_types = ['street', 'power', 'technical', 'balanced']
            for track_type in track_types:
                if f'track_{track_type}' not in df_engineered.columns:
                    df_engineered[f'track_{track_type}'] = (df_engineered['track_type'] == track_type).astype(int)
        else:
            # No track type data, create from race name
            for track_type in ['street', 'power', 'technical']:
                df_engineered[f'track_{track_type}'] = 0
                
            for idx, row in df_engineered.iterrows():
                race_lower = row['race_name'].lower()
                
                # Determine track type
                if any(name in race_lower for name in ['monaco', 'singapore', 'baku']):
                    df_engineered.at[idx, 'track_street'] = 1
                elif any(name in race_lower for name in ['monza', 'spa', 'azerbaijan']):
                    df_engineered.at[idx, 'track_power'] = 1
                elif any(name in race_lower for name in ['hungar', 'barcelona', 'catalunya']):
                    df_engineered.at[idx, 'track_technical'] = 1
        
        # Store feature names (excluding target and identifiers)
        self.feature_names = [col for col in df_engineered.columns 
                             if col not in ['year', 'race_name', 'driver', 'team', 'track_type', 'race_position']]
        
        print(f"Engineered {len(self.feature_names)} features")
        return df_engineered
    
    def prepare_training_data(self, df=None):
        """
        Prepare data for model training
        
        Parameters:
        - df: DataFrame with features (if None, uses self.historical_data)
        
        Returns:
        - X_train, X_test, y_train, y_test: Training and test data
        """
        if df is None:
            if self.historical_data is None:
                print("No historical data available. Please collect data first.")
                return None, None, None, None
            df = self.historical_data
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        if df_engineered is None:
            return None, None, None, None
        
        # Select features and target
        X = df_engineered[self.feature_names]
        y = df_engineered['race_position']
        
        # Split data with larger test set due to limited data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler for later use
        dump(self.scaler, os.path.join(self.model_dir, 'scaler.joblib'))
        
        print(f"Prepared training data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train=None, y_train=None):
        """
        Train XGBoost model optimized for limited data
        
        Parameters:
        - X_train, y_train: Training data (if None, calls prepare_training_data)
        
        Returns:
        - Trained XGBoost model
        """
        print("\nTraining XGBoost model optimized for limited data...")
        
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available. Cannot train model.")
            return None
        
        # Get training data if not provided
        if X_train is None or y_train is None:
            X_train, X_test, y_train, y_test = self.prepare_training_data()
            if X_train is None:
                return None
        
        # Set parameters optimized for limited data
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,     # Slower learning rate to avoid overfitting
            'max_depth': 3,            # Shallow trees to prevent overfitting
            'min_child_weight': 3,     # Helps prevent overfitting on small datasets
            'subsample': 0.8,          # Use 80% of data per tree
            'colsample_bytree': 0.8,   # Use 80% of features per tree
            'gamma': 0.1,              # Minimum loss reduction for split
            'reg_alpha': 0.1,          # L1 regularization
            'reg_lambda': 1.0,         # L2 regularization
            'random_state': 42
        }
        
        # Create model
        self.xgb_model = xgb.XGBRegressor(**params)
        
        # Use cross-validation with k-fold to maximize use of limited data
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X_train):
            # Split data
            cv_X_train, cv_X_val = X_train[train_idx], X_train[val_idx]
            cv_y_train, cv_y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train model
            self.xgb_model.fit(
                cv_X_train, cv_y_train,
                eval_set=[(cv_X_val, cv_y_val)],
                early_stopping_rounds=15,
                verbose=False
            )
            
            # Make predictions
            pred = self.xgb_model.predict(cv_X_val)
            
            # Calculate metrics
            mse = mean_squared_error(cv_y_val, pred)
            mae = mean_absolute_error(cv_y_val, pred)
            
            cv_scores.append({'mse': mse, 'mae': mae})
        
        # Final training with all data
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            early_stopping_rounds=15,
            verbose=False
        )
        
        # Report cross-validation results
        avg_mse = np.mean([s['mse'] for s in cv_scores])
        avg_mae = np.mean([s['mae'] for s in cv_scores])
        
        print(f"Cross-validation results:")
        print(f"  Average MSE: {avg_mse:.4f}")
        print(f"  Average MAE: {avg_mae:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'xgboost_model.joblib')
        dump(self.xgb_model, model_path)
        print(f"XGBoost model saved to {model_path}")
        
        # Save feature names
        with open(os.path.join(self.model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)
        
        return self.xgb_model
    
    def evaluate_model(self, X_test=None, y_test=None):
        """
        Evaluate trained model on test data
        
        Parameters:
        - X_test, y_test: Test data (if None, uses data from prepare_training_data)
        
        Returns:
        - Dictionary of evaluation metrics
        """
        if self.xgb_model is None:
            print("No trained model available. Please train model first.")
            return None
        
        # Get test data if not provided
        if X_test is None or y_test is None:
            _, X_test, _, y_test = self.prepare_training_data()
            if X_test is None:
                return None
        
        print("\nEvaluating model on test data...")
        
        # Make predictions
        preds = self.xgb_model.predict(X_test)
        
        # Round predictions (position must be integer)
        preds_rounded = np.clip(np.round(preds), 1, 20)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, preds_rounded)
        mae = mean_absolute_error(y_test, preds_rounded)
        r2 = r2_score(y_test, preds_rounded)
        
        # Calculate rank correlation
        corr, _ = spearmanr(y_test, preds_rounded)
        
        # Calculate position accuracy
        exact_acc = np.mean(preds_rounded == y_test)
        within_1_acc = np.mean(np.abs(preds_rounded - y_test) <= 1)
        within_2_acc = np.mean(np.abs(preds_rounded - y_test) <= 2)
        
        # Store metrics
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rank_correlation': corr,
            'exact_accuracy': exact_acc,
            'within_1_accuracy': within_1_acc,
            'within_2_accuracy': within_2_acc
        }
        
        self.metrics = metrics
        
        # Print metrics
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Rank Correlation: {corr:.4f}")
        print(f"Exact Position Accuracy: {exact_acc:.4f}")
        print(f"Within ±1 Position Accuracy: {within_1_acc:.4f}")
        print(f"Within ±2 Positions Accuracy: {within_2_acc:.4f}")
        
        # Visualize feature importance
        self.visualize_feature_importance()
        
        return metrics
    
    def visualize_feature_importance(self):
        """Visualize feature importance from trained model"""
        if self.xgb_model is None or self.feature_names is None:
            print("Model not trained or feature names not available")
            return
        
        try:
            # Get feature importance
            importance = self.xgb_model.feature_importances_
            
            # Create DataFrame
            feat_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Print feature importance
            print("\nFeature Importance (Top 10):")
            for i, (feature, imp) in enumerate(zip(feat_imp['Feature'].head(10), feat_imp['Importance'].head(10))):
                print(f"{i+1}. {feature}: {imp:.4f}")
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            
            # Use a fixed color to avoid RGBA errors
            plt.barh(feat_imp['Feature'].head(10), feat_imp['Importance'].head(10), color='skyblue')
            
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('XGBoost Feature Importance (Top 10)')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'), dpi=300)
            print(f"Feature importance visualization saved to {os.path.join(self.model_dir, 'feature_importance.png')}")
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing feature importance: {e}")
    
    def load_model(self):
        """Load saved model and components"""
        model_path = os.path.join(self.model_dir, 'xgboost_model.joblib')
        scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        feature_names_path = os.path.join(self.model_dir, 'feature_names.json')
        
        try:
            # Load model
            if os.path.exists(model_path):
                self.xgb_model = load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model file {model_path} not found")
                return False
            
            # Load scaler
            if os.path.exists(scaler_path):
                self.scaler = load(scaler_path)
            else:
                print(f"Scaler file {scaler_path} not found")
                return False
            
            # Load feature names
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
            else:
                print(f"Feature names file {feature_names_path} not found")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_race_features(self, race_name, year=None):
        """
        Extract features for race prediction
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race (default: current year)
        
        Returns:
        - DataFrame with features for each driver
        """
        if year is None:
            year = datetime.now().year
        
        print(f"\nExtracting features for {race_name} {year}...")
        
        # Load the qualifying session
        quali_session = load_session_safely(year, race_name, 'Q')
        
        if quali_session is None:
            print(f"Qualifying data not available for {race_name} {year}")
            return None
        
        # Get driver-team mapping
        driver_teams = get_driver_team_mapping(quali_session)
        
        # Get qualifying data
        quali_data, sector_data = get_qualifying_data(quali_session)
        
        # Get track characteristics from track_database if available
        track_chars = None
        if TRACK_DB_AVAILABLE:
            track_chars = get_track_data(race_name, year)
            
            if track_chars is None:
                print(f"No track data available for {race_name}. Using generic track characteristics.")
                # Create generic track characteristics if not in database
                track_chars = {
                    "track_type": "balanced",
                    "overtaking_difficulty": 0.6,
                    "tire_degradation": 0.6,
                    "qualifying_importance": 0.7,
                    "start_importance": 0.7,
                    "dirty_air_impact": 0.6
                }
        
        # Calculate additional race-specific factors
        starting_position_scores = calculate_starting_position_advantage(quali_data)
        
        # Create feature records for each driver
        driver_features = []
        
        for driver, data in quali_data.items():
            # Create base feature record
            driver_record = {
                'year': year,
                'race_name': race_name,
                'driver': driver,
                'team': driver_teams.get(driver, 'Unknown'),
                'quali_position': data['position'],
                'quali_time': data['lap_time'],
            }
            
            # Add track type indicators
            if track_chars:
                track_type = track_chars.get('track_type', 'balanced')
                driver_record['track_type'] = track_type
                
                # Add one-hot encoding for track type
                driver_record['track_street'] = 1 if track_type == 'street' else 0
                driver_record['track_power'] = 1 if track_type == 'power' else 0
                driver_record['track_technical'] = 1 if track_type == 'technical' else 0
                driver_record['track_balanced'] = 1 if track_type == 'balanced' else 0
                
                # Add track characteristics as features
                driver_record['overtaking_difficulty'] = track_chars.get('overtaking_difficulty', 0.6)
                driver_record['qualifying_importance'] = track_chars.get('qualifying_importance', 0.7)
                driver_record['tire_degradation'] = track_chars.get('tire_degradation', 0.6)
            else:
                # Default track type indicators based on race name
                race_lower = race_name.lower()
                driver_record['track_street'] = 1 if any(name in race_lower for name in ['monaco', 'singapore', 'baku']) else 0
                driver_record['track_power'] = 1 if any(name in race_lower for name in ['monza', 'spa', 'azerbaijan']) else 0
                driver_record['track_technical'] = 1 if any(name in race_lower for name in ['hungar', 'barcelona', 'catalunya']) else 0
                driver_record['track_balanced'] = 1 if not any([driver_record['track_street'], driver_record['track_power'], driver_record['track_technical']]) else 0
            
            # Add starting position features
            driver_record['front_row_start'] = 1 if data['position'] <= 2 else 0
            driver_record['top_three_start'] = 1 if data['position'] <= 3 else 0
            
            # Add sector times if available
            if driver in sector_data:
                for sector in ['sector1', 'sector2', 'sector3']:
                    if sector in sector_data[driver] and sector_data[driver][sector] is not None:
                        driver_record[f'{sector}_time'] = sector_data[driver][sector]
            
            # Add to dataset
            driver_features.append(driver_record)
        
        if not driver_features:
            print("No driver features extracted")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(driver_features)
        
        # Prepare team-based features
        # Calculate team average qualifying
        team_quali_results = df.groupby('team')['quali_position'].mean().reset_index()
        team_quali_results.rename(columns={'quali_position': 'team_avg_position'}, inplace=True)
        
        # Merge team performance to driver data
        df = pd.merge(df, team_quali_results, on='team', how='left')
        
        # Add team one-hot encoding
        for team in df['team'].unique():
            df[f'team_{team.replace(" ", "_")}'] = (df['team'] == team).astype(int)
        
        # Fill missing values
        for col in df.columns:
            if col not in ['year', 'race_name', 'driver', 'team', 'track_type']:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
        
        print(f"Extracted features for {len(df)} drivers")
        return df
    
    def predict_race(self, race_name, year=None, include_rule_based=True):
        """
        Predict race positions
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race (default: current year)
        - include_rule_based: Whether to include rule-based predictions
        
        Returns:
        - Dictionary with predictions for each driver
        """
        if year is None:
            year = datetime.now().year
        
        print(f"\nPredicting race positions for {race_name} {year}...")
        
        # Check if model is loaded
        if self.xgb_model is None:
            load_success = self.load_model()
            if not load_success:
                print("Failed to load model, falling back to rule-based prediction only")
                include_rule_based = True
                
                if 'improved_predict_race_winner' not in globals():
                    print("Rule-based prediction function not available")
                    return None
        
        # Get features for the race
        features_df = self.extract_race_features(race_name, year)
        
        if features_df is None or len(features_df) == 0:
            print("Could not extract features")
            return None
        
        # Initialize predictions dictionary
        predictions = {}
        
        # Get drivers
        drivers = features_df['driver'].tolist()
        
        # Make ML-based predictions if model is available
        ml_positions = {}
        
        if self.xgb_model is not None and self.feature_names is not None:
            try:
                # Prepare features
                X = features_df.copy()
                
                # Keep only features that are in the model
                common_features = [f for f in self.feature_names if f in X.columns]
                missing_features = set(self.feature_names) - set(common_features)
                
                if missing_features:
                    print(f"Warning: Missing {len(missing_features)} features from training data")
                    # Fill missing features with default values
                    for feature in missing_features:
                        if 'team_' in feature:
                            X[feature] = 0  # Team one-hot features
                        else:
                            X[feature] = 0  # Other missing features
                
                # Make sure we have all features needed for prediction
                for feature in self.feature_names:
                    if feature not in X.columns:
                        X[feature] = 0
                
                # Reorder columns to match training data
                X = X[self.feature_names]
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Make predictions
                ml_preds = self.xgb_model.predict(X_scaled)
                
                # Round to integer positions
                ml_preds_rounded = np.clip(np.round(ml_preds), 1, len(drivers))
                
                # Create predictions dictionary
                for i, driver in enumerate(drivers):
                    ml_positions[driver] = int(ml_preds_rounded[i])
                
                print("ML-based predictions:")
                for driver, pos in sorted(ml_positions.items(), key=lambda x: x[1]):
                    print(f"  {driver}: {pos}")
            
            except Exception as e:
                print(f"Error making ML predictions: {e}")
                ml_positions = {}
        
        # Get rule-based predictions if requested
        rule_positions = {}
        
        if include_rule_based and 'improved_predict_race_winner' in globals():
            try:
                # Use track_database for optimized weights if available
                if TRACK_DB_AVAILABLE:
                    # Get optimized weights for this track
                    optimized_weights = get_optimized_track_weights(race_name, year)
                    
                    # Make prediction with optimized weights
                    winner, scores = improved_predict_race_winner(race_name, optimized_weights)
                else:
                    # Use default weights
                    winner, scores = improved_predict_race_winner(race_name)
                
                # Convert scores to positions
                sorted_drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                for i, (driver, _) in enumerate(sorted_drivers):
                    rule_positions[driver] = i + 1
                
                print("Rule-based predictions:")
                for driver, pos in sorted(rule_positions.items(), key=lambda x: x[1]):
                    print(f"  {driver}: {pos}")
            
            except Exception as e:
                print(f"Error making rule-based predictions: {e}")
                rule_positions = {}
        
        # Combine predictions if both are available
        final_positions = {}
        
        if ml_positions and rule_positions:
            print("\nCombining predictions...")
            
            # Find common drivers
            common_drivers = set(ml_positions.keys()) & set(rule_positions.keys())
            
            # For common drivers, use weighted average
            for driver in common_drivers:
                # Weight ML predictions higher if we have good metrics
                if hasattr(self, 'metrics') and self.metrics.get('rank_correlation', 0) > 0.7:
                    ml_weight = 0.7
                    rule_weight = 0.3
                else:
                    # Otherwise give rule-based predictions more weight
                    ml_weight = 0.3
                    rule_weight = 0.7
                
                # Calculate weighted position
                weighted_pos = (ml_positions[driver] * ml_weight + 
                               rule_positions[driver] * rule_weight)
                
                final_positions[driver] = weighted_pos
            
            # Add drivers only in one set
            for driver in set(ml_positions.keys()) - common_drivers:
                final_positions[driver] = ml_positions[driver]
                
            for driver in set(rule_positions.keys()) - common_drivers:
                final_positions[driver] = rule_positions[driver]
                
        elif ml_positions:
            final_positions = ml_positions
        elif rule_positions:
            final_positions = rule_positions
        else:
            print("No predictions available")
            return None
        
        # Sort drivers by predicted position
        sorted_positions = sorted(final_positions.items(), key=lambda x: x[1])
        
        # Create final predictions with integer positions
        final_predictions = {}
        for i, (driver, _) in enumerate(sorted_positions):
            final_predictions[driver] = i + 1
        
        print("\nFinal predictions:")
        for driver, pos in sorted(final_predictions.items(), key=lambda x: x[1]):
            print(f"{pos}. {driver}")
        
        # Save predictions
        try:
            predictions_file = os.path.join(self.model_dir, f'{year}_{race_name.replace(" ", "_")}_predictions.json')
            with open(predictions_file, 'w') as f:
                json.dump({
                    'race': race_name,
                    'year': year,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'predictions': final_predictions,
                    'ml_predictions': ml_positions,
                    'rule_predictions': rule_positions
                }, f, indent=2)
            print(f"Saved predictions to {predictions_file}")
        except Exception as e:
            print(f"Error saving predictions: {e}")
        
        return final_predictions
    
    def visualize_predictions(self, race_name, year=None, predictions=None):
        """
        Create visualization of race predictions
        
        Parameters:
        - race_name: Name of the race
        - year: Year of the race (default: current year)
        - predictions: Predictions dictionary (if None, loads from file)
        """
        if year is None:
            year = datetime.now().year
        
        # Load predictions if not provided
        if predictions is None:
            predictions_file = os.path.join(self.model_dir, f'{year}_{race_name.replace(" ", "_")}_predictions.json')
            
            if not os.path.exists(predictions_file):
                print(f"Predictions file {predictions_file} not found")
                return
            
            try:
                with open(predictions_file, 'r') as f:
                    data = json.load(f)
                predictions = data['predictions']
            except Exception as e:
                print(f"Error loading predictions: {e}")
                return
        
        try:
            # Sort drivers by position
            sorted_drivers = sorted(predictions.items(), key=lambda x: x[1])
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Get top drivers (max 15 for readability)
            top_n = min(15, len(sorted_drivers))
            top_drivers = [d[0] for d in sorted_drivers[:top_n]]
            positions = [d[1] for d in sorted_drivers[:top_n]]
            
            # Create bar chart with a fixed color to avoid RGBA errors
            bars = plt.bar(top_drivers, positions, color='skyblue')
            
            # Customize chart
            plt.ylabel('Predicted Position')
            plt.xlabel('Driver')
            plt.title(f'F1 Race Prediction: {race_name} {year}')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Invert y-axis (lower position is better)
            plt.gca().invert_yaxis()
            
            # Add position values
            for i, (bar, pos) in enumerate(zip(bars, positions)):
                plt.text(
                    i, 
                    pos + 0.1, 
                    str(pos), 
                    ha='center', 
                    va='bottom',
                    color='black',
                    fontweight='bold'
                )
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.model_dir, f'{year}_{race_name.replace(" ", "_")}_visualization.png')
            plt.savefig(output_file, dpi=300)
            print(f"Saved prediction visualization to {output_file}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    """Main function for F1 gradient boosting prediction system"""
    print("=" * 70)
    print("F1 Race Prediction with Gradient Boosting and Track Database")
    print("=" * 70)
    
    # Setup FastF1 cache
    cache_dir = 'f1_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if FASTF1_AVAILABLE:
        fastf1.Cache.enable_cache(cache_dir)
    
    # Create model instance
    model = F1GradientBoostModel(model_dir="ml_models")
    
    # Command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='F1 Race Predictor with Gradient Boosting')
    parser.add_argument('--train', action='store_true', help='Train model with historical data')
    parser.add_argument('--predict', type=str, help='Race name to predict')
    parser.add_argument('--year', type=int, help='Year for prediction (default: current year)')
    parser.add_argument('--max-races', type=int, default=10, help='Maximum number of races to collect for training')
    parser.add_argument('--hybrid', action='store_true', help='Use hybrid prediction (ML + rule-based)')
    parser.add_argument('--setup-tracks', action='store_true', help='Set up track data files for current season')
    
    args = parser.parse_args()
    
    # Setup track database if requested
    if args.setup_tracks and TRACK_DB_AVAILABLE:
        print("Setting up track data files...")
        count = setup_tracks_for_season()
        print(f"Created {count} track data files")
    
    # Use command line arguments if provided, otherwise interactive mode
    if args.train or args.predict or args.setup_tracks:
        # Training mode
        if args.train:
            # Collect data
            historical_data = model.gather_historical_data(max_races=args.max_races)
            
            if historical_data is not None and len(historical_data) > 0:
                # Train model
                model.train_model()
                
                # Evaluate model
                model.evaluate_model()
        
        # Prediction mode
        if args.predict:
            race_name = args.predict
            year = args.year if args.year else datetime.now().year
            
            # Make prediction
            predictions = model.predict_race(race_name, year, include_rule_based=args.hybrid)
            
            if predictions:
                # Visualize prediction
                model.visualize_predictions(race_name, year, predictions)
    else:
        # Interactive mode
        print("\nOptions:")
        print("1. Train model with historical data")
        print("2. Predict race")
        print("3. Setup track database for current season")
        print("4. Exit")
        
        choice = input("\nEnter option (1-4): ")
        
        if choice == '1':
            # Ask for maximum races to collect
            max_races = int(input("Maximum number of races to collect (recommended: 8-15): ") or "10")
            
            # Collect data
            historical_data = model.gather_historical_data(max_races=max_races)
            
            if historical_data is not None and len(historical_data) > 0:
                # Train model
                model.train_model()
                
                # Evaluate model
                model.evaluate_model()
        
        elif choice == '2':
            # Load model if not already loaded
            if model.xgb_model is None:
                model.load_model()
            
            # Ask for race details
            race_name = input("Enter race name (e.g., 'Monaco Grand Prix'): ")
            year_input = input("Enter year (default: current year): ")
            year = int(year_input) if year_input else datetime.now().year
            
            # Ask for prediction method
            hybrid = input("Use hybrid prediction (ML + rule-based)? (y/n, default: y): ").lower() != 'n'
            
            # Make prediction
            predictions = model.predict_race(race_name, year, include_rule_based=hybrid)
            
            if predictions:
                # Visualize prediction
                model.visualize_predictions(race_name, year, predictions)
        
        elif choice == '3':
            if TRACK_DB_AVAILABLE:
                print("Setting up track data files...")
                count = setup_tracks_for_season()
                print(f"Created {count} track data files")
            else:
                print("Track database module not available")
        
        elif choice == '4':
            print("Exiting...")
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()