"""
F1 Enhanced Prediction System
Combines multiple prediction models with weather data and track-specific optimizations
for improved accuracy, especially for podium positions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import fastf1

# Import components
try:
    # Base prediction model
    from f1predict import (
        improved_predict_race_winner, 
        load_session_safely, 
        get_driver_team_mapping,
        get_qualifying_data,
        get_race_data
    )
    
    # Track database
    from track_database import (
        get_track_data,
        get_optimized_track_weights
    )
    
    # Specialized podium model
    from podium_model import (
        PodiumPredictionModel,
        predict_race_with_podium_focus
    )
    
    # Weather integration
    from weather_integration import (
        get_weather_forecast,
        apply_weather_adjustments,
        build_driver_weather_performance
    )
    
    # Ensemble prediction
    from ensemble_prediction import (
        ensemble_predict_race,
        predict_race_positions,
        evaluate_ensemble_accuracy
    )
    
    # Check which components are available
    COMPONENTS = {
        'base_prediction': True,
        'track_database': True,
        'podium_model': True,
        'weather_integration': True,
        'ensemble_prediction': True
    }
except ImportError as e:
    print(f"Warning: Some components not available - {e}")
    # Set available components
    COMPONENTS = {
        'base_prediction': 'f1predict' in globals(),
        'track_database': 'track_database' in globals(),
        'podium_model': 'podium_model' in globals(),
        'weather_integration': 'weather_integration' in globals(),
        'ensemble_prediction': 'ensemble_prediction' in globals()
    }

def initialize_system(build_databases=False):
    """
    Initialize the enhanced prediction system
    
    Parameters:
    - build_databases: Whether to build/rebuild data stores (slow)
    
    Returns:
    - status: True if initialization successful
    """
    print("Initializing F1 Enhanced Prediction System...")
    
    # Create cache directory for FastF1
    cache_dir = 'f1_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")
    
    if 'fastf1' in globals():
        fastf1.Cache.enable_cache(cache_dir)
    
    # Initialize podium model
    if COMPONENTS['podium_model']:
        print("\nInitializing podium prediction model...")
        podium_model = PodiumPredictionModel()
        
        # Load existing model or build new one
        if not podium_model.load_model() and build_databases:
            print("Building driver databases and training podium model...")
            # Load driver track affinity data
            podium_model.load_driver_track_affinity()
            # Load driver form data
            podium_model.load_driver_form()
            # Load weather performance data
            podium_model.load_weather_performance()
            
            # Build any missing databases
            if not podium_model.driver_track_affinity and build_databases:
                podium_model.build_driver_track_affinity()
            if not podium_model.driver_form and build_databases:
                podium_model.build_driver_form()
            if not podium_model.weather_performance and build_databases:
                podium_model.build_weather_performance()
            
            # Train the model
            if build_databases:
                podium_model.train_podium_model()
    
    # Initialize weather data
    if COMPONENTS['weather_integration'] and build_databases:
        print("\nInitializing weather integration...")
        build_driver_weather_performance()
    
    print("\nEnhanced Prediction System initialized!")
    print("Available components:")
    for component, available in COMPONENTS.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {component}: {status}")
    
    return True

def enhanced_predict_race(race_name, year=None, focus_podium=True, include_weather=True):
    """
    Make enhanced race predictions with special focus on podium accuracy
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    - focus_podium: Whether to prioritize podium accuracy
    - include_weather: Whether to include weather adjustments
    
    Returns:
    - winner: Predicted race winner
    - podium: Predicted podium finishers (list of 3 drivers)
    - positions: Dictionary with predicted positions for all drivers
    - visualization: Base64-encoded visualization (if created)
    """
    if year is None:
        year = datetime.now().year
    
    print(f"\nEnhanced prediction for {race_name} {year}")
    
    # Step 1: Get baseline prediction with track-optimized weights
    print("\nStep 1: Getting baseline prediction with track-optimized weights...")
    try:
        if COMPONENTS['track_database']:
            optimized_weights = get_optimized_track_weights(race_name, year)
            winner, baseline_predictions = improved_predict_race_winner(race_name, optimized_weights)
        else:
            winner, baseline_predictions = improved_predict_race_winner(race_name)
    except Exception as e:
        print(f"Error in baseline prediction: {e}")
        return None, None, None, None
    
    # Step 2: Add specialized podium prediction if available
    podium_predictions = {}
    if COMPONENTS['podium_model'] and focus_podium:
        print("\nStep 2: Adding specialized podium prediction...")
        try:
            podium_model = PodiumPredictionModel()
            if podium_model.load_model():
                # Get podium predictions
                podium_drivers, podium_probabilities = podium_model.predict_podium(race_name, year)
                
                if podium_drivers:
                    print(f"Predicted podium from specialized model: {podium_drivers}")
                    podium_predictions = podium_probabilities
            else:
                print("Podium model not available, using fallback method...")
                # Use fallback method
                podium_drivers, podium_predictions = podium_model.predict_podium_fallback(race_name, year)
                
                if podium_drivers:
                    print(f"Predicted podium from fallback method: {podium_drivers}")
        except Exception as e:
            print(f"Error in podium prediction: {e}")
            print("Continuing with baseline only")
    
    # Step 3: Apply weather adjustments if available
    weather_adjusted_predictions = baseline_predictions.copy()
    if COMPONENTS['weather_integration'] and include_weather:
        print("\nStep 3: Applying weather adjustments...")
        try:
            weather_adjusted_predictions = apply_weather_adjustments(baseline_predictions, race_name, year)
        except Exception as e:
            print(f"Error applying weather adjustments: {e}")
            print("Continuing without weather adjustments")
    
    # Step 4: Create ensemble prediction
    ensemble_predictions = {}
    visualization = None
    
    if COMPONENTS['ensemble_prediction'] and podium_predictions:
        print("\nStep 4: Creating ensemble prediction...")
        try:
            # Use complete ensemble prediction function
            winner, ensemble_predictions, visualization = ensemble_predict_race(race_name, year, True)
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
            print("Creating manual ensemble...")
            
            # Manual ensemble
            ensemble_predictions = {}
            
            # Get all drivers
            all_drivers = set(weather_adjusted_predictions.keys()) | set(podium_predictions.keys())
            
            # Create ensemble prediction
            for driver in all_drivers:
                baseline_score = weather_adjusted_predictions.get(driver, 0)
                podium_prob = podium_predictions.get(driver, 0)
                
                # Calculate ensemble score with weights
                if baseline_score > 0.7:  # Likely top position
                    ensemble_score = (baseline_score * 0.4) + (podium_prob * 0.6)
                else:
                    ensemble_score = (baseline_score * 0.7) + (podium_prob * 0.3)
                
                ensemble_predictions[driver] = ensemble_score
            
            # Get predicted winner from ensemble
            winner = max(ensemble_predictions.items(), key=lambda x: x[1])[0]
    else:
        # Use weather-adjusted baseline if no ensemble
        ensemble_predictions = weather_adjusted_predictions
        
        # Get predicted winner from weather-adjusted baseline
        winner = max(ensemble_predictions.items(), key=lambda x: x[1])[0]
    
    # Convert scores to positions
    positions = {}
    for i, (driver, _) in enumerate(sorted(ensemble_predictions.items(), key=lambda x: x[1], reverse=True)):
        positions[driver] = i + 1
    
    # Get predicted podium
    podium = [driver for driver, pos in positions.items() if pos <= 3]
    
    print("\nEnhanced Prediction - Final Results:")
    print(f"Predicted Winner: {winner}")
    print(f"Predicted Podium: {podium}")
    
    return winner, podium, positions, visualization

def evaluate_prediction_accuracy(race_name, year, method='enhanced'):
    """
    Evaluate prediction accuracy for a past race
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race
    - method: Prediction method ('enhanced', 'baseline', or 'all')
    
    Returns:
    - accuracy_data: Dictionary of accuracy metrics
    """
    print(f"\nEvaluating prediction accuracy for {race_name} {year}...")
    
    # Get actual race results
    race_session = load_session_safely(year, race_name, 'R')
    
    if race_session is None:
        print(f"Race data not available for {race_name} {year}")
        return None
    
    race_data = get_race_data(race_session)
    
    # Convert to positions
    actual_positions = {}
    for driver, data in race_data.items():
        if 'position' in data:
            actual_positions[driver] = data['position']
    
    # Find actual podium
    actual_podium = [driver for driver, pos in sorted(actual_positions.items(), key=lambda x: x[1]) if pos <= 3]
    
    print(f"Actual Podium: {actual_podium}")
    
    # Get predictions using different methods
    if method == 'all' or method == 'baseline':
        # Get baseline prediction
        try:
            if COMPONENTS['track_database']:
                optimized_weights = get_optimized_track_weights(race_name, year)
                winner, baseline_predictions = improved_predict_race_winner(race_name, optimized_weights)
            else:
                winner, baseline_predictions = improved_predict_race_winner(race_name)
            
            # Convert to positions
            baseline_positions = {}
            for i, (driver, _) in enumerate(sorted(baseline_predictions.items(), key=lambda x: x[1], reverse=True)):
                baseline_positions[driver] = i + 1
            
            # Get baseline podium
            baseline_podium = [driver for driver, pos in baseline_positions.items() if pos <= 3]
            
            print(f"Baseline Podium: {baseline_podium}")
            
            # Calculate baseline accuracy
            baseline_correct = sum(1 for driver in baseline_podium if driver in actual_podium)
            baseline_accuracy = baseline_correct / 3
            
            print(f"Baseline Podium Accuracy: {baseline_accuracy * 100:.1f}% ({baseline_correct}/3 correct)")
        except Exception as e:
            print(f"Error in baseline prediction: {e}")
    
    if method == 'all' or method == 'enhanced':
        # Get enhanced prediction
        try:
            winner, podium, positions, _ = enhanced_predict_race(race_name, year)
            
            if podium and positions:
                # Calculate enhanced accuracy
                enhanced_correct = sum(1 for driver in podium if driver in actual_podium)
                enhanced_accuracy = enhanced_correct / 3
                
                print(f"Enhanced Podium Accuracy: {enhanced_accuracy * 100:.1f}% ({enhanced_correct}/3 correct)")
                
                # Use ensemble evaluation for detailed metrics
                if COMPONENTS['ensemble_prediction']:
                    accuracy_data = evaluate_ensemble_accuracy(race_name, year, True)
                    return accuracy_data
                
                # Simple metrics calculation
                accuracy_data = {
                    'podium_accuracy': enhanced_accuracy,
                    'podium_correct': enhanced_correct
                }
                
                return accuracy_data
        except Exception as e:
            print(f"Error in enhanced prediction: {e}")
    
    return None

def create_visualization(race_name, year, positions, save_file=True):
    """
    Create visualization of race prediction
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race
    - positions: Dictionary with predicted positions
    - save_file: Whether to save visualization to file
    
    Returns:
    - base64_image: Base64-encoded visualization
    """
    try:
        # Generate visualization
        plt.figure(figsize=(12, 6))
        
        # Get top drivers (maximum 10 for visibility)
        sorted_positions = sorted(positions.items(), key=lambda x: x[1])
        top_n = min(10, len(sorted_positions))
        top_drivers = [d[0] for d in sorted_positions[:top_n]]
        pos_values = [positions[d] for d in top_drivers]
        
        # Create bar chart
        bars = plt.bar(top_drivers, pos_values, color='steelblue', alpha=0.7)
        
        # Highlight podium positions
        for i, pos in enumerate(pos_values):
            if pos <= 3:
                bars[i].set_color(['gold', 'silver', '#CD7F32'][pos-1])
                bars[i].set_alpha(0.9)
        
        plt.xlabel('Driver')
        plt.ylabel('Predicted Position')
        plt.title(f'F1 Race Prediction: {race_name} {year}')
        plt.xticks(rotation=45)
        
        # Invert y-axis (lower position is better)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        # Save visualization if requested
        if save_file:
            output_file = f"prediction_{race_name.replace(' ', '_')}_{year}.png"
            plt.savefig(output_file, dpi=300)
            print(f"Saved visualization to {output_file}")
        
        # Convert to base64 for web display
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plt.close()
        
        return base64_image
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Enhanced Prediction System')
    parser.add_argument('--race', type=str, help='Race name')
    parser.add_argument('--year', type=int, default=datetime.now().year, help='Year')
    parser.add_argument('--initialize', action='store_true', help='Initialize system and build databases')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate prediction accuracy')
    parser.add_argument('--method', type=str, default='enhanced', help='Prediction method (enhanced, baseline, or all)')
    
    args = parser.parse_args()
    
    # Initialize system if requested
    if args.initialize:
        initialize_system(build_databases=True)
    
    # Make prediction if race specified
    if args.race:
        if args.evaluate:
            evaluate_prediction_accuracy(args.race, args.year, args.method)
        else:
            winner, podium, positions, _ = enhanced_predict_race(args.race, args.year)
            
            if positions:
                # Create visualization
                create_visualization(args.race, args.year, positions)
    else:
        # Just initialize without building databases
        initialize_system(build_databases=False)