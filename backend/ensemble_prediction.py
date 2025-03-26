"""
Ensemble F1 Race Prediction Module
Combines multiple models for optimal prediction accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

try:
    # Import from existing modules
    from f1predict import improved_predict_race_winner, load_session_safely, get_driver_team_mapping
    from track_database import get_track_data, get_optimized_track_weights
    from podium_model import PodiumPredictionModel
    
    # Check if podium model is available
    PODIUM_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available - {e}")
    # Check specifically which modules are missing
    PODIUM_MODEL_AVAILABLE = False

def ensemble_predict_race(race_name, year=None, podium_focus=True):
    """
    Make race predictions using an ensemble of models
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    - podium_focus: Whether to prioritize podium accuracy
    
    Returns:
    - predicted_winner: Driver code of predicted winner
    - positions: Dictionary with predicted positions for all drivers
    - visualization: Base64-encoded visualization (if created)
    """
    if year is None:
        year = datetime.now().year
    
    print(f"\nEnsemble prediction for {race_name} {year}")
    
    # Component 1: Base prediction model
    print("\nComponent 1: Getting baseline prediction...")
    try:
        # Use track-optimized weights if available
        optimized_weights = get_optimized_track_weights(race_name, year)
        winner, baseline_predictions = improved_predict_race_winner(race_name, optimized_weights)
    except Exception as e:
        print(f"Error in baseline prediction: {e}")
        print("Using default weights")
        winner, baseline_predictions = improved_predict_race_winner(race_name)
    
    # Convert to positions
    baseline_positions = {}
    for i, (driver, _) in enumerate(sorted(baseline_predictions.items(), key=lambda x: x[1], reverse=True)):
        baseline_positions[driver] = i + 1
    
    # Component 2: Podium-focused model (if available)
    podium_probabilities = {}
    if PODIUM_MODEL_AVAILABLE and podium_focus:
        print("\nComponent 2: Getting specialized podium prediction...")
        try:
            podium_model = PodiumPredictionModel()
            
            # Check if model exists or needs training
            if not podium_model.load_model():
                print("Podium model not found. Using fallback methods.")
            
            # Get podium predictions
            podium_drivers, podium_probabilities = podium_model.predict_podium(race_name, year)
            
            if not podium_drivers:
                print("Could not get podium predictions. Using baseline only.")
        except Exception as e:
            print(f"Error in podium prediction: {e}")
            print("Using baseline prediction only")
    
    # Create ensemble prediction
    ensemble_predictions = {}
    
    # Get all drivers
    all_drivers = set(baseline_predictions.keys()) | set(podium_probabilities.keys())
    
    if podium_probabilities and podium_focus:
        print("\nCreating ensemble prediction with podium focus...")
        for driver in all_drivers:
            # Get baseline score and position
            baseline_score = baseline_predictions.get(driver, 0)
            baseline_position = baseline_positions.get(driver, 20)  # Default to last if not found
            
            # Get podium probability
            podium_prob = podium_probabilities.get(driver, 0)
            
            # Calculate ensemble score:
            # - For likely podium positions, give more weight to podium model
            # - For other positions, rely more on baseline model
            if baseline_position <= 5:  # Top positions
                # Higher weight to podium model for potential podium finishers
                ensemble_score = (baseline_score * 0.4) + (podium_prob * 0.6 * 2)  # Double podium influence
            else:
                # Higher weight to baseline model for lower positions
                ensemble_score = (baseline_score * 0.8) + (podium_prob * 0.2)
            
            ensemble_predictions[driver] = ensemble_score
    else:
        # No podium model, use baseline prediction
        ensemble_predictions = baseline_predictions.copy()
    
    # Convert scores to positions
    final_positions = {}
    for i, (driver, _) in enumerate(sorted(ensemble_predictions.items(), key=lambda x: x[1], reverse=True)):
        final_positions[driver] = i + 1
    
    # Determine predicted winner
    predicted_winner = next(iter(sorted(ensemble_predictions.items(), key=lambda x: x[1], reverse=True)))[0]
    
    # Create visualization
    visualization = None
    
    try:
        # Generate visualization
        plt.figure(figsize=(12, 6))
        
        # Get top drivers (maximum 10 for visibility)
        sorted_results = sorted(ensemble_predictions.items(), key=lambda x: x[1], reverse=True)
        top_n = min(10, len(sorted_results))
        top_drivers = [d[0] for d in sorted_results[:top_n]]
        scores = [ensemble_predictions[d] for d in top_drivers]
        positions = [final_positions[d] for d in top_drivers]
        
        # Create bar chart
        bars = plt.bar(top_drivers, scores, color='steelblue', alpha=0.7)
        
        # Highlight podium positions
        for i, pos in enumerate(positions):
            if pos <= 3:
                bars[i].set_color(['gold', 'silver', '#CD7F32'][pos-1])
                bars[i].set_alpha(0.9)
        
        plt.xlabel('Driver')
        plt.ylabel('Prediction Score')
        plt.title(f'F1 Race Prediction (Ensemble): {race_name} {year}')
        plt.xticks(rotation=45)
        
        # Add position labels
        for i, (score, pos) in enumerate(zip(scores, positions)):
            plt.text(i, score + 0.02, f"P{pos}", ha='center')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = f"ensemble_prediction_{race_name.replace(' ', '_')}_{year}.png"
        plt.savefig(output_file, dpi=300)
        print(f"Saved visualization to {output_file}")
        
        # Convert to base64 for web display
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        visualization = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plt.close()
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    print("\nEnsemble Prediction - Top 5:")
    for i, (driver, _) in enumerate(sorted(ensemble_predictions.items(), key=lambda x: x[1], reverse=True)[:5]):
        print(f"{i+1}. {driver} (Score: {ensemble_predictions[driver]:.3f})")
    
    return predicted_winner, ensemble_predictions, visualization

def predict_race_positions(race_name, year=None, podium_focus=True):
    """
    Predict race positions using the ensemble model
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race (default: current year)
    - podium_focus: Whether to prioritize podium accuracy
    
    Returns:
    - positions: List of dictionaries with driver and predicted position
    - podium: List of drivers predicted to finish on podium
    """
    if year is None:
        year = datetime.now().year
    
    # Get predictions
    predicted_winner, prediction_scores, _ = ensemble_predict_race(race_name, year, podium_focus)
    
    # Convert to positions
    positions = []
    for i, (driver, score) in enumerate(sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)):
        positions.append({
            'position': i + 1,
            'driver': driver,
            'score': score
        })
    
    # Get predicted podium
    podium = [p['driver'] for p in positions if p['position'] <= 3]
    
    return positions, podium

def evaluate_ensemble_accuracy(race_name, year, save_results=True):
    """
    Evaluate ensemble prediction accuracy for a past race
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race
    - save_results: Whether to save evaluation results
    
    Returns:
    - accuracy_metrics: Dictionary of accuracy metrics
    """
    print(f"\nEvaluating ensemble prediction accuracy for {race_name} {year}")
    
    # Get race session
    race_session = load_session_safely(year, race_name, 'R')
    
    if race_session is None:
        print(f"Race data not available for {race_name} {year}")
        return None
    
    # Get actual results
    from f1predict import get_race_data
    actual_results = get_race_data(race_session)
    
    # Convert to positions
    actual_positions = {}
    for driver, data in actual_results.items():
        if 'position' in data:
            actual_positions[driver] = data['position']
    
    # Get prediction with ensemble model
    _, prediction_scores, _ = ensemble_predict_race(race_name, year)
    
    # Convert to positions
    predicted_positions = {}
    for i, (driver, _) in enumerate(sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)):
        predicted_positions[driver] = i + 1
    
    # Calculate accuracy metrics
    metrics = {}
    
    # Get common drivers
    common_drivers = set(actual_positions.keys()) & set(predicted_positions.keys())
    
    if not common_drivers:
        print("No common drivers between prediction and actual results")
        return None
    
    # 1. Winner prediction
    actual_winner = next((driver for driver, pos in actual_positions.items() if pos == 1), None)
    predicted_winner = next((driver for driver, pos in predicted_positions.items() if pos == 1), None)
    
    metrics['winner_correct'] = actual_winner == predicted_winner
    
    # 2. Podium accuracy
    actual_podium = [driver for driver, pos in actual_positions.items() if pos <= 3]
    predicted_podium = [driver for driver, pos in predicted_positions.items() if pos <= 3]
    
    podium_correct = sum(1 for driver in predicted_podium if driver in actual_podium)
    metrics['podium_accuracy'] = podium_correct / 3
    
    # 3. Position accuracy
    total_error = 0
    exact_matches = 0
    
    for driver in common_drivers:
        pred_pos = predicted_positions[driver]
        actual_pos = actual_positions[driver]
        
        # Position error
        error = abs(pred_pos - actual_pos)
        total_error += error
        
        # Exact matches
        if error == 0:
            exact_matches += 1
    
    metrics['avg_position_error'] = total_error / len(common_drivers)
    metrics['exact_matches'] = exact_matches
    metrics['exact_match_rate'] = exact_matches / len(common_drivers)
    
    # 4. Rank correlation
    from scipy.stats import spearmanr
    
    pred_ranks = [predicted_positions[driver] for driver in common_drivers]
    actual_ranks = [actual_positions[driver] for driver in common_drivers]
    
    rank_corr, p_value = spearmanr(pred_ranks, actual_ranks)
    metrics['rank_correlation'] = rank_corr
    metrics['correlation_p_value'] = p_value
    
    # Print results
    print("\nEnsemble Prediction Accuracy:")
    print(f"Winner prediction: {'Correct' if metrics['winner_correct'] else 'Incorrect'}")
    print(f"Podium accuracy: {metrics['podium_accuracy'] * 100:.1f}% ({podium_correct}/3 correct)")
    print(f"Average position error: {metrics['avg_position_error']:.2f} positions")
    print(f"Exact position matches: {metrics['exact_matches']} ({metrics['exact_match_rate'] * 100:.1f}%)")
    print(f"Rank correlation: {metrics['rank_correlation']:.3f} (p={metrics['correlation_p_value']:.3f})")
    
    # Create comparison visualization
    try:
        create_comparison_visualization(
            race_name, year, 
            predicted_positions, actual_positions,
            metrics
        )
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
    
    # Save results
    if save_results:
        try:
            # Create output directory
            output_dir = 'ensemble_evaluation'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save metrics
            metrics_file = os.path.join(output_dir, f"{year}_{race_name.replace(' ', '_')}_metrics.json")
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'race_name': race_name,
                    'year': year,
                    'metrics': metrics,
                    'predicted_positions': predicted_positions,
                    'actual_positions': actual_positions
                }, f, indent=2)
            
            print(f"Saved evaluation results to {metrics_file}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
    
    return metrics

def create_comparison_visualization(race_name, year, predicted_positions, actual_positions, metrics=None):
    """
    Create visualization comparing predicted vs actual positions
    
    Parameters:
    - race_name: Name of the race
    - year: Year of the race
    - predicted_positions: Dictionary of predicted positions
    - actual_positions: Dictionary of actual positions
    - metrics: Accuracy metrics (optional)
    
    Returns:
    - None
    """
    # Get common drivers
    common_drivers = set(predicted_positions.keys()) & set(actual_positions.keys())
    
    if not common_drivers:
        print("No common drivers to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get top 10 drivers by actual position
    top_drivers = [d for d, p in sorted(actual_positions.items(), key=lambda x: x[1]) if d in common_drivers][:10]
    
    # Prepare data for visualization
    drivers = []
    pred_pos = []
    act_pos = []
    
    for driver in top_drivers:
        drivers.append(driver)
        pred_pos.append(predicted_positions[driver])
        act_pos.append(actual_positions[driver])
    
    # Create bar chart
    x = np.arange(len(drivers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pred_bars = ax.bar(x - width/2, pred_pos, width, label='Predicted', color='steelblue', alpha=0.7)
    act_bars = ax.bar(x + width/2, act_pos, width, label='Actual', color='green', alpha=0.7)
    
    # Customize chart
    ax.set_xlabel('Driver')
    ax.set_ylabel('Position')
    ax.set_title(f'Prediction vs Actual Results: {race_name} {year}')
    ax.set_xticks(x)
    ax.set_xticklabels(drivers)
    ax.legend()
    
    # Invert y-axis (lower position is better)
    ax.invert_yaxis()
    
    # Add position values
    for i, v in enumerate(pred_pos):
        ax.text(i - width/2, v + 0.1, str(v), ha='center')
    
    for i, v in enumerate(act_pos):
        ax.text(i + width/2, v + 0.1, str(v), ha='center')
    
    # Add metrics if provided
    if metrics:
        metrics_text = (
            f"Winner prediction: {'✓' if metrics['winner_correct'] else '✗'}\n"
            f"Podium accuracy: {metrics['podium_accuracy'] * 100:.1f}%\n"
            f"Avg position error: {metrics['avg_position_error']:.2f}"
        )
        
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = 'ensemble_evaluation'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"{year}_{race_name.replace(' ', '_')}_comparison.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved comparison visualization to {output_file}")
    
    plt.close()

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Ensemble Race Prediction')
    parser.add_argument('--race', type=str, help='Race name')
    parser.add_argument('--year', type=int, default=datetime.now().year, help='Year')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate accuracy (for past races)')
    
    args = parser.parse_args()
    
    if args.race:
        if args.evaluate:
            evaluate_ensemble_accuracy(args.race, args.year)
        else:
            positions, podium = predict_race_positions(args.race, args.year)
            
            print(f"\nPredicted Podium for {args.race} {args.year}:")
            for i, driver in enumerate(podium):
                print(f"{i+1}. {driver}")