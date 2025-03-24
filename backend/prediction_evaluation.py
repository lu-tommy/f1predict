import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import json
import os
from datetime import datetime

def analyze_prediction_accuracy(race_name, actual_race_session=None, predicted_results=None, prediction_function=None,
                              load_session_function=None, get_driver_team_mapping_function=None):
    """
    Compare predicted race results with actual results to evaluate model accuracy
    and suggest potential improvements.
    
    Parameters:
    - race_name: Name of the race
    - actual_race_session: FastF1 race session object (if already loaded)
    - predicted_results: Dictionary of predicted driver scores (if already calculated)
    - prediction_function: Function to call for predictions if needed
    - load_session_function: Function to load race session if needed
    - get_driver_team_mapping_function: Function to get driver-team mapping
    
    Returns:
    - accuracy_data: Dictionary containing accuracy metrics
    """
    print(f"\nAnalyzing prediction accuracy for {race_name}...")
    
    # Get current year
    current_year = datetime.now().year
    
    # Step 1: Get predictions if not provided
    if predicted_results is None:
        if prediction_function is not None:
            # Generate predictions for this race
            predicted_winner, predicted_results = prediction_function(race_name)
        else:
            print("Error: No predictions provided and no prediction function available")
            return None
    else:
        # Sort predictions to get the predicted winner
        sorted_predictions = sorted(predicted_results.items(), key=lambda x: x[1], reverse=True)
        predicted_winner = sorted_predictions[0][0]
    
    # Convert predictions to a ranked list
    predicted_rankings = {driver: i+1 for i, (driver, _) in 
                          enumerate(sorted(predicted_results.items(), key=lambda x: x[1], reverse=True))}
    
    # Step 2: Get actual race results if not provided
    if actual_race_session is None:
        if load_session_function is None:
            print("Error: No race session provided and no load function available")
            return None
            
        print(f"Loading race results for {race_name} {current_year}...")
        actual_race_session = load_session_function(current_year, race_name, 'R')
        
        # If race hasn't happened yet
        if actual_race_session is None:
            print(f"No race results available for {race_name} {current_year}. Race may not have occurred yet.")
            return None
    
    # Extract final race positions
    actual_results = {}
    try:
        results = actual_race_session.results
        for _, row in results.iterrows():
            if 'Abbreviation' in row and 'Position' in row and pd.notna(row['Position']):
                driver_abbr = row['Abbreviation']
                position = int(row['Position'])
                actual_results[driver_abbr] = position
        
        # Check if we have actual results
        if not actual_results:
            print(f"No race results data available for {race_name} {current_year}.")
            return None
        
        # Find actual winner
        actual_winner = None
        for driver, position in actual_results.items():
            if position == 1:
                actual_winner = driver
                break
                
        print(f"Actual winner: {actual_winner}")
        print(f"Predicted winner: {predicted_winner}")
        
        # Calculate accuracy metrics
        accuracy_data = {}
        
        # 1. Top-N accuracy
        accuracy_data['top1_correct'] = predicted_winner == actual_winner
        
        # Get drivers predicted to be in top 3
        predicted_top3 = [driver for driver, rank in predicted_rankings.items() if rank <= 3]
        actual_top3 = [driver for driver, pos in actual_results.items() if pos <= 3]
        
        top3_correct = sum(1 for driver in predicted_top3 if driver in actual_top3)
        accuracy_data['top3_accuracy'] = top3_correct / 3 if len(predicted_top3) >= 3 else 0
        
        # 2. Rank correlation
        # Get common drivers between predictions and results
        common_drivers = set(predicted_rankings.keys()) & set(actual_results.keys())
        
        if common_drivers:
            pred_ranks = [predicted_rankings[driver] for driver in common_drivers]
            actual_ranks = [actual_results[driver] for driver in common_drivers]
            
            # Calculate Spearman's rank correlation
            correlation, p_value = spearmanr(pred_ranks, actual_ranks)
            accuracy_data['rank_correlation'] = correlation
            accuracy_data['correlation_p_value'] = p_value
            
            print(f"Rank correlation: {correlation:.3f} (p={p_value:.3f})")
        else:
            accuracy_data['rank_correlation'] = None
            accuracy_data['correlation_p_value'] = None
        
        # 3. Points-based scoring
        # Award points based on position accuracy (F1 points system)
        # 25, 18, 15, 12, 10, 8, 6, 4, 2, 1 for top 10 positions
        f1_points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        
        max_points = sum(points for pos, points in f1_points.items() if pos <= min(10, len(common_drivers)))
        
        earned_points = 0
        for driver in common_drivers:
            predicted_pos = predicted_rankings[driver]
            actual_pos = actual_results[driver]
            
            # Award points if actual position is in top 10
            if actual_pos <= 10:
                if predicted_pos <= 10:  # If correctly predicted in top 10
                    # Points based on position difference
                    pos_diff = abs(predicted_pos - actual_pos)
                    if pos_diff == 0:  # Exact position match
                        points = f1_points[actual_pos]
                    elif pos_diff == 1:  # Off by one position
                        points = f1_points[actual_pos] * 0.75
                    elif pos_diff == 2:  # Off by two positions
                        points = f1_points[actual_pos] * 0.5
                    else:  # Off by more positions
                        points = f1_points[actual_pos] * 0.25
                    
                    earned_points += points
        
        accuracy_data['prediction_score'] = earned_points / max_points if max_points > 0 else 0
        print(f"Prediction score: {accuracy_data['prediction_score']:.2f} (higher is better)")
        
        # 4. Calculate error analysis by team
        team_errors = {}
        
        # Get driver-team mapping
        if get_driver_team_mapping_function is not None:
            driver_teams = get_driver_team_mapping_function(actual_race_session)
        else:
            # Fallback to empty mapping if function not provided
            driver_teams = {}
        
        for driver in common_drivers:
            team = driver_teams.get(driver)
            if team:
                if team not in team_errors:
                    team_errors[team] = []
                
                # Position difference (predicted vs actual)
                pos_diff = predicted_rankings[driver] - actual_results[driver]
                team_errors[team].append(pos_diff)
        
        # Calculate average error by team
        team_avg_errors = {}
        for team, errors in team_errors.items():
            team_avg_errors[team] = sum(errors) / len(errors)
        
        accuracy_data['team_avg_errors'] = team_avg_errors
        
        # Log significant team biases
        print("\nTeam prediction biases:")
        for team, avg_error in sorted(team_avg_errors.items(), key=lambda x: abs(x[1]), reverse=True):
            bias_type = "underestimated" if avg_error < 0 else "overestimated"
            print(f"  {team}: {abs(avg_error):.1f} positions {bias_type}")
        
        # 5. Identify specific driver outliers
        driver_errors = {}
        for driver in common_drivers:
            # Position difference (predicted vs actual)
            pos_diff = predicted_rankings[driver] - actual_results[driver]
            driver_errors[driver] = pos_diff
        
        # Sort by absolute error
        sorted_driver_errors = sorted(driver_errors.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Log significant driver mispredictions
        print("\nSignificant driver mispredictions:")
        for driver, error in sorted_driver_errors[:5]:  # Top 5 mispredictions
            bias_type = "underestimated" if error < 0 else "overestimated"
            print(f"  {driver}: {abs(error)} positions {bias_type}")
        
        accuracy_data['driver_errors'] = driver_errors
        
        # Save accuracy data for future model improvement
        save_accuracy_data(race_name, current_year, accuracy_data, predicted_rankings, actual_results, driver_teams)
        
        # Create visualization of predictions vs results
        visualize_prediction_comparison(race_name, current_year, predicted_rankings, actual_results)
        
        # Generate suggested improvements based on analysis
        suggest_model_improvements(accuracy_data, race_name)
        
        return accuracy_data
    
    except Exception as e:
        print(f"Error analyzing prediction accuracy: {e}")
        return None

def save_accuracy_data(race_name, year, accuracy_data, predicted_rankings, actual_results, driver_teams=None):
    """Save prediction accuracy data for later analysis and model improvement"""
    # Create directory if it doesn't exist
    history_dir = 'prediction_history'
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
    # Prepare data for saving
    data_to_save = {
        'race_name': race_name,
        'year': year,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'metrics': accuracy_data,
        'predicted_rankings': predicted_rankings,
        'actual_results': actual_results
    }
    
    # Add driver-team mapping if available
    if driver_teams:
        data_to_save['driver_teams'] = driver_teams
    
    # Convert any non-serializable objects
    for key in data_to_save['metrics']:
        if isinstance(data_to_save['metrics'][key], np.float64):
            data_to_save['metrics'][key] = float(data_to_save['metrics'][key])
    
    # Save to file
    filename = f"{history_dir}/{year}_{race_name.replace(' ', '_')}.json"
    try:
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Saved prediction data to {filename}")
    except Exception as e:
        print(f"Error saving prediction data: {e}")

def visualize_prediction_comparison(race_name, year, predicted_rankings, actual_results):
    """Create a visual comparison of predicted vs actual race results"""
    # Get common drivers
    common_drivers = set(predicted_rankings.keys()) & set(actual_results.keys())
    
    if not common_drivers:
        print("No common drivers between predictions and results.")
        return
    
    # Prepare data for visualization
    drivers = []
    pred_positions = []
    actual_positions = []
    
    # Get top 10 drivers by actual position
    top_drivers = sorted([(driver, pos) for driver, pos in actual_results.items() 
                         if driver in common_drivers], key=lambda x: x[1])[:10]
    
    for driver, actual_pos in top_drivers:
        drivers.append(driver)
        pred_positions.append(predicted_rankings[driver])
        actual_positions.append(actual_pos)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set position range
    max_pos = max(max(pred_positions), max(actual_positions))
    positions = range(1, max_pos + 1)
    
    # Plot
    bar_width = 0.35
    opacity = 0.8
    
    plt.barh([p + bar_width/2 for p in range(len(drivers))], pred_positions, 
             bar_width, alpha=opacity, color='b', label='Predicted')
    
    plt.barh([p - bar_width/2 for p in range(len(drivers))], actual_positions, 
             bar_width, alpha=opacity, color='g', label='Actual')
    
    # Customize plot
    plt.yticks(range(len(drivers)), drivers)
    plt.xlabel('Position')
    plt.title(f'Predicted vs Actual Results - {race_name} {year}')
    plt.legend()
    
    # Invert y-axis (better positions at the top)
    plt.gca().invert_yaxis()
    # Invert x-axis (position 1 at the left)
    plt.gca().invert_xaxis()
    
    # Add grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'prediction_comparison_{year}_{race_name.replace(" ", "_")}.png', dpi=300)
    print(f"Saved comparison visualization to prediction_comparison_{year}_{race_name.replace(' ', '_')}.png")
    
    # Show plot
    plt.close()

def suggest_model_improvements(accuracy_data, race_name):
    """Suggest potential model improvements based on accuracy analysis"""
    print("\nModel improvement suggestions:")
    
    # Analyze team biases
    team_biases = accuracy_data.get('team_avg_errors', {})
    significant_team_biases = [(team, bias) for team, bias in team_biases.items() if abs(bias) > 2]
    
    if significant_team_biases:
        print("1. Team-specific adjustments:")
        for team, bias in sorted(significant_team_biases, key=lambda x: abs(x[1]), reverse=True):
            adjustment = "increase" if bias > 0 else "decrease"
            print(f"   - {adjustment} prediction scores for {team} drivers (avg error: {bias:.1f} positions)")
    
    # Analyze correlation
    correlation = accuracy_data.get('rank_correlation')
    if correlation is not None:
        if correlation < 0.5:
            print("2. Feature weight rebalancing:")
            print("   - Consider adjusting weights based on track characteristics")
            print("   - Review starting position impact for this track type")
            print("   - Reevaluate team dynamics factor for recent races")
    
    # Analyze individual driver mispredictions
    driver_errors = accuracy_data.get('driver_errors', {})
    significant_driver_errors = [(driver, error) for driver, error in driver_errors.items() if abs(error) > 5]
    
    if significant_driver_errors:
        print("3. Driver-specific considerations:")
        for driver, error in sorted(significant_driver_errors, key=lambda x: abs(x[1]), reverse=True)[:3]:
            issue = "underestimated" if error < 0 else "overestimated"
            print(f"   - {driver} performance {issue} by {abs(error)} positions")
    
    # Analyze winner prediction
    if not accuracy_data.get('top1_correct', False):
        print("4. Winner prediction improvement:")
        print("   - Review pole position advantage for this track")
        print("   - Check team order patterns for frontrunning teams")
    
    # Track-specific suggestions
    print(f"5. {race_name} specific factors:")
    if "Monaco" in race_name or "Singapore" in race_name:
        print("   - Increase qualifying importance and decrease overtaking weight")
    elif "Monza" in race_name or "Spa" in race_name:
        print("   - Reassess straight-line speed and DRS effect")
    elif "Australia" in race_name:
        print("   - Review impact of track evolution on qualifying")
    elif "China" in race_name:
        print("   - Reassess tire degradation impact and team strategy weight")

def evaluate_multiple_races(races=None, prediction_function=None, load_session_function=None, get_driver_team_mapping_function=None):
    """Evaluate prediction accuracy across multiple races to find patterns"""
    # If no races specified, try to use prediction history
    if races is None:
        history_dir = 'prediction_history'
        if not os.path.exists(history_dir):
            print("No prediction history found. Please specify races to evaluate.")
            return
        
        # Load all saved prediction data
        history_files = [f for f in os.listdir(history_dir) if f.endswith('.json')]
        
        if not history_files:
            print("No prediction history found. Please specify races to evaluate.")
            return
        
        # Load data from each file
        all_race_data = []
        for filename in history_files:
            try:
                with open(os.path.join(history_dir, filename), 'r') as f:
                    race_data = json.load(f)
                    all_race_data.append(race_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        if not all_race_data:
            print("No valid prediction history found.")
            return
    else:
        # Analyze specified races
        all_race_data = []
        current_year = datetime.now().year
        
        for race_name in races:
            # Skip if no load function provided
            if load_session_function is None:
                print(f"Cannot load race session for {race_name} without load_session_function")
                continue
                
            # Get race results
            race_session = load_session_function(current_year, race_name, 'R')
            if race_session is None:
                print(f"No results available for {race_name}.")
                continue
            
            # Skip if no prediction function provided
            if prediction_function is None:
                print(f"Cannot generate predictions for {race_name} without prediction_function")
                continue
                
            # Get predictions
            _, predicted_results = prediction_function(race_name)
            if predicted_results is None:
                print(f"Could not generate predictions for {race_name}.")
                continue
            
            # Analyze accuracy
            accuracy_data = analyze_prediction_accuracy(
                race_name, 
                race_session, 
                predicted_results,
                prediction_function,
                load_session_function,
                get_driver_team_mapping_function
            )
            
            if accuracy_data is None:
                continue
                
            # Prepare data
            race_data = {
                'race_name': race_name,
                'year': current_year,
                'metrics': accuracy_data
            }
            all_race_data.append(race_data)
    
    # If we have data for at least 2 races, perform cross-race analysis
    if len(all_race_data) >= 2:
        print("\n==== Cross-Race Analysis ====")
        print(f"Analyzing {len(all_race_data)} races")
        
        # Calculate overall metrics - FIXED with safe access
        avg_correlation = np.mean([data.get('metrics', {}).get('rank_correlation', 0) 
                                  for data in all_race_data if data.get('metrics', {}).get('rank_correlation') is not None])
        
        top1_accuracy = np.mean([1 if data.get('metrics', {}).get('top1_correct', False) else 0 
                                for data in all_race_data])
        
        top3_accuracy = np.mean([data.get('metrics', {}).get('top3_accuracy', 0) 
                                for data in all_race_data if data.get('metrics', {}).get('top3_accuracy') is not None])
        
        avg_prediction_score = np.mean([data.get('metrics', {}).get('prediction_score', 0) 
                                       for data in all_race_data if data.get('metrics', {}).get('prediction_score') is not None])
        
        print(f"Overall winner prediction accuracy: {top1_accuracy:.1%}")
        print(f"Overall top-3 prediction accuracy: {top3_accuracy:.1%}")
        print(f"Average rank correlation: {avg_correlation:.3f}")
        print(f"Average prediction score: {avg_prediction_score:.2f}")
        
        # Aggregate team biases across races
        all_team_biases = {}
        for data in all_race_data:
            team_biases = data.get('metrics', {}).get('team_avg_errors', {})
            for team, bias in team_biases.items():
                if team not in all_team_biases:
                    all_team_biases[team] = []
                all_team_biases[team].append(bias)
        
        # Calculate average team bias
        avg_team_biases = {team: sum(biases)/len(biases) for team, biases in all_team_biases.items() if biases}
        
        # Print significant team biases
        print("\nConsistent team biases across races:")
        for team, avg_bias in sorted(avg_team_biases.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(avg_bias) > 1:  # Only show significant biases
                bias_type = "underestimated" if avg_bias < 0 else "overestimated"
                print(f"  {team}: {abs(avg_bias):.1f} positions {bias_type}")
        
        # Create visualizations
        visualize_cross_race_performance(all_race_data)
        
        # Suggest global model improvements
        print("\nGlobal model improvement suggestions:")
        
        # 1. Analyze team-based performance
        print("1. Team-based adjustments:")
        for team, avg_bias in sorted(avg_team_biases.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            adjustment = "increase" if avg_bias > 0 else "decrease"
            print(f"   - {adjustment} performance weights for {team}")
        
        # 2. Track-type specific suggestions
        print("2. Track-type considerations:")
        print("   - Review feature weights for different track categories")
        print("   - Adjust dirty air impact based on track width/layout")
        
        # 3. Feature importance analysis
        if avg_correlation < 0.6:
            print("3. Feature importance reassessment:")
            print("   - Review relative importance of qualifying vs race pace")
            print("   - Consider recalibrating starting position advantage")
        
        return {
            'top1_accuracy': top1_accuracy,
            'top3_accuracy': top3_accuracy,
            'avg_correlation': avg_correlation,
            'avg_prediction_score': avg_prediction_score,
            'team_biases': avg_team_biases
        }
    else:
        print("Not enough race data for cross-race analysis.")
        return None

def visualize_cross_race_performance(all_race_data):
    """Visualize prediction performance across multiple races"""
    # Extract race names and metrics
    race_names = []
    top3_accuracies = []
    correlations = []
    prediction_scores = []
    
    for data in all_race_data:
        if 'race_name' in data:
            race_names.append(data['race_name'])
            metrics = data.get('metrics', {})
            top3_accuracies.append(metrics.get('top3_accuracy', 0))
            correlations.append(metrics.get('rank_correlation', 0))
            prediction_scores.append(metrics.get('prediction_score', 0))
    
    # Plot performance metrics across races
    plt.figure(figsize=(14, 10))
    
    # Create 3-panel plot
    plt.subplot(3, 1, 1)
    plt.bar(race_names, top3_accuracies, color='skyblue')
    plt.ylabel('Top-3 Accuracy')
    plt.title('Prediction Performance Across Races')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    plt.subplot(3, 1, 2)
    plt.bar(race_names, correlations, color='lightgreen')
    plt.ylabel('Rank Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-1, 1)
    
    plt.subplot(3, 1, 3)
    plt.bar(race_names, prediction_scores, color='salmon')
    plt.ylabel('Prediction Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('cross_race_performance.png', dpi=300)
    print("Saved cross-race performance visualization to cross_race_performance.png")
    plt.close()
    
    # Create visualization of team biases
    all_team_biases = {}
    for data in all_race_data:
        team_biases = data.get('metrics', {}).get('team_avg_errors', {})
        for team, bias in team_biases.items():
            if team not in all_team_biases:
                all_team_biases[team] = []
            all_team_biases[team].append(bias)
    
    # Only include teams with data from at least 2 races
    valid_teams = {team: biases for team, biases in all_team_biases.items() if len(biases) >= 2}
    
    if valid_teams:
        # Prepare data for boxplot
        team_names = list(valid_teams.keys())
        team_bias_data = [valid_teams[team] for team in team_names]
        
        # Create boxplot
        plt.figure(figsize=(12, 8))
        plt.boxplot(team_bias_data, labels=team_names, patch_artist=True)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.ylabel('Position Bias (negative = underestimated, positive = overestimated)')
        plt.title('Team Prediction Biases Across Races')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('team_prediction_biases.png', dpi=300)
        print("Saved team prediction biases visualization to team_prediction_biases.png")
        plt.close()

def apply_team_bias_correction(predicted_results, team_biases, correction_strength=0.5):
    """
    Apply correction to prediction results based on observed team biases
    
    Parameters:
    - predicted_results: Dictionary of driver prediction scores
    - team_biases: Dictionary of team average position errors
    - correction_strength: Factor to control correction intensity (0-1)
    
    Returns:
    - corrected_results: Corrected prediction scores
    """
    corrected_results = predicted_results.copy()
    
    if not team_biases:
        return corrected_results
    
    # Get driver to team mapping
    driver_teams = {}
    for race_name in os.listdir('prediction_history'):
        if race_name.endswith('.json'):
            try:
                with open(os.path.join('prediction_history', race_name), 'r') as f:
                    race_data = json.load(f)
                
                if 'driver_teams' in race_data:
                    driver_teams.update(race_data['driver_teams'])
            except:
                pass
    
    # If no driver-team mapping found, cannot apply correction
    if not driver_teams:
        return corrected_results
    
    # Get base score range
    min_score = min(corrected_results.values())
    max_score = max(corrected_results.values())
    score_range = max_score - min_score
    
    # Apply correction based on team bias
    for driver, score in corrected_results.items():
        if driver in driver_teams:
            team = driver_teams[driver]
            if team in team_biases:
                # Negative bias means team is underestimated (need to increase score)
                # Positive bias means team is overestimated (need to decrease score)
                bias = team_biases[team]
                
                # Convert bias to score adjustment
                # Max adjustment is 30% of the score range
                adjustment = -bias * 0.03 * score_range * correction_strength
                
                # Apply adjustment
                corrected_results[driver] = score + adjustment
    
    return corrected_results

def update_model_weights_based_on_analysis(analysis_results, weight_adjustment_strength=0.1):
    """
    Update model feature weights based on analysis of prediction accuracy
    
    Parameters:
    - analysis_results: Results from evaluate_multiple_races
    - weight_adjustment_strength: Factor controlling adjustment magnitude (0.0-1.0)
    
    Returns:
    - suggested_weight_adjustments: Dictionary of suggested weight changes
    """
    if not analysis_results:
        print("No analysis results to use for weight updates.")
        return None
    
    # Start with base feature weights
    base_weights = {
        'quali_weight': 0.4,
        'sector_performance_weight': 0.10,
        'tire_management_weight': 0.10,
        'race_start_weight': 0.08,
        'overtaking_ability_weight': 0.06,
        'team_strategy_weight': 0.08,
        'starting_position_weight': 0.18,
        'team_dynamics_weight': 0.12,
        'dirty_air_weight': 0.12,
        'weather_impact_weight': 0.08,
        'team_form_weight': 0.08,
        'driver_consistency_weight': 0.08
    }
    
    # Create suggested weight adjustments
    suggested_adjustments = {}
    
    # 1. Analyze overall prediction accuracy
    top1_accuracy = analysis_results.get('top1_accuracy', 0)
    avg_correlation = analysis_results.get('avg_correlation', 0)
    
    # If winner predictions are poor but correlation is decent
    if top1_accuracy < 0.5 and avg_correlation > 0.5:
        # Increase weights for factors that affect winning
        suggested_adjustments['starting_position_weight'] = base_weights['starting_position_weight'] * (1 + weight_adjustment_strength)
        suggested_adjustments['team_dynamics_weight'] = base_weights['team_dynamics_weight'] * (1 + weight_adjustment_strength)
        
        # Decrease qualifying weight slightly
        suggested_adjustments['quali_weight'] = base_weights['quali_weight'] * (1 - 0.5 * weight_adjustment_strength)
        
    # If correlation is poor overall
    if avg_correlation < 0.4:
        # The model may be overvaluing minor factors
        suggested_adjustments['sector_performance_weight'] = base_weights['sector_performance_weight'] * (1 - weight_adjustment_strength)
        suggested_adjustments['race_start_weight'] = base_weights['race_start_weight'] * (1 - weight_adjustment_strength)
        
        # Increase core performance indicators
        suggested_adjustments['quali_weight'] = base_weights['quali_weight'] * (1 + weight_adjustment_strength)
        suggested_adjustments['team_strategy_weight'] = base_weights['team_strategy_weight'] * (1 + weight_adjustment_strength)
    
    # 2. Analyze team biases
    team_biases = analysis_results.get('team_biases', {})
    
    # If there are consistent team biases, suggest team-specific adjustments
    significant_biases = [(team, bias) for team, bias in team_biases.items() if abs(bias) > 2]
    
    if significant_biases:
        print("\nSuggested team-specific adjustments:")
        for team, bias in significant_biases:
            adjustment_factor = bias * 0.05 * weight_adjustment_strength
            direction = "Increase" if bias > 0 else "Decrease"
            print(f"  - {direction} prediction scores for {team} by {abs(adjustment_factor):.2f}")
    
    # 3. Normalize and finalize weight adjustments
    # Fill in missing weights with original values
    for weight, value in base_weights.items():
        if weight not in suggested_adjustments:
            suggested_adjustments[weight] = value
    
    # Make sure weights still sum to approximately 1
    total_weight = sum(suggested_adjustments.values())
    normalized_adjustments = {weight: value/total_weight for weight, value in suggested_adjustments.items()}
    
    print("\nSuggested feature weight adjustments:")
    for weight, original_value in base_weights.items():
        new_value = normalized_adjustments[weight]
        if abs(new_value - original_value) > 0.01:
            change = (new_value - original_value) / original_value * 100
            direction = "increase" if change > 0 else "decrease"
            print(f"  - {weight}: {original_value:.2f} -> {new_value:.2f} ({abs(change):.1f}% {direction})")
    
    return normalized_adjustments