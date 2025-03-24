from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO

# Import your prediction modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("FastF1 not available. Install with: pip install fastf1")

try:
    from f1predict import improved_predict_race_winner, load_session_safely, get_driver_team_mapping
    from track_analyzer import analyze_track_characteristics
    from track_database import get_track_data, get_optimized_track_weights
    from improved_track_visual import visualize_track_for_web
    PREDICTION_MODULES_AVAILABLE = True
except ImportError as e:
    PREDICTION_MODULES_AVAILABLE = False
    print(f"Failed to import prediction modules: {e}")

# Create Flask app
app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app)  # Enable CORS for all routes

# Setup FastF1 cache
if FASTF1_AVAILABLE:
    cache_dir = 'f1_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)

@app.route('/')
def index():
    """Serve the frontend app"""
    return app.send_static_file('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get backend status and available modules"""
    return jsonify({
        'status': 'online',
        'fastf1_available': FASTF1_AVAILABLE,
        'prediction_modules_available': PREDICTION_MODULES_AVAILABLE
    })

@app.route('/api/races', methods=['GET'])
def get_races():
    """Get list of available races for prediction"""
    year = request.args.get('year', default=datetime.now().year, type=int)
    
    if not FASTF1_AVAILABLE:
        return jsonify({'error': 'FastF1 module not available'}), 400
    
    try:
        schedule = fastf1.get_event_schedule(year)
        races = []
        
        current_date = pd.Timestamp.now()
        
        for idx, event in schedule.iterrows():
            # Check if race has already happened
            race_date = event['EventDate'] if pd.notna(event['EventDate']) else None
            is_past_race = race_date and race_date < current_date
            
            races.append({
                'name': event['EventName'],
                'round': int(event['RoundNumber']),
                'date': event['EventDate'].strftime('%Y-%m-%d') if pd.notna(event['EventDate']) else None,
                'country': event['Country'],
                'isPast': is_past_race,
                'hasResults': is_past_race  # We'll assume past races have results, but this could be checked more precisely
            })
        
        return jsonify({'races': races, 'year': year})
    except Exception as e:
        return jsonify({'error': f'Error getting races: {str(e)}'}), 500

@app.route('/api/drivers', methods=['GET'])
def get_drivers():
    """Get current drivers and their teams for a specific race"""
    year = request.args.get('year', default=datetime.now().year, type=int)
    race_name = request.args.get('race', default=None, type=str)
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    if not FASTF1_AVAILABLE or not PREDICTION_MODULES_AVAILABLE:
        return jsonify({'error': 'Required modules not available'}), 400
    
    try:
        # Load any session from the race to get driver info
        session = load_session_safely(year, race_name, 'Q')
        
        if session is None:
            # Try FP sessions if qualifying isn't available
            session = load_session_safely(year, race_name, 'FP3')
            
        if session is None:
            session = load_session_safely(year, race_name, 'FP2')
            
        if session is None:
            session = load_session_safely(year, race_name, 'FP1')
            
        if session is None:
            return jsonify({'error': 'No session data available for this race'}), 404
        
        # Get driver-team mapping
        driver_teams = get_driver_team_mapping(session)
        
        # Format response
        drivers = []
        for driver, team in driver_teams.items():
            drivers.append({
                'code': driver,
                'team': team
            })
        
        return jsonify({'drivers': drivers})
    except Exception as e:
        return jsonify({'error': f'Error getting driver data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_race():
    """Make a race prediction or get actual results for past races"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    race_name = data.get('race')
    year = data.get('year', datetime.now().year)
    show_comparison = data.get('compare', False)  # New flag to explicitly request comparison
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    if not PREDICTION_MODULES_AVAILABLE:
        return jsonify({'error': 'Prediction modules not available'}), 400
    
    try:
        # Check if the race has already happened
        race_date = None
        is_past_race = False
        
        # Get race schedule to check date
        schedule = fastf1.get_event_schedule(year)
        for idx, event in schedule.iterrows():
            if event['EventName'] == race_name:
                if pd.notna(event['EventDate']):
                    race_date = event['EventDate']
                    # Check if race date is in the past
                    is_past_race = race_date < pd.Timestamp.now()
                break
        
        # If race is in the past and we want to show comparison
        if is_past_race and show_comparison:
            # For past races with comparison requested, we'll get both prediction and actual results
            
            # 1. Make prediction
            optimized_weights = get_optimized_track_weights(race_name, year)
            winner, predicted_results = improved_predict_race_winner(race_name, optimized_weights)
            
            # Format prediction results
            predicted_positions = []
            for i, (driver, score) in enumerate(sorted(predicted_results.items(), key=lambda x: x[1], reverse=True)):
                predicted_positions.append({
                    'position': i + 1,
                    'driver': driver,
                    'score': float(score)
                })
            
            # Create prediction visualization
            prediction_img = create_prediction_visualization(race_name, year, predicted_results)
            
            # 2. Get actual results
            race_session = load_session_safely(year, race_name, 'R')
            
            if race_session is None:
                # If we can't load race results, fall back to prediction only
                return jsonify({
                    'winner': winner,
                    'positions': predicted_positions,
                    'visualization': prediction_img,
                    'isPrediction': True,
                    'raceName': race_name,
                    'year': year
                })
            
            # Get actual results
            actual_positions = []
            driver_teams = get_driver_team_mapping(race_session)
            actual_winner = None
            
            # Extract race results
            results = race_session.results
            for _, row in results.iterrows():
                if 'Abbreviation' in row and 'Position' in row and pd.notna(row['Position']):
                    driver_code = row['Abbreviation']
                    position = int(row['Position'])
                    
                    if position == 1:
                        actual_winner = driver_code
                    
                    # Get additional data if available
                    team = driver_teams.get(driver_code, 'Unknown')
                    points = float(row['Points']) if 'Points' in row and pd.notna(row['Points']) else 0
                    
                    # Include fastest lap status if available
                    fastest_lap = False
                    if 'FastestLap' in row and pd.notna(row['FastestLap']) and row['FastestLap'] == 1:
                        fastest_lap = True
                    
                    actual_positions.append({
                        'position': position,
                        'driver': driver_code,
                        'team': team,
                        'points': points,
                        'fastestLap': fastest_lap
                    })
            
            # Sort by position
            actual_positions = sorted(actual_positions, key=lambda x: x['position'])
            
            # Create actual results visualization
            results_img = create_results_visualization(race_name, year, actual_positions)
            
            # 3. Calculate accuracy metrics
            accuracy_metrics = calculate_prediction_accuracy(predicted_positions, actual_positions)
            
            # 4. Create comparison visualization
            comparison_img = create_comparison_visualization(race_name, year, predicted_positions, actual_positions)
            
            # Return combined data
            return jsonify({
                'comparison': True,
                'prediction': {
                    'winner': winner,
                    'positions': predicted_positions,
                    'visualization': prediction_img,
                    'isPrediction': True
                },
                'actual': {
                    'winner': actual_winner,
                    'positions': actual_positions,
                    'visualization': results_img,
                    'isPrediction': False,
                    'actualResults': True
                },
                'accuracy': accuracy_metrics,
                'comparisonVisualization': comparison_img,
                'raceName': race_name,
                'year': year
            })
            
        elif is_past_race:
            # For past races without comparison, just get actual results
            race_session = load_session_safely(year, race_name, 'R')
            
            if race_session is None:
                # If race session can't be loaded, fallback to prediction
                optimized_weights = get_optimized_track_weights(race_name, year)
                winner, predicted_results = improved_predict_race_winner(race_name, optimized_weights)
                
                # Format results
                positions = []
                for i, (driver, score) in enumerate(sorted(predicted_results.items(), key=lambda x: x[1], reverse=True)):
                    positions.append({
                        'position': i + 1,
                        'driver': driver,
                        'score': float(score)
                    })
                
                # Create visualization
                img_base64 = create_prediction_visualization(race_name, year, predicted_results)
                
                return jsonify({
                    'winner': winner,
                    'positions': positions,
                    'visualization': img_base64,
                    'isPrediction': True,
                    'raceName': race_name,
                    'year': year
                })
            
            # Extract race results
            results = race_session.results
            positions = []
            
            # Get driver team mapping for this race
            driver_teams = get_driver_team_mapping(race_session)
            
            # Find winner
            winner = None
            
            for _, row in results.iterrows():
                if 'Abbreviation' in row and 'Position' in row and pd.notna(row['Position']):
                    driver_code = row['Abbreviation']
                    position = int(row['Position'])
                    
                    if position == 1:
                        winner = driver_code
                    
                    # Get additional data if available
                    team = driver_teams.get(driver_code, 'Unknown')
                    points = float(row['Points']) if 'Points' in row and pd.notna(row['Points']) else 0
                    
                    # Include fastest lap status if available
                    fastest_lap = False
                    if 'FastestLap' in row and pd.notna(row['FastestLap']) and row['FastestLap'] == 1:
                        fastest_lap = True
                    
                    positions.append({
                        'position': position,
                        'driver': driver_code,
                        'team': team,
                        'points': points,
                        'fastestLap': fastest_lap
                    })
            
            # Sort by position
            positions = sorted(positions, key=lambda x: x['position'])
            
            # Create visualization
            img_base64 = create_results_visualization(race_name, year, positions)
            
            return jsonify({
                'winner': winner,
                'positions': positions,
                'visualization': img_base64,
                'isPrediction': False,
                'raceName': race_name,
                'year': year,
                'actualResults': True
            })
        else:
            # Future race, make a prediction
            optimized_weights = get_optimized_track_weights(race_name, year)
            winner, predicted_results = improved_predict_race_winner(race_name, optimized_weights)
            
            # Format results
            positions = []
            for i, (driver, score) in enumerate(sorted(predicted_results.items(), key=lambda x: x[1], reverse=True)):
                positions.append({
                    'position': i + 1,
                    'driver': driver,
                    'score': float(score)
                })
            
            # Create visualization
            img_base64 = create_prediction_visualization(race_name, year, predicted_results)
            
            return jsonify({
                'winner': winner,
                'positions': positions,
                'visualization': img_base64,
                'isPrediction': True,
                'raceName': race_name,
                'year': year
            })
    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500

def create_prediction_visualization(race_name, year, predicted_results):
    """Create a visualization for predictions"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Get top drivers (for better visibility)
        sorted_results = sorted(predicted_results.items(), key=lambda x: x[1], reverse=True)
        top_n = min(10, len(sorted_results))
        top_drivers = [d[0] for d in sorted_results[:top_n]]
        scores = [predicted_results[d] for d in top_drivers]
        
        plt.bar(top_drivers, scores, color='steelblue')
        plt.xlabel('Driver')
        plt.ylabel('Prediction Score')
        plt.title(f'F1 Race Winner Prediction: {race_name} {year}')
        plt.xticks(rotation=45)
        
        # Save visualization to base64 string
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error creating prediction visualization: {e}")
        return None

def create_results_visualization(race_name, year, positions):
    """Create a visualization for actual race results"""
    try:
        plt.figure(figsize=(10, 6))
        
        # Get top drivers for visualization (max 10)
        top_drivers = [p['driver'] for p in positions[:10]]
        points = [p['points'] for p in positions[:10]]
        
        # Create a colorful bar chart
        bars = plt.bar(top_drivers, points, color='green')
        plt.xlabel('Driver')
        plt.ylabel('Points')
        plt.title(f'F1 Race Results: {race_name} {year}')
        plt.xticks(rotation=45)
        
        # Highlight fastest lap if available
        for i, p in enumerate(positions[:10]):
            if p.get('fastestLap', False):
                bars[i].set_color('purple')
        
        # Save visualization to base64 string
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error creating results visualization: {e}")
        return None

def create_comparison_visualization(race_name, year, predicted_positions, actual_positions):
    """Create visualization comparing predicted vs actual positions"""
    try:
        # Get common drivers between predictions and results
        pred_by_driver = {p['driver']: p for p in predicted_positions}
        actual_by_driver = {p['driver']: p for p in actual_positions}
        common_drivers = set(pred_by_driver.keys()) & set(actual_by_driver.keys())
        
        # Limit to top drivers for visibility
        top_actual_drivers = [p['driver'] for p in actual_positions if p['position'] <= 10]
        common_top_drivers = [d for d in top_actual_drivers if d in common_drivers]
        
        if not common_top_drivers:
            return None
        
        # Collect positions for visualization
        labels = common_top_drivers
        pred_positions = [pred_by_driver[d]['position'] for d in common_top_drivers]
        actual_positions = [actual_by_driver[d]['position'] for d in common_top_drivers]
        
        # Create comparison chart
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, pred_positions, width, label='Predicted', color='steelblue', alpha=0.7)
        rects2 = ax.bar(x + width/2, actual_positions, width, label='Actual', color='green', alpha=0.7)
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Position (lower is better)')
        ax.set_title(f'Predicted vs Actual Positions: {race_name} {year}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Invert the y-axis since lower positions are better
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        # Save visualization to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plt.close()
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        return None

def calculate_prediction_accuracy(predicted_positions, actual_positions):
    """Calculate accuracy metrics for prediction vs actual results"""
    try:
        # Initialize metrics
        metrics = {
            'winner_correct': False,
            'podium_accuracy': 0.0,
            'top_10_accuracy': 0.0,
            'position_avg_error': 0.0,
            'matched_positions': 0,
            'total_drivers': min(len(predicted_positions), len(actual_positions))
        }
        
        # Build lookup dictionaries
        pred_by_driver = {p['driver']: p for p in predicted_positions}
        actual_by_driver = {p['driver']: p for p in actual_positions}
        
        # Get common drivers
        common_drivers = set(pred_by_driver.keys()) & set(actual_by_driver.keys())
        metrics['total_drivers'] = len(common_drivers)
        
        if not common_drivers:
            return metrics
        
        # Check winner prediction
        pred_winner = next((p['driver'] for p in predicted_positions if p['position'] == 1), None)
        actual_winner = next((p['driver'] for p in actual_positions if p['position'] == 1), None)
        metrics['winner_correct'] = pred_winner == actual_winner
        
        # Calculate podium accuracy (top 3)
        pred_podium = [p['driver'] for p in predicted_positions if p['position'] <= 3]
        actual_podium = [p['driver'] for p in actual_positions if p['position'] <= 3]
        podium_correct = sum(1 for driver in pred_podium if driver in actual_podium)
        metrics['podium_accuracy'] = podium_correct / 3 if pred_podium and actual_podium else 0
        
        # Calculate top 10 accuracy
        pred_top10 = [p['driver'] for p in predicted_positions if p['position'] <= 10]
        actual_top10 = [p['driver'] for p in actual_positions if p['position'] <= 10]
        top10_correct = sum(1 for driver in pred_top10 if driver in actual_top10)
        metrics['top_10_accuracy'] = top10_correct / 10 if pred_top10 and actual_top10 else 0
        
        # Calculate position error
        position_errors = []
        exact_matches = 0
        
        for driver in common_drivers:
            pred_pos = pred_by_driver[driver]['position']
            actual_pos = actual_by_driver[driver]['position']
            error = abs(pred_pos - actual_pos)
            position_errors.append(error)
            
            if error == 0:
                exact_matches += 1
        
        metrics['position_avg_error'] = sum(position_errors) / len(position_errors) if position_errors else 0
        metrics['matched_positions'] = exact_matches
        
        return metrics
    except Exception as e:
        print(f"Error calculating prediction accuracy: {e}")
        return {
            'error': str(e),
            'winner_correct': False,
            'podium_accuracy': 0.0,
            'top_10_accuracy': 0.0,
            'position_avg_error': 0.0
        }

@app.route('/api/track-data', methods=['GET'])
def get_track_data_api():
    """Get track data for a specific race"""
    race_name = request.args.get('race', default=None, type=str)
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    try:
        # Get track data
        track_data = get_track_data(race_name)
        
        if track_data is None:
            return jsonify({'error': 'No track data available for this race'}), 404
        
        # Convert any numpy types to native Python types
        clean_data = {}
        for key, value in track_data.items():
            if isinstance(value, (np.int32, np.int64)):
                clean_data[key] = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                clean_data[key] = float(value)
            else:
                clean_data[key] = value
        
        return jsonify(clean_data)
    except Exception as e:
        return jsonify({'error': f'Error getting track data: {str(e)}'}), 500
@app.route('/api/track-visualization', methods=['GET'])
def get_track_visualization():
    """Get track visualization image for a specific race"""
    race_name = request.args.get('race', default=None, type=str)
    year = request.args.get('year', default=datetime.now().year - 1, type=int)
    viz_type = request.args.get('type', default='track', type=str)
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    try:
        # Generate track visualization
        visualization = visualize_track_for_web(year, race_name, visualization_type=viz_type)
        
        if not visualization['success']:
            return jsonify({'error': visualization['error']}), 404
        
        return jsonify({
            'image': visualization['image'],
            'track_name': visualization['track_name'],
            'year': visualization['year'],
            'visualization_type': visualization['visualization_type']
        })
    except Exception as e:
        return jsonify({'error': f'Error generating track visualization: {str(e)}'}), 500

@app.route('/api/driver-comparison', methods=['POST'])
def get_driver_comparison():
    """Compare driving lines for multiple drivers on a specific track"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    race_name = data.get('race')
    year = data.get('year', datetime.now().year - 1)
    driver_numbers = data.get('drivers', [])
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    if not driver_numbers or len(driver_numbers) < 2:
        return jsonify({'error': 'At least two drivers required for comparison'}), 400
    
    try:
        # Generate driver comparison
        visualization = visualize_track_for_web(
            year, 
            race_name, 
            visualization_type='comparison',
            driver_numbers=driver_numbers
        )
        
        if not visualization['success']:
            return jsonify({'error': visualization['error']}), 404
        
        return jsonify({
            'image': visualization['image'],
            'track_name': visualization['track_name'],
            'year': visualization['year'],
            'drivers': visualization['drivers']
        })
    except Exception as e:
        return jsonify({'error': f'Error generating driver comparison: {str(e)}'}), 500

@app.route('/api/all-track-visualizations', methods=['GET'])
def get_all_track_visualizations():
    """Get all types of track visualizations for a specific race"""
    race_name = request.args.get('race', default=None, type=str)
    year = request.args.get('year', default=datetime.now().year - 1, type=int)
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    try:
        # Generate all types of track visualizations
        track_viz = visualize_track_for_web(year, race_name, visualization_type='track')
        characteristics_viz = visualize_track_for_web(year, race_name, visualization_type='characteristics')
        
        # Generate comparison of top 3 qualifiers
        comparison_viz = visualize_track_for_web(year, race_name, visualization_type='comparison')
        
        # Collect results
        result = {
            'track_name': race_name,
            'year': year,
            'visualizations': {}
        }
        
        # Add successful visualizations
        if track_viz['success']:
            result['visualizations']['track'] = track_viz['image']
        
        if characteristics_viz['success']:
            result['visualizations']['characteristics'] = characteristics_viz['image']
        
        if comparison_viz['success']:
            result['visualizations']['comparison'] = comparison_viz['image']
            result['comparison_drivers'] = comparison_viz.get('drivers', [])
        
        # Return error if no visualizations were successful
        if not result['visualizations']:
            return jsonify({'error': 'Could not generate any track visualizations'}), 404
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error generating track visualizations: {str(e)}'}), 500
    
@app.route('/api/analyze-track', methods=['POST'])
def analyze_track():
    """Analyze track characteristics"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    race_name = data.get('race')
    year = data.get('year', datetime.now().year - 1)  # Use previous year by default
    
    if not race_name:
        return jsonify({'error': 'Race name required'}), 400
    
    if not PREDICTION_MODULES_AVAILABLE:
        return jsonify({'error': 'Analysis modules not available'}), 400
    
    try:
        # Analyze track
        track_data = analyze_track_characteristics(
            race_name, 
            year=year,
            load_session_function=load_session_safely
        )
        
        if track_data is None:
            return jsonify({'error': 'Could not analyze track data'}), 404
        
        # Convert any numpy types to native Python types
        clean_data = {}
        for key, value in track_data.items():
            if isinstance(value, (np.int32, np.int64)):
                clean_data[key] = int(value)
            elif isinstance(value, (np.float32, np.float64)):
                clean_data[key] = float(value)
            elif isinstance(value, dict):
                clean_dict = {}
                for k, v in value.items():
                    if isinstance(v, (np.int32, np.int64)):
                        clean_dict[k] = int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        clean_dict[k] = float(v)
                    else:
                        clean_dict[k] = v
                clean_data[key] = clean_dict
            else:
                clean_data[key] = value
        
        # Path to visualization file
        viz_path = f"track_analysis/{year}_{race_name.replace(' ', '_')}_analysis.png"
        
        # Check if visualization exists
        image_base64 = None
        if os.path.exists(viz_path):
            with open(viz_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        return jsonify({
            'track_data': clean_data,
            'visualization': image_base64
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing track: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)