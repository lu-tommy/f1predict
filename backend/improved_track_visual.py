import fastf1
import fastf1.plotting
from fastf1.core import Laps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import pandas as pd
import base64
from io import BytesIO
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('track_visual')

# Enable the cache to save web requests
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

# Configure plotting settings
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, color_scheme='fastf1')

def load_session_data(year, gp_name, session_name):
    """Load data for a specific F1 session.
    
    Args:
        year (int): Year of the Grand Prix
        gp_name (str): Name of the Grand Prix (e.g., 'Monza', 'Monaco')
        session_name (str): Session to load (e.g., 'R' for race, 'Q' for qualifying)
        
    Returns:
        fastf1.core.Session: Session object with loaded data or None if error
    """
    try:
        # Load the session
        session = fastf1.get_session(year, gp_name, session_name)
        # Load the lap and telemetry data
        session.load()
        logger.info(f"Loaded data for {year} {gp_name} {session_name}")
        return session
    except Exception as e:
        logger.error(f"Error loading session data for {year} {gp_name} {session_name}: {e}")
        return None

def get_fastest_lap_telemetry(session, driver=None):
    """Extract telemetry for fastest lap (overall or for a specific driver).
    
    Args:
        session (fastf1.core.Session): Session object with loaded data
        driver (int or str, optional): Driver number or abbreviation. If None, gets overall fastest.
        
    Returns:
        pd.DataFrame: Telemetry data for the fastest lap, or None if error
    """
    try:
        if driver is None:
            # Get overall fastest lap
            fastest_lap = session.laps.pick_fastest()
        else:
            # Get specific driver's fastest lap
            fastest_lap = session.laps.pick_driver(driver).pick_fastest()
        
        if fastest_lap.empty:
            logger.warning(f"No fastest lap found for {driver if driver else 'any driver'}")
            return None
            
        # Get telemetry data for this lap
        telemetry = fastest_lap.get_telemetry()
        return telemetry
    except Exception as e:
        logger.error(f"Error extracting fastest lap telemetry: {e}")
        return None

def create_track_visualization(telemetry, title=None, color_by='Speed', cmap='viridis', 
                              figsize=(10, 6), as_base64=False):
    """Create a track visualization from telemetry data.
    
    Args:
        telemetry (pd.DataFrame): Telemetry data with X and Y position coordinates
        title (str, optional): Plot title. Defaults to None.
        color_by (str, optional): Column name to use for color-coding. Defaults to 'Speed'.
        cmap (str, optional): Matplotlib colormap name. Defaults to 'viridis'.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        as_base64 (bool, optional): Return as base64 encoded string. Defaults to False.
        
    Returns:
        Union[tuple, str]: (fig, ax) matplotlib objects or base64-encoded image string
    """
    try:
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize, dpi=100, facecolor='white')
        
        # Prepare the data
        x = telemetry['X']
        y = telemetry['Y']
        
        # Create points
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Color based on the specified column
        if color_by in telemetry.columns:
            # Create a color map
            norm = plt.Normalize(telemetry[color_by].min(), telemetry[color_by].max())
            lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm, linewidth=3)
            lc.set_array(telemetry[color_by])
            line = ax.add_collection(lc)
            cbar = fig.colorbar(line, ax=ax)
            cbar.set_label(color_by)
        else:
            # If column doesn't exist, use a solid color
            lc = LineCollection(segments, colors='steelblue', linewidth=3)
            line = ax.add_collection(lc)
        
        # Set plot limits and aspect ratio
        ax.set_xlim(x.min() - 100, x.max() + 100)
        ax.set_ylim(y.min() - 100, y.max() + 100)
        ax.set_aspect('equal')
        
        # Remove axis ticks and add title
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        
        # Add a marker for the start/finish line
        if 'Distance' in telemetry.columns:
            start_idx = telemetry['Distance'].abs().idxmin()
            ax.plot(telemetry['X'][start_idx], telemetry['Y'][start_idx], 'ro', 
                    markersize=8, label='Start/Finish')
            ax.legend(loc='upper right')
        
        # Add north arrow (assuming Y-axis points north, which is often the case)
        ax.annotate('', xy=(x.max() + 50, y.max() - 50), xytext=(x.max() + 50, y.max() - 150),
                  arrowprops=dict(arrowstyle='->', color='black', linewidth=2))
        ax.text(x.max() + 60, y.max() - 100, 'N', fontsize=12)
        
        # Add scale bar (assuming units are in meters)
        scale_bar_length = 100  # meters
        scale_bar_x = x.min() + 100
        scale_bar_y = y.min() + 100
        ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], [scale_bar_y, scale_bar_y], 
                'k-', linewidth=2)
        ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - 20, 
                f'{scale_bar_length}m', ha='center')
        
        plt.tight_layout()
        
        if as_base64:
            # Save to a BytesIO object and encode as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            return fig, ax
            
    except Exception as e:
        logger.error(f"Error creating track visualization: {e}")
        if as_base64:
            return None
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error creating visualization: {e}", 
                    ha='center', va='center', transform=ax.transAxes)
            return fig, ax

def compare_driver_lines(session, drivers, lap_number=None, color_by='Team', 
                        figsize=(10, 6), title=None, as_base64=False):
    """Compare driving lines for multiple drivers.
    
    Args:
        session (fastf1.core.Session): Session object with loaded data
        drivers (list): List of driver numbers or abbreviations
        lap_number (int, optional): Specific lap number to compare. Defaults to fastest lap.
        color_by (str, optional): How to color lines ('Team' or 'Driver'). Defaults to 'Team'.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        title (str, optional): Plot title. Defaults to None.
        as_base64 (bool, optional): Return as base64 encoded string. Defaults to False.
        
    Returns:
        Union[tuple, str]: (fig, ax) matplotlib objects or base64-encoded image string
    """
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=100, facecolor='white')
        
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        
        for driver in drivers:
            # Get laps for the driver
            driver_laps = session.laps.pick_driver(driver)
            
            # Skip if no laps available
            if driver_laps.empty:
                logger.warning(f"No laps available for driver {driver}")
                continue
            
            # Select the specific lap or the fastest lap
            if lap_number:
                lap = driver_laps.pick_lap(lap_number)
            else:
                lap = driver_laps.pick_fastest()
            
            if lap.empty:
                logger.warning(f"No data available for driver {driver} on selected lap")
                continue
            
            # Get telemetry data
            telemetry = lap.get_telemetry()
            
            # Update plot limits
            x_min = min(x_min, telemetry['X'].min())
            x_max = max(x_max, telemetry['X'].max())
            y_min = min(y_min, telemetry['Y'].min())
            y_max = max(y_max, telemetry['Y'].max())
            
            # Get driver info
            driver_info = session.get_driver(driver)
            driver_abbr = driver_info['Abbreviation']
            team_name = driver_info['TeamName']
            
            # Determine color based on team or driver
            if color_by == 'Team':
                color = fastf1.plotting.team_color(team_name)
            else:
                color = fastf1.plotting.driver_color(driver_abbr)
            
            # Plot the driving line
            ax.plot(telemetry['X'], telemetry['Y'], color=color, 
                   label=f"{driver_abbr} ({team_name})", linewidth=2)
        
        # Set plot limits with some padding
        padding = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect('equal')
        
        # Remove axis ticks and add title
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title("Driver Line Comparison", fontsize=14, pad=20)
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.7)
        
        # Add north arrow and scale bar (same as in create_track_visualization)
        ax.annotate('', xy=(x_max - 50, y_max - 50), xytext=(x_max - 50, y_max - 150),
                  arrowprops=dict(arrowstyle='->', color='black', linewidth=2))
        ax.text(x_max - 40, y_max - 100, 'N', fontsize=12)
        
        scale_bar_length = 100  # meters
        scale_bar_x = x_min + 100
        scale_bar_y = y_min + 100
        ax.plot([scale_bar_x, scale_bar_x + scale_bar_length], [scale_bar_y, scale_bar_y], 
                'k-', linewidth=2)
        ax.text(scale_bar_x + scale_bar_length/2, scale_bar_y - 20, 
                f'{scale_bar_length}m', ha='center')
        
        plt.tight_layout()
        
        if as_base64:
            # Save to a BytesIO object and encode as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            return fig, ax
    
    except Exception as e:
        logger.error(f"Error creating driver comparison visualization: {e}")
        if as_base64:
            return None
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error creating driver comparison: {e}", 
                    ha='center', va='center', transform=ax.transAxes)
            return fig, ax

def get_track_characteristics_visual(session=None, race_name=None, telemetry=None,
                                   as_base64=False, figsize=(10, 6)):
    """Create track characteristics visualization with corner numbers and speed heatmap.
    
    Args:
        session (fastf1.core.Session, optional): Session object with loaded data
        race_name (str, optional): Name of the race for title
        telemetry (pd.DataFrame, optional): Pre-loaded telemetry data
        as_base64 (bool, optional): Return as base64 encoded string. Defaults to False.
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        
    Returns:
        Union[tuple, str]: (fig, ax) matplotlib objects or base64-encoded image string
    """
    try:
        if telemetry is None and session is not None:
            # Get fastest lap telemetry from the session
            fastest_lap = session.laps.pick_fastest()
            telemetry = fastest_lap.get_telemetry()
        
        if telemetry is None:
            raise ValueError("Either telemetry or session must be provided")
        
        # Create figure with the track layout colored by speed
        fig, ax = plt.subplots(figsize=figsize, dpi=100, facecolor='white')
        
        # Prepare the data
        x = telemetry['X']
        y = telemetry['Y']
        
        # Create points for LineCollection
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a color map based on speed
        norm = plt.Normalize(telemetry['Speed'].min(), telemetry['Speed'].max())
        cmap = plt.get_cmap('viridis')
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4)
        lc.set_array(telemetry['Speed'])
        line = ax.add_collection(lc)
        
        # Add colorbar
        cbar = fig.colorbar(line, ax=ax)
        cbar.set_label('Speed (km/h)')
        
        # Set axis limits and aspect ratio
        ax.set_xlim(x.min() - 100, x.max() + 100)
        ax.set_ylim(y.min() - 100, y.max() + 100)
        ax.set_aspect('equal')
        
        # Remove axis ticks
        ax.set_axis_off()
        
        # Add title
        if race_name:
            title = f"{race_name} Track Characteristics"
        else:
            title = "Track Characteristics Analysis"
        ax.set_title(title, fontsize=14, pad=20)
        
        # Mark start/finish line
        if 'Distance' in telemetry.columns:
            start_idx = telemetry['Distance'].abs().idxmin()
            ax.plot(telemetry['X'][start_idx], telemetry['Y'][start_idx], 'ro', 
                    markersize=8, label='Start/Finish')
            ax.legend(loc='upper right')
        
        # Detect and label corners by finding points of significant direction change
        try:
            # Calculate the direction change at each point
            dx = np.diff(np.array(telemetry['X']))
            dy = np.diff(np.array(telemetry['Y']))
            angles = np.arctan2(dy, dx)
            
            # Calculate change in angle (direction change)
            angle_changes = np.diff(angles)
            
            # Adjust for -pi to pi transition
            angle_changes = np.mod(angle_changes + np.pi, 2 * np.pi) - np.pi
            
            # Find points of significant direction change (corners)
            threshold = 0.3  # Adjust based on track layout complexity
            potential_corners = np.where(np.abs(angle_changes) > threshold)[0]
            
            # Filter out corners that are too close to each other
            min_distance = 10  # Minimum number of points between corners
            filtered_corners = [potential_corners[0]]
            for corner in potential_corners[1:]:
                if corner - filtered_corners[-1] > min_distance:
                    filtered_corners.append(corner)
            
            # Label significant corners
            for i, corner_idx in enumerate(filtered_corners):
                # Add 1 because we're working with diff arrays
                plot_idx = corner_idx + 1
                if plot_idx < len(telemetry):
                    corner_x = telemetry['X'].iloc[plot_idx]
                    corner_y = telemetry['Y'].iloc[plot_idx]
                    corner_speed = telemetry['Speed'].iloc[plot_idx]
                    
                    # Create a small circle to mark the corner
                    ax.plot(corner_x, corner_y, 'o', markersize=6, 
                           markerfacecolor='white', markeredgecolor='black')
                    
                    # Label the corner with a number
                    ax.text(corner_x, corner_y, f"{i+1}", fontsize=8, 
                           ha='center', va='center', fontweight='bold')
                    
                    # Optionally, add speed information
                    # ax.text(corner_x + 15, corner_y + 15, f"{corner_speed:.0f} km/h", 
                    #        fontsize=7, ha='left', va='bottom')
        except Exception as e:
            logger.warning(f"Error detecting corners: {e}")
        
        plt.tight_layout()
        
        if as_base64:
            # Save to a BytesIO object and encode as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            plt.close(fig)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            return fig, ax
    
    except Exception as e:
        logger.error(f"Error creating track characteristics visualization: {e}")
        if as_base64:
            return None
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"Error creating track characteristics: {e}", 
                    ha='center', va='center', transform=ax.transAxes)
            return fig, ax

def visualize_track_for_web(year, race_name, visualization_type='track', driver_numbers=None):
    """Create track visualizations for web display.
    
    Args:
        year (int): Year of the Grand Prix
        race_name (str): Name of the Grand Prix
        visualization_type (str): Type of visualization ('track', 'characteristics', 'comparison')
        driver_numbers (list, optional): List of driver numbers for comparison
        
    Returns:
        dict: Dictionary with visualization data and metadata
    """
    result = {
        'success': False,
        'error': None,
        'image': None,
        'track_name': race_name,
        'year': year,
        'visualization_type': visualization_type
    }
    
    try:
        # Use qualifying session for better track data
        session = load_session_data(year, race_name, 'Q')
        
        # Fall back to race session if qualifying not available
        if session is None:
            session = load_session_data(year, race_name, 'R')
            
        if session is None:
            result['error'] = f"No data available for {race_name} {year}"
            return result
        
        # Get circuit name from event data
        if 'EventName' in session.event:
            result['track_name'] = session.event['EventName']
        
        # Different visualizations based on type
        if visualization_type == 'track':
            # Basic track layout
            telemetry = get_fastest_lap_telemetry(session)
            if telemetry is not None:
                image_data = create_track_visualization(
                    telemetry, 
                    title=f"{race_name} {year} Track Layout",
                    color_by='Speed',
                    as_base64=True
                )
                if image_data:
                    result['image'] = image_data
                    result['success'] = True
                else:
                    result['error'] = "Failed to create track visualization"
            else:
                result['error'] = "No telemetry data available"
                
        elif visualization_type == 'characteristics':
            # Track characteristics with corner numbers and speed heatmap
            image_data = get_track_characteristics_visual(
                session=session,
                race_name=race_name,
                as_base64=True
            )
            if image_data:
                result['image'] = image_data
                result['success'] = True
            else:
                result['error'] = "Failed to create track characteristics"
            
        elif visualization_type == 'comparison':
            # Driver comparison (if driver_numbers provided)
            if not driver_numbers:
                # Use top 3 qualifiers if no drivers specified
                top_laps = session.laps.pick_fastest_per_driver().sort_values('LapTime').head(3)
                driver_numbers = top_laps.index.get_level_values('DriverNumber').tolist()
            
            image_data = compare_driver_lines(
                session,
                drivers=driver_numbers,
                title=f"{race_name} {year} Driver Comparison",
                as_base64=True
            )
            if image_data:
                result['image'] = image_data
                result['success'] = True
                result['drivers'] = driver_numbers
            else:
                result['error'] = "Failed to create driver comparison"
        
        else:
            result['error'] = f"Unknown visualization type: {visualization_type}"
        
    except Exception as e:
        logger.error(f"Error in visualize_track_for_web: {e}")
        result['error'] = str(e)
    
    return result

if __name__ == "__main__":
    # Example usage
    visualization = visualize_track_for_web(2023, 'Monaco', 'characteristics')
    if visualization['success']:
        print(f"Successfully created visualization for {visualization['track_name']}")
        # You would pass visualization['image'] to a web client
    else:
        print(f"Error: {visualization['error']}")