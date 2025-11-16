"""
Load & preprocess GTFS and realtime feeds.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def load_gtfs_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load GTFS data files from the data directory.
    
    Args:
        data_dir: Path to the data directory containing GTFS files
        
    Returns:
        Dictionary of DataFrames for each GTFS file
    """
    gtfs_files = {
        'stops': 'stops.txt',
        'routes': 'routes.txt',
        'trips': 'trips.txt',
        'stop_times': 'stop_times.txt',
        'calendar': 'calendar.txt',
    }
    
    data = {}
    for key, filename in gtfs_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)
    
    return data


def load_realtime_json(filepath: Path) -> Dict:
    """
    Load real-time transit data from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing real-time data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def load_vehicle_positions(filepath: Path) -> pd.DataFrame:
    """
    Load vehicle positions CSV file.
    
    Args:
        filepath: Path to the vehicle positions CSV file
        
    Returns:
        DataFrame with vehicle position data
    """
    df = pd.read_csv(filepath)
    
    # Fix typo in column name: 'spped' -> 'speed'
    if 'spped' in df.columns:
        df = df.rename(columns={'spped': 'speed'})
    
    # Convert timestamp to datetime
    # Timestamps are Unix timestamps (UTC), but GTFS times are in local time (EST/EDT)
    # Convert UTC timestamps to EST/EDT for proper comparison
    if 'timestamp' in df.columns:
        # Convert UTC timestamp to timezone-aware datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        # Convert to EST/EDT timezone
        import pytz
        est_tz = pytz.timezone('US/Eastern')
        df['datetime'] = df['datetime'].dt.tz_convert(est_tz)
        # Remove timezone info for comparison with GTFS (which has no timezone)
        df['datetime'] = df['datetime'].dt.tz_localize(None)
    
    # Convert empty strings to NaN for numeric columns
    numeric_cols = ['bearing', 'speed', 'latitude', 'longitude', 'current_stop_sequence']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def preprocess_gtfs(gtfs_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess GTFS data for feature extraction.
    
    Args:
        gtfs_data: Dictionary of GTFS DataFrames
        
    Returns:
        Preprocessed dictionary of DataFrames
    """
    processed = {}
    
    # Preprocess stop_times: convert time strings to datetime
    if 'stop_times' in gtfs_data:
        stop_times = gtfs_data['stop_times'].copy()
        # Convert arrival_time and departure_time to seconds since midnight
        for col in ['arrival_time', 'departure_time']:
            if col in stop_times.columns:
                stop_times[f'{col}_seconds'] = stop_times[col].apply(
                    lambda x: sum(int(t) * (60 ** (2 - i)) for i, t in enumerate(x.split(':'))) 
                    if pd.notna(x) and ':' in str(x) else None
                )
        processed['stop_times'] = stop_times
    
    # Copy other dataframes
    for key, df in gtfs_data.items():
        if key not in processed:
            processed[key] = df.copy()
    
    return processed


def merge_vehicle_positions_with_stop_times(
    vehicle_positions: pd.DataFrame,
    stop_times: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge vehicle positions with stop_times data.
    Uses time-based matching to find the correct scheduled time when trip_id matching
    results in large time differences.
    
    Args:
        vehicle_positions: DataFrame with vehicle position data
        stop_times: DataFrame with stop_times GTFS data
        
    Returns:
        Merged DataFrame with vehicle positions and scheduled stop information
    """
    
    # Convert stop_times trip_id to string for merging
    stop_times_df = stop_times.copy()
    stop_times_df['trip_id'] = stop_times_df['trip_id'].astype(str)
    
    # Merge on trip_id and stop_id/current_stop_sequence
    # First try merging on trip_id and stop_id
    merged = vehicle_positions.merge(
        stop_times_df,
        left_on=['trip_id', 'stop_id'],
        right_on=['trip_id', 'stop_id'],
        how='left',
        suffixes=('', '_scheduled')
    )
    
    # If no match, try merging on trip_id and current_stop_sequence
    no_match = merged['arrival_time'].isna()
    if no_match.any():
        stop_times_by_sequence = stop_times_df.groupby(['trip_id', 'stop_sequence']).first().reset_index()
        # Merge unmatched rows separately
        unmatched_rows = merged[no_match].copy()
        unmatched_merged = unmatched_rows.merge(
            stop_times_by_sequence,
            left_on=['trip_id_numeric', 'current_stop_sequence'],
            right_on=['trip_id', 'stop_sequence'],
            how='left',
            suffixes=('', '_seq')
        )
        # Combine matched and unmatched rows
        matched_rows = merged[~no_match]
        merged = pd.concat([matched_rows, unmatched_merged], ignore_index=True)
    
    # Calculate time difference between observation and scheduled arrival
    # This helps identify incorrectly matched records
    if 'datetime' in merged.columns and 'arrival_time_seconds' in merged.columns:
        merged['actual_time_seconds'] = (
            merged['datetime'].dt.hour * 3600 + 
            merged['datetime'].dt.minute * 60 + 
            merged['datetime'].dt.second
        )
        merged['time_diff_minutes'] = abs(merged['actual_time_seconds'] - merged['arrival_time_seconds']) / 60.0
        
        # If time difference is > 60 minutes, try to find a better match based on time proximity
        large_diff_mask = merged['time_diff_minutes'] > 60
        if large_diff_mask.any():
            # For records with large time differences, try matching by stop_id and time proximity
            large_diff_rows = merged[large_diff_mask].copy()
            
            def find_best_time_match(row, stop_times_df):
                """Find the scheduled stop with closest arrival time for this stop_id"""
                if pd.isna(row['stop_id']) or pd.isna(row['actual_time_seconds']):
                    return None
                
                # Get all scheduled stops for this stop_id
                stop_schedules = stop_times_df[stop_times_df['stop_id'] == row['stop_id']].copy()
                if len(stop_schedules) == 0:
                    return None
                
                # Calculate time difference
                stop_schedules['time_diff'] = abs(stop_schedules['arrival_time_seconds'] - row['actual_time_seconds']) / 60.0
                
                # Find the closest match (within 2 hours)
                best_match = stop_schedules[stop_schedules['time_diff'] < 120].sort_values('time_diff')
                if len(best_match) > 0:
                    return best_match.iloc[0]
                return None
            
            # Try to find better matches for rows with large time differences
            better_matches = {}
            for idx, row in large_diff_rows.iterrows():
                best_match = find_best_time_match(row, stop_times_df)
                if best_match is not None and best_match['time_diff'] < row['time_diff_minutes']:
                    # Store better match info
                    better_matches[idx] = {
                        'arrival_time': best_match['arrival_time'],
                        'arrival_time_seconds': best_match['arrival_time_seconds'],
                        'departure_time': best_match.get('departure_time', best_match['arrival_time']),
                        'departure_time_seconds': best_match.get('departure_time_seconds', best_match['arrival_time_seconds']),
                        'stop_sequence': best_match['stop_sequence']
                    }
            
            # Update merged dataframe with better matches
            if better_matches:
                for idx, match_info in better_matches.items():
                    merged.loc[idx, 'arrival_time'] = match_info['arrival_time']
                    merged.loc[idx, 'arrival_time_seconds'] = match_info['arrival_time_seconds']
                    merged.loc[idx, 'departure_time'] = match_info['departure_time']
                    if 'departure_time_seconds' in merged.columns:
                        merged.loc[idx, 'departure_time_seconds'] = match_info['departure_time_seconds']
                    merged.loc[idx, 'stop_sequence'] = match_info['stop_sequence']
        
        # Clean up temporary columns
        if 'time_diff_minutes' in merged.columns:
            merged = merged.drop(columns=['time_diff_minutes'])
        if 'actual_time_seconds' in merged.columns and 'actual_time_seconds' not in vehicle_positions.columns:
            # Only drop if it was created here (features.py will create it later)
            pass
    
    return merged


def save_processed_data(data: Dict, output_dir: Path):
    """
    Save processed data to CSV files.
    
    Args:
        data: Dictionary of DataFrames to save
        output_dir: Directory to save processed data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for key, df in data.items():
        output_path = output_dir / f"{key}.csv"
        df.to_csv(output_path, index=False)

