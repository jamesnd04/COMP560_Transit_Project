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
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
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
    
    Args:
        vehicle_positions: DataFrame with vehicle position data
        stop_times: DataFrame with stop_times GTFS data
        
    Returns:
        Merged DataFrame with vehicle positions and scheduled stop information
    """
    # Use full trip_id from vehicle positions (including date suffix like "-JUNE25")
    vehicle_df = vehicle_positions.copy()
    vehicle_df['trip_id'] = vehicle_df['trip_id'].astype(str)
    
    # Convert stop_times trip_id to string for merging
    stop_times_df = stop_times.copy()
    stop_times_df['trip_id'] = stop_times_df['trip_id'].astype(str)
    
    # Diagnostic: Check for potential matches before merging
    vehicle_trip_ids = set(vehicle_df['trip_id'].unique())
    stop_times_trip_ids = set(stop_times_df['trip_id'].unique())
    trip_overlap = len(vehicle_trip_ids.intersection(stop_times_trip_ids))
    
    vehicle_stop_ids = set(vehicle_df['stop_id'].dropna().astype(str).unique())
    stop_times_stop_ids = set(stop_times_df['stop_id'].astype(str).unique())
    stop_overlap = len(vehicle_stop_ids.intersection(stop_times_stop_ids))
    
    if trip_overlap == 0 or stop_overlap == 0:
        import warnings
        warnings.warn(
            f"⚠️ Merge Warning: No matching identifiers found. "
            f"Trip ID overlap: {trip_overlap}/{len(vehicle_trip_ids)} vehicle trips, "
            f"Stop ID overlap: {stop_overlap}/{len(vehicle_stop_ids)} vehicle stops. "
            f"This may indicate incompatible datasets or missing GTFS files (trips.txt, stops.txt).",
            UserWarning
        )
    
    # Merge on trip_id and stop_id/current_stop_sequence
    # First try merging on trip_id and stop_id
    # Convert stop_id to string for both to ensure matching
    vehicle_df['stop_id'] = vehicle_df['stop_id'].astype(str)
    stop_times_df['stop_id'] = stop_times_df['stop_id'].astype(str)
    
    merged = vehicle_df.merge(
        stop_times_df,
        left_on=['trip_id', 'stop_id'],
        right_on=['trip_id', 'stop_id'],
        how='left',
        suffixes=('', '_scheduled')
    )
    
    # Check first merge success
    first_merge_success = merged['arrival_time'].notna().sum()
    
    # If no match, try merging on trip_id and current_stop_sequence
    no_match = merged['arrival_time'].isna()
    if no_match.any():
        # Convert current_stop_sequence to match stop_sequence type
        if 'current_stop_sequence' in vehicle_df.columns:
            vehicle_df['current_stop_sequence'] = pd.to_numeric(vehicle_df['current_stop_sequence'], errors='coerce')
        stop_times_df['stop_sequence'] = pd.to_numeric(stop_times_df['stop_sequence'], errors='coerce')
        
        stop_times_by_sequence = stop_times_df.groupby(['trip_id', 'stop_sequence']).first().reset_index()
        # Merge unmatched rows separately
        unmatched_rows = merged[no_match].copy()
        unmatched_merged = unmatched_rows.merge(
            stop_times_by_sequence,
            left_on=['trip_id', 'current_stop_sequence'],
            right_on=['trip_id', 'stop_sequence'],
            how='left',
            suffixes=('', '_seq')
        )
        # Combine matched and unmatched rows
        matched_rows = merged[~no_match]
        merged = pd.concat([matched_rows, unmatched_merged], ignore_index=True)
    
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

