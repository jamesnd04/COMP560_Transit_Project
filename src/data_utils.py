"""
Load & preprocess GTFS and realtime feeds.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional


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


def preprocess_gtfs(gtfs_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Preprocess GTFS data for feature extraction.
    
    Args:
        gtfs_data: Dictionary of GTFS DataFrames
        
    Returns:
        Preprocessed dictionary of DataFrames
    """
    processed = {}
    
    # Add preprocessing logic here
    for key, df in gtfs_data.items():
        processed[key] = df.copy()
    
    return processed


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

