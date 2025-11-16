"""
Feature extraction (time, route, traffic, etc.).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from pathlib import Path


def extract_time_features(df: pd.DataFrame, time_col: str = 'datetime') -> pd.DataFrame:
    """
    Extract time-based features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of the datetime column (default: 'datetime')
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Handle both datetime and timestamp columns
    if time_col not in df.columns and 'timestamp' in df.columns:
        df[time_col] = pd.to_datetime(df['timestamp'], unit='s')
    elif time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
    else:
        return df
    
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['day_of_month'] = df[time_col].dt.day
    df['month'] = df[time_col].dt.month
    df['is_weekend'] = df[time_col].dt.dayofweek >= 5
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    df['is_morning_rush'] = df['hour'].isin([7, 8, 9])
    df['is_evening_rush'] = df['hour'].isin([17, 18, 19])
    df['time_of_day'] = df['hour'] + df[time_col].dt.minute / 60.0
    
    return df


def extract_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract route-based features.
    
    Args:
        df: DataFrame with route information
        
    Returns:
        DataFrame with additional route features
    """
    df = df.copy()
    
    # Extract route ID components if route_id exists
    if 'route_id' in df.columns:
        # Route ID might be in format like "910-13188"
        route_parts = df['route_id'].astype(str).str.split('-', expand=True)
        if len(route_parts.columns) >= 2:
            df['route_prefix'] = route_parts[0]
            df['route_suffix'] = route_parts[1]
        
        # Route frequency (how many times this route appears)
        df['route_frequency'] = df.groupby('route_id')['route_id'].transform('count')
    
    # Extract trip ID components
    if 'trip_id' in df.columns:
        # Trip ID might be in format like "10910002140434-JUNE25"
        trip_parts = df['trip_id'].astype(str).str.split('-', expand=True)
        if len(trip_parts.columns) >= 1:
            df['trip_id_numeric'] = trip_parts[0]
        if len(trip_parts.columns) >= 2:
            df['trip_date_suffix'] = trip_parts[1]
    
    # Vehicle ID features
    if 'vehicle_id' in df.columns:
        df['vehicle_frequency'] = df.groupby('vehicle_id')['vehicle_id'].transform('count')
    
    return df


def extract_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract traffic-related features from vehicle positions.
    
    Args:
        df: DataFrame with vehicle position and traffic data
        
    Returns:
        DataFrame with additional traffic features
    """
    df = df.copy()
    
    # Speed features
    if 'speed' in df.columns:
        df['has_speed'] = df['speed'].notna()
        df['speed'] = df['speed'].fillna(0)
        df['is_moving'] = df['speed'] > 0
        df['speed_category'] = pd.cut(
            df['speed'],
            bins=[-np.inf, 0, 5, 15, 30, np.inf],
            labels=['stopped', 'very_slow', 'slow', 'moderate', 'fast']
        )
    
    # Bearing features
    if 'bearing' in df.columns:
        df['has_bearing'] = df['bearing'].notna()
        df['bearing'] = df['bearing'].fillna(0)
        # Convert bearing to cardinal directions
        df['bearing_direction'] = pd.cut(
            df['bearing'],
            bins=[-1, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360],
            labels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'],
            ordered = False
        )
    
    # Location features
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['has_location'] = df['latitude'].notna() & df['longitude'].notna()
        # Calculate distance from a reference point (e.g., city center) if needed
        # For now, just ensure coordinates are valid
        df['latitude'] = df['latitude'].fillna(0)
        df['longitude'] = df['longitude'].fillna(0)
    
    # Current status features
    if 'current_status' in df.columns:
        df['status_stopped'] = df['current_status'] == 'STOPPED_AT'
        df['status_in_transit'] = df['current_status'] == 'IN_TRANSIT_TO'
        df['status_incoming'] = df['current_status'] == 'INCOMING_AT'
        # One-hot encode status if needed
        status_dummies = pd.get_dummies(df['current_status'], prefix='status')
        df = pd.concat([df, status_dummies], axis=1)
    
    # Stop sequence features
    if 'current_stop_sequence' in df.columns:
        df['stop_sequence'] = df['current_stop_sequence'].fillna(0)
        df['is_at_stop'] = df['current_stop_sequence'].notna()
    
    return df


def extract_delay_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract delay features by comparing actual vs scheduled times.
    
    Args:
        df: DataFrame with merged vehicle positions and stop_times data
        
    Returns:
        DataFrame with delay features
    """
    df = df.copy()
    
    # Calculate delay if we have both actual timestamp and scheduled times
    if 'datetime' in df.columns and 'arrival_time_seconds' in df.columns:
        # Get time of day in seconds from datetime
        df['actual_time_seconds'] = (
            df['datetime'].dt.hour * 3600 + 
            df['datetime'].dt.minute * 60 + 
            df['datetime'].dt.second
        )
        
        # Calculate delay (positive = late, negative = early)
        df['arrival_delay_seconds'] = (
            df['actual_time_seconds'] - df['arrival_time_seconds']
        )
        df['arrival_delay_minutes'] = df['arrival_delay_seconds'] / 60.0
        df['is_delayed'] = df['arrival_delay_seconds'] > 0
        df['is_early'] = df['arrival_delay_seconds'] < 0
        df['is_on_time'] = (df['arrival_delay_seconds'] >= -60) & (df['arrival_delay_seconds'] <= 60)
    
    if 'datetime' in df.columns and 'departure_time_seconds' in df.columns:
        if 'actual_time_seconds' not in df.columns:
            df['actual_time_seconds'] = (
                df['datetime'].dt.hour * 3600 + 
                df['datetime'].dt.minute * 60 + 
                df['datetime'].dt.second
            )
        
        df['departure_delay_seconds'] = (
            df['actual_time_seconds'] - df['departure_time_seconds']
        )
        df['departure_delay_minutes'] = df['departure_delay_seconds'] / 60.0
    
    # Stop sequence comparison
    if 'current_stop_sequence' in df.columns and 'stop_sequence' in df.columns:
        df['sequence_match'] = (
            df['current_stop_sequence'] == df['stop_sequence']
        )
        df['sequence_diff'] = (
            df['current_stop_sequence'] - df['stop_sequence']
        )
    
    return df


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from vehicle positions and stop_times data.
    
    Args:
        df: DataFrame with vehicle positions (optionally merged with stop_times)
        
    Returns:
        DataFrame with all extracted features
    """
    df = df.copy()
    
    # Extract time features
    df = extract_time_features(df)
    
    # Extract route features
    df = extract_route_features(df)
    
    # Extract traffic/vehicle features
    df = extract_traffic_features(df)
    
    # Extract delay features (if stop_times data is present)
    if 'arrival_time' in df.columns or 'departure_time' in df.columns:
        df = extract_delay_features(df)
    
    return df


def create_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Create feature matrix from DataFrame.
    
    Args:
        df: DataFrame with features
        feature_cols: List of column names to use as features
        
    Returns:
        Feature matrix as numpy array
    """
    # Handle categorical columns by converting to numeric
    feature_df = df[feature_cols].copy()
    
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object' or feature_df[col].dtype.name == 'category':
            feature_df[col] = pd.Categorical(feature_df[col]).codes
    
    # Fill NaN values with 0
    feature_df = feature_df.fillna(0)
    
    return feature_df.values

