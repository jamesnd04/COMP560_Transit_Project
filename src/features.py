"""
Feature extraction (time, route, traffic, etc.).
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List


def extract_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Extract time-based features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of the datetime column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['month'] = df[time_col].dt.month
    df['is_weekend'] = df[time_col].dt.dayofweek >= 5
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19])
    
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
    
    # Add route feature extraction logic here
    # Example: route frequency, route type, etc.
    
    return df


def extract_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract traffic-related features.
    
    Args:
        df: DataFrame with traffic data
        
    Returns:
        DataFrame with additional traffic features
    """
    df = df.copy()
    
    # Add traffic feature extraction logic here
    # Example: congestion levels, historical averages, etc.
    
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
    return df[feature_cols].values

