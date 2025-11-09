"""
Load model and make travel time predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from model import load_model


def predict_travel_time(model_path: Path, features: np.ndarray) -> np.ndarray:
    """
    Predict travel times using trained model.
    
    Args:
        model_path: Path to the trained model
        features: Feature matrix for predictions
        
    Returns:
        Array of predicted travel times
    """
    model = load_model(model_path)
    predictions = model.predict(features)
    return predictions


def predict_from_dataframe(model_path: Path, df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """
    Predict travel times from DataFrame.
    
    Args:
        model_path: Path to the trained model
        df: DataFrame with feature columns
        feature_cols: List of column names to use as features
        
    Returns:
        Series of predicted travel times
    """
    features = df[feature_cols].values
    predictions = predict_travel_time(model_path, features)
    return pd.Series(predictions, name='predicted_travel_time')


if __name__ == '__main__':
    # Example usage
    model_path = Path('models/travel_time_model.pkl')
    # Add your prediction logic here

