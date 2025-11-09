"""
Train and save predictive model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path
from typing import Tuple


def load_training_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from CSV file.
    
    Args:
        data_path: Path to the training data CSV
        
    Returns:
        Tuple of (X, y) where X is features and y is targets
    """
    df = pd.read_csv(data_path)
    
    # Assume last column is target, adjust as needed
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return X, y


def train_model(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[RandomForestRegressor, dict]:
    """
    Train a predictive model.
    
    Args:
        X: Feature matrix
        y: Target values
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
    }
    
    return model, metrics


def save_model(model: RandomForestRegressor, model_path: Path):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_path: Path to save the model
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path: Path) -> RandomForestRegressor:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    return joblib.load(model_path)

