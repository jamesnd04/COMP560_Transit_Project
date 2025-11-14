"""
Load model and make travel time predictions.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, Dict, List
from datetime import datetime, timedelta
from model import load_model
from data_utils import (
    load_vehicle_positions,
    load_gtfs_data,
    preprocess_gtfs,
    merge_vehicle_positions_with_stop_times
)
from features import extract_all_features, create_feature_matrix


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
    features = create_feature_matrix(df, feature_cols)
    predictions = predict_travel_time(model_path, features)
    return pd.Series(predictions, name='predicted_travel_time')


def get_route_predictions(
    route_id: str,
    data_dir: Path,
    model_path: Optional[Path] = None,
    vehicle_positions_file: Optional[str] = None,
    stop_times_file: str = 'stop_times.txt',
    feature_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get delay and arrival predictions for a specific route.
    
    Args:
        route_id: Route ID to analyze
        data_dir: Directory containing data files
        model_path: Optional path to trained model (if None, returns actual delays)
        vehicle_positions_file: Name of vehicle positions CSV file
        stop_times_file: Name of stop_times.txt file
        feature_cols: List of feature columns for model (if using model)
        
    Returns:
        DataFrame with predictions, delays, and arrival information
    """
    # Load vehicle positions
    if vehicle_positions_file:
        vehicle_positions_path = data_dir / vehicle_positions_file
        if not vehicle_positions_path.exists():
            raise FileNotFoundError(f"Vehicle positions file not found: {vehicle_positions_path}")
        vehicle_positions = load_vehicle_positions(vehicle_positions_path)
    else:
        # Try to find vehicle positions file automatically
        vehicle_positions_files = list(data_dir.glob('vehicle_positions*.csv'))
        if not vehicle_positions_files:
            raise FileNotFoundError(f"No vehicle positions CSV found in {data_dir}")
        vehicle_positions = load_vehicle_positions(vehicle_positions_files[0])
    
    # Filter by route_id
    vehicle_positions = vehicle_positions[vehicle_positions['route_id'] == route_id].copy()
    
    if len(vehicle_positions) == 0:
        raise ValueError(f"No data found for route_id: {route_id}")
    
    # Load GTFS data
    gtfs_data = load_gtfs_data(data_dir)
    if 'stop_times' not in gtfs_data:
        raise FileNotFoundError(f"stop_times.txt not found in {data_dir}")
    
    # Preprocess GTFS
    gtfs_data = preprocess_gtfs(gtfs_data)
    
    # Merge vehicle positions with stop_times
    merged_data = merge_vehicle_positions_with_stop_times(
        vehicle_positions,
        gtfs_data['stop_times']
    )
    
    # Extract features
    features_df = extract_all_features(merged_data)
    
    # If model is provided, use it for predictions
    if model_path and model_path.exists():
        # Load feature columns if not provided
        if feature_cols is None:
            feature_cols_path = model_path.parent / 'feature_columns.json'
            if feature_cols_path.exists():
                with open(feature_cols_path, 'r') as f:
                    feature_cols = json.load(f)
            else:
                print("Warning: feature_columns.json not found. Using available features.")
                feature_cols = None
        
        if feature_cols:
            # Ensure all feature columns exist
            available_cols = [col for col in feature_cols if col in features_df.columns]
            if len(available_cols) != len(feature_cols):
                missing = set(feature_cols) - set(available_cols)
                print(f"Warning: Missing feature columns: {missing}")
            
            # Predict delays using model
            predicted_delays = predict_from_dataframe(model_path, features_df, available_cols)
            features_df['predicted_delay_minutes'] = predicted_delays
        else:
            # Model exists but no feature cols - use actual delays
            if 'arrival_delay_minutes' in features_df.columns:
                features_df['predicted_delay_minutes'] = features_df['arrival_delay_minutes']
            else:
                features_df['predicted_delay_minutes'] = np.nan
    else:
        # Use actual delays if available
        if 'arrival_delay_minutes' in features_df.columns:
            features_df['predicted_delay_minutes'] = features_df['arrival_delay_minutes']
        else:
            features_df['predicted_delay_minutes'] = np.nan
    
    # Calculate expected arrival times
    if 'datetime' in features_df.columns and 'arrival_time' in features_df.columns:
        # Calculate expected arrival based on scheduled time + predicted delay
        try:
            # Combine date from datetime with time from arrival_time
            features_df['scheduled_arrival'] = pd.to_datetime(
                features_df['datetime'].dt.date.astype(str) + ' ' + features_df['arrival_time'].astype(str),
                errors='coerce'
            )
            # Only calculate expected arrival where we have valid scheduled arrival and delay
            valid_mask = features_df['scheduled_arrival'].notna() & features_df['predicted_delay_minutes'].notna()
            features_df['expected_arrival'] = pd.NaT
            features_df.loc[valid_mask, 'expected_arrival'] = (
                features_df.loc[valid_mask, 'scheduled_arrival'] + 
                pd.to_timedelta(features_df.loc[valid_mask, 'predicted_delay_minutes'], unit='minutes')
            )
        except Exception as e:
            # If calculation fails, set to NaT
            features_df['scheduled_arrival'] = pd.NaT
            features_df['expected_arrival'] = pd.NaT
    
    return features_df


def get_route_summary(
    route_id: str,
    data_dir: Path,
    model_path: Optional[Path] = None,
    vehicle_positions_file: Optional[str] = None
) -> Dict:
    """
    Get summary statistics for a route including average delays and arrival predictions.
    
    Args:
        route_id: Route ID to analyze
        data_dir: Directory containing data files
        model_path: Optional path to trained model
        vehicle_positions_file: Name of vehicle positions CSV file
        
    Returns:
        Dictionary with summary statistics
    """
    predictions_df = get_route_predictions(
        route_id, data_dir, model_path, vehicle_positions_file
    )
    
    summary = {
        'route_id': route_id,
        'total_observations': len(predictions_df),
        'unique_trips': predictions_df['trip_id'].nunique() if 'trip_id' in predictions_df.columns else 0,
        'unique_stops': predictions_df['stop_id'].nunique() if 'stop_id' in predictions_df.columns else 0,
    }
    
    # Delay statistics
    if 'predicted_delay_minutes' in predictions_df.columns:
        delay_col = predictions_df['predicted_delay_minutes']
        summary['avg_delay_minutes'] = delay_col.mean()
        summary['median_delay_minutes'] = delay_col.median()
        summary['std_delay_minutes'] = delay_col.std()
        summary['min_delay_minutes'] = delay_col.min()
        summary['max_delay_minutes'] = delay_col.max()
        summary['on_time_percentage'] = (
            (delay_col >= -1) & (delay_col <= 1)
        ).sum() / len(delay_col) * 100 if len(delay_col) > 0 else 0
    
    # Current status breakdown
    if 'current_status' in predictions_df.columns:
        summary['status_breakdown'] = predictions_df['current_status'].value_counts().to_dict()
    
    # Time-based statistics
    if 'hour' in predictions_df.columns:
        summary['peak_delay_hours'] = (
            predictions_df.groupby('hour')['predicted_delay_minutes'].mean()
            .sort_values(ascending=False).head(3).to_dict()
            if 'predicted_delay_minutes' in predictions_df.columns else {}
        )
    
    # Stop-level delays
    if 'stop_id' in predictions_df.columns and 'predicted_delay_minutes' in predictions_df.columns:
        stop_delays = predictions_df.groupby('stop_id')['predicted_delay_minutes'].agg([
            'mean', 'count'
        ]).sort_values('mean', ascending=False)
        summary['worst_stops'] = stop_delays.head(5).to_dict('index')
        summary['best_stops'] = stop_delays.tail(5).to_dict('index')
    
    return summary


def predict_arrival_time(
    route_id: str,
    stop_id: Union[str, int],
    data_dir: Path,
    model_path: Optional[Path] = None,
    vehicle_positions_file: Optional[str] = None,
    current_time: Optional[datetime] = None
) -> Dict:
    """
    Predict arrival time at a specific stop for a route.
    
    Args:
        route_id: Route ID
        stop_id: Stop ID to predict arrival for
        data_dir: Directory containing data files
        model_path: Optional path to trained model
        vehicle_positions_file: Name of vehicle positions CSV file
        current_time: Current time (defaults to now)
        
    Returns:
        Dictionary with arrival predictions
    """
    if current_time is None:
        current_time = datetime.now()
    
    predictions_df = get_route_predictions(
        route_id, data_dir, model_path, vehicle_positions_file
    )
    
    # Filter for the specific stop
    stop_predictions = predictions_df[
        predictions_df['stop_id'].astype(str) == str(stop_id)
    ].copy()
    
    if len(stop_predictions) == 0:
        raise ValueError(f"No data found for route_id={route_id}, stop_id={stop_id}")
    
    # Get most recent predictions
    if 'datetime' in stop_predictions.columns:
        stop_predictions = stop_predictions.sort_values('datetime', ascending=False)
    
    result = {
        'route_id': route_id,
        'stop_id': stop_id,
        'current_time': current_time,
        'observations': len(stop_predictions)
    }
    
    # Average delay for this stop
    if 'predicted_delay_minutes' in stop_predictions.columns:
        avg_delay = stop_predictions['predicted_delay_minutes'].mean()
        result['avg_delay_minutes'] = avg_delay
        result['avg_delay_formatted'] = f"{avg_delay:.1f} minutes"
    
    # Expected arrival times
    if 'expected_arrival' in stop_predictions.columns:
        # Get next expected arrivals
        future_arrivals = stop_predictions[
            pd.to_datetime(stop_predictions['expected_arrival']) > current_time
        ]
        if len(future_arrivals) > 0:
            next_arrivals = future_arrivals.head(3)['expected_arrival'].tolist()
            result['next_expected_arrivals'] = [
                pd.to_datetime(arr).strftime('%Y-%m-%d %H:%M:%S') 
                for arr in next_arrivals
            ]
    
    # Scheduled arrival times
    if 'arrival_time' in stop_predictions.columns:
        # Get unique scheduled times
        scheduled_times = stop_predictions['arrival_time'].unique()
        result['scheduled_arrival_times'] = sorted(scheduled_times.tolist())[:5]
    
    return result


if __name__ == '__main__':
    import sys
    
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python predict.py <route_id> [stop_id]")
        print("\nExamples:")
        print("  python predict.py 910-13188")
        print("  python predict.py 910-13188 30005")
        sys.exit(1)
    
    route_id = sys.argv[1]
    data_dir = Path('data/raw')
    model_path = Path('models/travel_time_model.pkl') if Path('models/travel_time_model.pkl').exists() else None
    
    if model_path:
        print(f"Using trained model: {model_path}")
    else:
        print("No trained model found. Using actual delays from data.")
    
    try:
        if len(sys.argv) >= 3:
            # Predict arrival at specific stop
            stop_id = sys.argv[2]
            result = predict_arrival_time(route_id, stop_id, data_dir, model_path)
            print("\n=== Arrival Prediction ===")
            print(f"Route ID: {result['route_id']}")
            print(f"Stop ID: {result['stop_id']}")
            print(f"Current Time: {result['current_time']}")
            if 'avg_delay_formatted' in result:
                print(f"Average Delay: {result['avg_delay_formatted']}")
            if 'next_expected_arrivals' in result:
                print("\nNext Expected Arrivals:")
                for arrival in result['next_expected_arrivals']:
                    print(f"  - {arrival}")
        else:
            # Get route summary
            summary = get_route_summary(route_id, data_dir, model_path)
            print("\n=== Route Summary ===")
            print(f"Route ID: {summary['route_id']}")
            print(f"Total Observations: {summary['total_observations']}")
            print(f"Unique Trips: {summary['unique_trips']}")
            print(f"Unique Stops: {summary['unique_stops']}")
            
            if 'avg_delay_minutes' in summary:
                print(f"\nDelay Statistics:")
                print(f"  Average Delay: {summary['avg_delay_minutes']:.2f} minutes")
                print(f"  Median Delay: {summary['median_delay_minutes']:.2f} minutes")
                print(f"  On-Time Percentage: {summary['on_time_percentage']:.1f}%")
            
            if 'worst_stops' in summary and summary['worst_stops']:
                print(f"\nWorst Performing Stops (by delay):")
                for stop_id, stats in list(summary['worst_stops'].items())[:3]:
                    print(f"  Stop {stop_id}: {stats['mean']:.2f} min avg delay ({int(stats['count'])} observations)")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

