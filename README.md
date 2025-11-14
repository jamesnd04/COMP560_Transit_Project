# TransitTracker

A machine learning project for predicting transit travel times using GTFS and real-time data.

## Project Structure

```
├── data/                    # Store static GTFS, realtime JSON, and cleaned CSVs
│   ├── raw/                # Raw downloaded data
│   └── processed/          # Preprocessed & feature-ready data
│
├── notebooks/              # Jupyter notebooks for experiments
│   ├── 01_explore.ipynb
│   ├── 02_features.ipynb
│   └── 03_model.ipynb
│
├── src/                    # Core scripts
│   ├── data_utils.py       # Load & preprocess GTFS and realtime feeds
│   ├── features.py         # Feature extraction (time, route, traffic, etc.)
│   ├── model.py            # Train and save predictive model
│   └── predict.py          # Load model and make travel time predictions
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Step-by-Step Workflow

#### 1. Data Exploration (`notebooks/01_explore.ipynb`)
   - Loads vehicle positions and stop_times data
   - Merges data and extracts features
   - Visualizes delays and patterns
   - **Run this first** to understand your data

#### 2. Feature Engineering (`notebooks/02_features.ipynb`)
   - Processes full dataset (or sample for faster processing)
   - Extracts all features from merged data
   - Selects features for model training
   - Saves training data to `data/processed/training_data.csv`
   - **Run this second** to prepare data for training

#### 3. Model Training (`notebooks/03_model.ipynb`)
   - Loads prepared training data
   - Trains Random Forest model to predict delays
   - Evaluates model performance
   - Visualizes feature importance and predictions
   - Saves model to `models/travel_time_model.pkl`
   - **Run this third** to train your model

#### 4. Make Predictions (`src/predict.py`)
   ```bash
   # Get route summary with delay statistics
   python src/predict.py <route_id>
   
   # Predict arrival at specific stop
   python src/predict.py <route_id> <stop_id>
   
   # Example:
   python src/predict.py 910-13188
   python src/predict.py 910-13188 30005
   ```

### Quick Start

1. **Explore your data:**
   ```bash
   jupyter notebook notebooks/01_explore.ipynb
   ```

2. **Prepare features:**
   ```bash
   jupyter notebook notebooks/02_features.ipynb
   ```
   - Adjust `SAMPLE_SIZE` in the notebook if needed
   - Set `USE_SAMPLE = False` for full dataset

3. **Train model:**
   ```bash
   jupyter notebook notebooks/03_model.ipynb
   ```

4. **Make predictions:**
   ```bash
   python src/predict.py 910-13188
   ```

### Notes

- The notebooks process data sequentially - run them in order (01 → 02 → 03)
- For faster development, use samples in notebooks 01 and 02
- The model works with or without training - if no model exists, it uses actual delays from data
- Feature columns are automatically saved and loaded for predictions

## Data

Place your GTFS and real-time data files in `data/raw/`. Processed data will be saved to `data/processed/`.

