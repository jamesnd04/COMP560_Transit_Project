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

1. Explore data: Start with `notebooks/01_explore.ipynb`
2. Extract features: Use `notebooks/02_features.ipynb`
3. Train model: Run `notebooks/03_model.ipynb` or use `src/model.py`
4. Make predictions: Use `src/predict.py`

## Data

Place your GTFS and real-time data files in `data/raw/`. Processed data will be saved to `data/processed/`.

