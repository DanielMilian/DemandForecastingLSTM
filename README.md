## Getting the data
1. Create a Kaggle API token at https://www.kaggle.com/settings → API
2. Place it at `~/.kaggle/kaggle.json`
3. Run: `python scripts/download_data.py`

# Demand Forecasting with LSTM (Work in Progress)

Forecasting weekly product demand using an LSTM in PyTorch, built on a Kaggle SKU-level demand dataset (76 stores × 28 SKUs, weekly, 2011–2013).

**Status:** in progress. Single-series pipeline (data loading, feature engineering, windowing, LSTM, training loop, baseline comparison) is working end-to-end. Currently extending to a pooled model across all 1,155 store/SKU series with embeddings.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_data.py   # requires Kaggle API token, see below
```

## Structure
src/
├── eda.py # exploratory analysis, seasonality checks
├── dataset.py # feature engineering, windowing, PyTorch Dataset
├── model.py # LSTM architecture
└── train.py # training loop + naive baseline comparison
