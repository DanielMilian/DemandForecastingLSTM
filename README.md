# Demand Forecasting with LSTM (PyTorch)

An end-to-end weekly demand forecasting project: from raw retail data to a trained LSTM that beats a naive baseline, built to practice deep learning for time series.

**Stack:** Python · PyTorch · pandas · scikit-learn

## What it does

Forecasts next week's unit sales for 1,155 store/product combinations (76 stores × 28 SKUs), using an LSTM that learns shared temporal patterns across all series while distinguishing individual stores and products via embeddings.

```
Raw weekly sales data → feature engineering → per-series scaling → sliding windows
                                                                        ↓
                                              LSTM + store/SKU embeddings → next-week forecast
```

## The dataset detour

I originally started this project on a different Kaggle retail dataset (5 stores × 20 products, daily, 2 years). EDA showed no real weekly seasonality and near-zero correlation between sales and price/discount/weather features — the data looked like unstructured noise rather than realistic demand. Training an LSTM on it would have been meaningless: there was nothing temporal to learn, and any apparent "result" would likely have been noise-fitting rather than a real signal.

I switched to a different dataset (SKU-level weekly demand, 2011–2013) after confirming with EDA that it had genuine signal: visible monthly seasonality (~30% swing between low and high months) and meaningful correlations between sales and `is_featured_sku` (0.34), `is_display_sku` (0.36), and price (-0.24). 

## Data & features

- **1,155 series**, 130 weeks each (2011-01-17 to 2013-07-09), no missing dates
- **Target:** `units_sold`, log-transformed (`log1p`) to tame a handful of extreme outlier weeks
- **Features:** lagged log-sales, price, `is_featured_sku`, `is_display_sku`, and month encoded as sin/cos (avoids a discontinuity between December and January)
- **Dropped:** nothing leaky was present in this dataset (unlike the one I discarded, which had a near-perfect-correlation `Demand Forecast` column)
- **Split:** chronological, per series — last 15 weeks as test, prior 15 as validation, rest as train. Val/test windows include 8 weeks of preceding context so their first prediction has real history to look back on.
- **Scaling:** each series is scaled independently (a 500-units/week product and a 20-units/week product shouldn't share one scale), fit only on that series' training portion to avoid leakage.

## Model

A single-layer LSTM (hidden size 32) reads an 8-week window of features. Its final hidden state is concatenated with learned embeddings for `store_id` (dim 4) and `sku_id` (dim 8), then passed through a linear layer to produce next week's forecast.

## Results

Evaluated on the pooled test set (all 1,155 series, real units):

| Model | MAE | RMSE |
|---|---|---|
| Naive baseline (predict last week's value) | 18.89 | 40.27 |
| **LSTM (store/SKU embeddings)** | **16.85** | **38.50** |

The LSTM improves on the naive baseline by **~11% on MAE** and **~4.4% on RMSE**. RMSE is much larger than MAE for both models, which reflects a few extreme-outlier series in the data (identified during EDA) rather than a problem with the model — most weekly predictions are reasonably close, but a handful of unpredictable spikes pull the squared-error metric up disproportionately.

### A negative result worth reporting
An earlier single-series version of this model (trained on just one store/product pair, ~92 training windows) did **not** beat the baseline (LSTM MAE 12.77 vs. baseline 8.73) — it overfit almost immediately on that little data. Pooling all 1,155 series into one training set (106,260 windows) is what turned this into a working model. I also tried adding dropout (0.2) as a regularization experiment; it made no meaningful difference (MAE 16.91 vs. 16.85), which tells me the model's fast convergence/overfitting pattern is a property of this architecture and dataset combination, not something a light regularization tweak fixes. The dropout code is left in place, commented out, for reference.

## Running it

```bash
git clone https://github.com/DanielMilian/DemandForecastingLSTM
cd DemandForecastingLSTM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get the data via Kaggle API (requires a token at `~/.kaggle/kaggle.json`, see https://www.kaggle.com/settings → API):
```bash
python3 scripts/download_data.py
```

Then:
```bash
python3 src/eda.py       # exploratory analysis, seasonality checks
python3 src/dataset.py   # verify data pipeline (prints shapes)
python3 src/model.py     # verify model architecture (prints shapes)
python3 src/train.py     # train + evaluate against baseline
```

## Project structure

```
.
├── scripts/
│   └── download_data.py
├── src/
│   ├── eda.py         # seasonality/correlation checks
│   ├── dataset.py     # feature engineering, per-series scaling, windowing
│   ├── model.py        # LSTM + store/SKU embeddings
│   └── train.py        # training loop + baseline comparison
├── reports/eda/         # saved EDA plots
├── requirements.txt
└── data/                # gitignored — populated by download_data.py
```

## What I'd do with more time

- Hyperparameter search (learning rate, hidden size, embedding dimensions) rather than the single dropout experiment tried here
- Investigate the extreme-outlier series individually rather than relying on log-transform alone
- Multi-step forecasting (predict several weeks ahead, not just the next one)

