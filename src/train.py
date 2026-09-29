import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
        load_raw, add_features, encode_ids, get_series, chronological_split, 
        build_all_series, SalesDataset, FEATURE_COLS, TARGET_COL, LOOKBACK, N_VAL, N_TEST,
        )
from model import LSTMForecaster

torch.manual_seed(42)

def build_target_lookup(scalers, store_to_idx, sku_to_idx, target_idx, n_stores, n_skus):
    """Precompute (n_stores, n_skus) arrays of each series' log_sales mean/std,
    so any batch of predictions can be inverse-transformed in one vectorized step."""
    mean_lookup = np.zeros((n_stores, n_skus), dtype=np.float32)
    scale_lookup = np.ones((n_stores, n_skus), dtype=np.float32)
    for (store_id, sku_id), scaler in scalers.items():
        s_idx = store_to_idx[store_id]
        k_idx = sku_to_idx[sku_id]
        mean_lookup[s_idx, k_idx] = scaler.mean_[target_idx]
        scale_lookup[s_idx, k_idx] = scaler.scale_[target_idx]
    return mean_lookup, scale_lookup

def invert_batch(scaled_vals: np.ndarray, store_idx: np.ndarray, sku_idx:np.ndarray, 
                  mean_lookup: np.ndarray, scale_lookup: np.ndarray) -> np.ndarray:
    mean = mean_lookup[store_idx, sku_idx]
    scale = scale_lookup[store_idx, sku_idx]
    log_vals = scaled_vals * scale + mean
    return np.expm1(log_vals)

def build_global_baseline(df):
    """Naive lag-1 baseline ('same as last week'), pooled accross every series."""
    combos = df[["store_id", "sku_id"]].drop_duplicates().values.tolist()
    all_true, all_pred = [], []
    for store_id, sku_id in combos:
        series = get_series(df, store_id, sku_id)
        if len(series) < N_VAL + N_TEST + LOOKBACK +1:
            continue
        _, _, test = chronological_split(series)
        actual = test["units_sold"].values
        y_true = actual[LOOKBACK:]
        y_pred = actual[LOOKBACK -1: -1]
        all_true.append(y_true)
        all_pred.append(y_pred)
    return np.concatenate(all_true), np.concatenate(all_pred) 

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs +1):
        model.train()
        train_losses = []
        for xb, store_idx_b, sku_idx_b, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb, store_idx_b, sku_idx_b)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, store_idx_b, sku_idx_b, yb in val_loader:
                preds = model(xb, store_idx_b, sku_idx_b)
                val_losses.append(loss_fn(preds, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model

if __name__ == "__main__":
    df = add_features(load_raw())
    df, store_to_idx, sku_to_idx = encode_ids(df)
    n_stores, n_skus = len(store_to_idx), len(sku_to_idx)
    
    pooled, scalers = build_all_series(df)

    train_ds = SalesDataset(pooled["train"]["X"], pooled["train"]["y"], pooled["train"]["store_idx"], pooled["train"]["sku_idx"])
    val_ds = SalesDataset(pooled["val"]["X"], pooled["val"]["y"], pooled["val"]["store_idx"], pooled["val"]["sku_idx"])
    test_ds = SalesDataset(pooled["test"]["X"], pooled["test"]["y"], pooled["test"]["store_idx"], pooled["test"]["sku_idx"])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    model = LSTMForecaster(n_features=len(FEATURE_COLS), n_stores=n_stores, n_skus=n_skus, dropout=0.2)
    model = train_model(model, train_loader, val_loader, lr=5e-4)
    
    target_idx = FEATURE_COLS.index(TARGET_COL)
    mean_lookup, scale_lookup = build_target_lookup(scalers, store_to_idx, sku_to_idx, target_idx, n_stores, n_skus)

    model.eval()
    with torch.no_grad():
        xb, store_idx_b, sku_idx_b, yb = next(iter(test_loader))
        preds_scaled = model(xb, store_idx_b, sku_idx_b).numpy()
        y_true_scaled = yb.numpy()
        store_idx_np = store_idx_b.numpy()
        sku_idx_np = sku_idx_b.numpy()

    preds_units = invert_batch(preds_scaled, store_idx_np, sku_idx_np, mean_lookup, scale_lookup)
    y_true_units = invert_batch(y_true_scaled, store_idx_np, sku_idx_np, mean_lookup, scale_lookup)

    baseline_true, baseline_pred = build_global_baseline(df)

    print("\n-- Pooled test set results (real units, all 1,155 series) ---")
    print(f"LSTM        MAE={mae(y_true_units, preds_units):.2f}        RMSE={rmse(y_true_units, preds_units):.2f}")
    print(f"Baseline    MAE={mae(baseline_true, baseline_pred):.2f}     RMSE={rmse(baseline_true, baseline_pred):.2f}")


