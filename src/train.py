import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
        load_raw, add_features, get_series, chronological_split, fit_scaler, make_windows, SalesDataset, FEATURE_COLS, TARGET_COL, LOOKBACK,
        )
from model import LSTMForecaster

torch.manual_seed(42)

def inverse_log_sales(scaled: np.ndarray, scaler, target_idx: int) -> np.ndarray:
    mean = scaler.mean_[target_idx]
    scale = scaler.scale_[target_idx]
    log_vals = scaled * scale + mean
    return np.expm1(log_vals)

def naive_baseline(split_df, lookback:int):
    actual = split_df["units_sold"].values
    y_true = actual[lookback:]
    y_pred = actual[lookback -1: -1]
    return y_true, y_pred

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs +1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_losses.append(loss_fn(preds, yb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return model

if __name__ == "__main__":
    df = add_features(load_raw())
    series = get_series(df, store_id=9823, sku_id=673209)
    train, val, test = chronological_split(series)

    scaler = fit_scaler(train)
    X_train, y_train = make_windows(train, scaler)
    X_val, y_val = make_windows(val, scaler)
    X_test, y_test = make_windows(test, scaler)

    train_loader = DataLoader(SalesDataset(X_train, y_train), batch_size=8, shuffle=True)
    val_loader = DataLoader(SalesDataset(X_val, y_val), batch_size=len(X_val), shuffle=False)
    test_loader = DataLoader(SalesDataset(X_test, y_test), batch_size=len(X_test), shuffle=False)

    model = LSTMForecaster(n_features=len(FEATURE_COLS))
    model = train_model(model, train_loader, val_loader)
    
    target_idx = FEATURE_COLS.index(TARGET_COL)
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(test_loader))
        preds_scaled = model(xb).numpy()
        y_true_scaled = yb.numpy()

    preds_units = inverse_log_sales(preds_scaled, scaler, target_idx)
    y_true_units = inverse_log_sales(y_true_scaled, scaler, target_idx)

    baseline_true, baseline_pred = naive_baseline(test, LOOKBACK)

    print("\n-- Test set results (real units) ---")
    print(f"LSTM        MAE={mae(y_true_units, preds_units):.2f}        RMSE={rmse(y_true_units, preds_units):.2f}")
    print(f"Baseline    MAE={mae(baseline_true, baseline_pred):.2f}     RMSE={rmse(baseline_true, baseline_pred):.2f}")


