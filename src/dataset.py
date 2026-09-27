import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/train.csv"
LOOKBACK = 8

FEATURE_COLS = ["log_sales", "total_price", "is_featured_sku", "is_display_sku", "month_sin", "month_cos"]
TARGET_COL = "log_sales"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["week"] = pd.to_datetime(df["week"], format="%d/%m/%y")
    df["total_price"] = df["total_price"].fillna(df["base_price"])
    return df.sort_values(["store_id", "sku_id", "week"])

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_sales"] = np.log1p(df["units_sold"])
    month = df["week"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df

def get_series(df: pd.DataFrame, store_id: int, sku_id: int) -> pd.DataFrame:
    s = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].sort_values("week")
    return s.reset_index(drop=True)

def chronological_split(series: pd.DataFrame, n_val: int = 15, n_test: int = 15):
    n = len(series)
    train = series.iloc[: n - n_val - n_test]
    val = series.iloc[n - n_val - n_test - LOOKBACK: n - n_test]
    test = series.iloc[n - n_test - LOOKBACK:]
    return train, val, test

def fit_scaler(train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train[FEATURE_COLS])
    return scaler

def make_windows(series: pd.DataFrame, scaler: StandardScaler):
    scaled = scaler.transform(series[FEATURE_COLS])
    target_idx = FEATURE_COLS.index(TARGET_COL)
    X, y = [], []
    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i: i + LOOKBACK])
        y.append(scaled[i + LOOKBACK, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class SalesDataset(Dataset):
    def __init__(self, X: np.ndarray, y:np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    df = add_features(load_raw())
    series = get_series(df, store_id=9823, sku_id=673209)
    print("Series length:", len(series))

    train, val, test = chronological_split(series)
    print("Train/val/test raw lengths:", len(train), len(val), len(test))

    scaler = fit_scaler(train)
    X_train, y_train = make_windows(train, scaler)
    X_val, y_val = make_windows(val, scaler)
    X_test, y_test = make_windows(test, scaler)
    print("Windowed shapes:", X_train.shape, X_val.shape, X_test.shape)

    train_ds = SalesDataset(X_train, y_train)
    print("First sample X shape:", train_ds[0][0].shape, "y:", train_ds[0][1].item())






    


