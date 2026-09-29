import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/train.csv"
LOOKBACK = 8

FEATURE_COLS = ["log_sales", "total_price", "is_featured_sku", "is_display_sku", "month_sin", "month_cos"]
TARGET_COL = "log_sales"
N_VAL = 15
N_TEST = 15


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

def encode_ids(df: pd.DataFrame):
    "Map raw store_id/sku_id values to contiguous 0...N-1 indices for embeddings."
    store_ids = sorted(df["store_id"].unique())
    sku_ids = sorted(df["sku_id"].unique())
    store_to_idx = {s: i for i, s in enumerate(store_ids)}
    sku_to_idx = {s: i for i, s in enumerate(sku_ids)}

    df = df.copy()
    df["store_idx"] = df["store_id"].map(store_to_idx)
    df["sku_idx"] = df["sku_id"].map(sku_to_idx)
    return df, store_to_idx, sku_to_idx


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
    "Return X (windows), y (targets), store_idx, sku_idx (one per window)"
    scaled = scaler.transform(series[FEATURE_COLS])
    target_idx = FEATURE_COLS.index(TARGET_COL)
    store_idx_val = series["store_idx"].iloc[0]
    sku_idx_val = series["sku_idx"].iloc[0]

    X, y, store_idxs, sku_idxs = [], [], [], []
    for i in range(len(scaled) - LOOKBACK):
        X.append(scaled[i: i + LOOKBACK])
        y.append(scaled[i + LOOKBACK, target_idx])
        store_idxs.append(store_idx_val)
        sku_idxs.append(sku_idx_val)

    return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(store_idxs, dtype=np.int64),
            np.array(sku_idxs, dtype=np.int64)
         )

def build_all_series(df: pd.DataFrame):
    """Loops over every (store_id, sku_id) pair, splits and scales each independentely,
    and pools all windows into single train/val/test arrays.
    Return dicts keyed by split name, plus the per-series scalers"""
    combos = df[["store_id", "sku_id"]].drop_duplicates().values.tolist()

    pooled = {
            "train":    {"X": [], "y": [], "store_idx": [], "sku_idx":[]},
            "val":      {"X": [], "y": [], "store_idx": [], "sku_idx":[]},
            "test":     {"X": [], "y": [], "store_idx": [], "sku_idx":[]}
            }
    scalers = {}

    skipped = 0
    for store_id, sku_id in combos:
        series = get_series(df, store_id, sku_id)
        if len(series) < N_VAL + N_TEST + LOOKBACK + 1:
            skipped += 1
            continue
        
        train, val, test = chronological_split(series)
        scaler = fit_scaler(train)
        scalers[(store_id, sku_id)] = scaler

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            X, y, store_idxs, sku_idxs = make_windows(split_df, scaler)
            if len(X) == 0:
                continue
            pooled[split_name]["X"].append(X)
            pooled[split_name]["y"].append(y)
            pooled[split_name]["store_idx"].append(store_idxs)
            pooled[split_name]["sku_idx"].append(sku_idxs)

    if skipped:
        print(f"Skipped {skipped} series (too short for the split).")

    result = {}
    for split_name in pooled:
        result[split_name] = {
                "X": np.concatenate(pooled[split_name]["X"], axis=0),
                "y": np.concatenate(pooled[split_name]["y"], axis=0),
                "store_idx": np.concatenate(pooled[split_name]["store_idx"], axis=0),
                "sku_idx": np.concatenate(pooled[split_name]["sku_idx"], axis=0)

                }
    return result, scalers


class SalesDataset(Dataset):
    def __init__(self, X: np.ndarray, y:np.ndarray, store_idx: np.ndarray, sku_idx: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.store_idx = torch.from_numpy(store_idx)
        self.sku_idx = torch.from_numpy(sku_idx)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.store_idx[idx], self.sku_idx[idx], self.y[idx] 

if __name__ == "__main__":
    df = add_features(load_raw())
    df, store_to_idx, sku_to_idx = encode_ids(df) 
    print("Num stores: ", len(store_to_idx), "Num skus: ", len(sku_to_idx))
    
    pooled, scalers = build_all_series(df)
    for split_name in ["train", "val", "test"]:
        d = pooled[split_name]
        print(f"{split_name}: X={d['X'].shape} y={d['y'].shape}"
                f"store_idx range=({d['store_idx'].min()},{d['store_idx'].max()})"
                f"sku_idx range=({d['sku_idx'].min()},{d['sku_idx'].max()})"
              )
    
    print("Num series scaled:", len(scalers))

    train_ds = SalesDataset(pooled["train"]["X"], pooled["train"]["y"],
                            pooled["train"]["store_idx"], pooled["train"]["sku_idx"])
    x, s_idx, k_idx, y = train_ds[0]
    print("Sample: X", x.shape, "store_idx", s_idx.item(), "sku_idx", k_idx.item(), "y:", y.item())






    


