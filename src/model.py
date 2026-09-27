import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_feature)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        pred = self.head(last_hidden)
        return pred.squeeze(-1)

if __name__ == "__main__":
    from dataset import (
            load_raw, add_features, get_series, chronological_split, fit_scaler, make_windows, SalesDataset, FEATURE_COLS,
            )
    from torch.utils.data import DataLoader

    df = add_features(load_raw())
    series = get_series(df, store_id=9823, sku_id=673209)
    train, val, test = chronological_split(series)
    scaler = fit_scaler(train)
    X_train, y_train = make_windows(train, scaler)

    train_ds = SalesDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = LSTMForecaster(n_features=len(FEATURE_COLS))
    xb, yb = next(iter(loader))
    print("Batch X shape:", xb.shape, "Batch y shape:", yb.shape)

    preds = model(xb)
    print("Prediction shape:", preds.shape)
    print("Sample predictions:", preds[:3].detach().numpy())




