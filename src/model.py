import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, n_stores: int, n_skus:int,
                 hidden_size: int = 32, store_emb_dim: int=4, sku_emb_dim: int = 8,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                )
        self.store_emb = nn.Embedding(n_stores, store_emb_dim)
        self.sku_emb = nn.Embedding(n_skus, sku_emb_dim)

        combined_dim = hidden_size + store_emb_dim + sku_emb_dim
        # DROPOUT LAYER
        # self.dropout = nn.Dropout(dropout) #Tried as a regularization experiment
        self.head = nn.Linear(combined_dim, 1)
        
    def forward(self, x: torch.Tensor, store_idx: torch.Tensor, sku_idx: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_feature)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        
        store_vec = self.store_emb(store_idx)
        sku_vec = self.sku_emb(sku_idx)
        
        combined = torch.cat([last_hidden, store_vec, sku_vec], dim=1)
        #DROPOUTlayer
        # combined = self.dropout(combined) #Tried as a regularization experiment

        pred = self.head(combined)
        return pred.squeeze(-1)

if __name__ == "__main__":
    from dataset import (
            load_raw, add_features, encode_ids, build_all_series, SalesDataset, FEATURE_COLS,
            )
    from torch.utils.data import DataLoader

    df = add_features(load_raw())
    df, store_to_idx, sku_to_idx = encode_ids(df)
    pooled, scalers = build_all_series(df)
  
    train_ds = SalesDataset(pooled["train"]["X"], pooled["train"]["y"],
                            pooled["train"]["store_idx"], pooled["train"]["sku_idx"])
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = LSTMForecaster(n_features=len(FEATURE_COLS),
                           n_stores=len(store_to_idx),
                           n_skus=len(sku_to_idx),
                           )
    xb, store_idx_b, sku_idx_b, yb = next(iter(loader))
    print("X batch:", xb.shape, "store_idx batch:", store_idx_b.shape, "sku_idx batch:", sku_idx_b.shape)

    preds = model(xb, store_idx_b, sku_idx_b)
    print("Prediction shape:", preds.shape)
    print("Sample predictions:", preds[:3].detach().numpy())




