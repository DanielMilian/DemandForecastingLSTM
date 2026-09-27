import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/train.csv"
OUT_DIR = Path("reports/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["week"] = pd.to_datetime(df["week"], format="%d/%m/%y")
    df["total_price"] = df["total_price"].fillna(df["base_price"])
    return df.sort_values(["store_id", "sku_id", "week"])

def summary(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Data range:", df["week"].min(), "->", df["week"].max())
    print("Weeks:", df["week"].nunique())
    print("Series (store x sku):", df.groupby(["store_id", "sku_id"]).ngroups)
    print("\nUnits Sold stats:")
    print(df["units_sold"].describe()) 
    print("\nCorrelation with units_sold:")
    cols = ["units_sold", "total_price", "base_price", "is_featured_sku", "is_display_sku"]
    print(df[cols].corr()["units_sold"].sort_values(ascending=False))

def plot_sample_series(df: pd.DataFrame, n: int = 4) -> None:
    combos = df[["store_id", "sku_id"]].drop_duplicates().sample(n, random_state=42)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    for ax, (_, row) in zip(axes, combos.iterrows()):
        s = df[(df["store_id"] == row["store_id"]) & (df["sku_id"] == row["sku_id"])]
        ax.plot(s["week"], s["units_sold"])
        ax.set_title(f"store {row['store_id']} / sku {row['sku_id']}")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "sample_series.png")
    print(f"Saved {OUT_DIR / 'sample_series.png'}")

def plot_seasonal_pattern(df: pd.DataFrame) -> None:
    df = df.copy()
    df["month"] = df["week"].dt.month
    means = df.groupby("month")["units_sold"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg Units Sold")
    ax.set_xlabel("Month")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "monthly_seasonality.png")
    print(f"Saved {OUT_DIR / 'monthly_seasonality.png'}")

if __name__ == "__main__":
    df = load()
    summary(df)
    plot_sample_series(df)
    plot_seasonal_pattern(df)


