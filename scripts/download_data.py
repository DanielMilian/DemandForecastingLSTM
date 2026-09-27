import kagglehub
from kagglehub import KaggleDatasetAdapter

def download():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "anirudhchauhan/retail-store-inventory-forecasting-dataset",
        "retail_store_inventory.csv",
    )
    df.to_csv("data/retail_store_inventory.csv", index=False)
    print(f"Saved {df.shape[0]} rows to data/retail_store_inventory.csv")

if __name__ == "__main__":
    download()
