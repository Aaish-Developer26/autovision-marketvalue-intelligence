from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
OUTPUT_PATH = DATASET_DIR / "truecar_merged.csv"

def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        print(f"[WARN] Parser error found in {path.name}. Retrying with bad-line skipping...")
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip"
        )

def main() -> None:
    file_1 = DATASET_DIR / "true_car_listing_01.csv"
    file_2 = DATASET_DIR / "true_car_listings_02.csv"

    df1 = safe_read_csv(file_1)
    df2 = safe_read_csv(file_2)

    df1.columns = [col.strip() for col in df1.columns]
    df2.columns = [col.strip() for col in df2.columns]

    if "Id" in df1.columns:
        df1 = df1.drop(columns=["Id"])

    df1["State"] = df1["State"].astype(str).str.strip()
    df2["State"] = df2["State"].astype(str).str.strip()

    merged = pd.concat([df1, df2], ignore_index=True)

    before = merged.shape[0]
    merged = merged.drop_duplicates(subset=["Vin"], keep="first")
    after = merged.shape[0]

    merged.to_csv(OUTPUT_PATH, index=False)

    print("=" * 100)
    print(f"Saved merged dataset to: {OUTPUT_PATH}")
    print(f"Rows before VIN deduplication: {before}")
    print(f"Rows after VIN deduplication : {after}")
    print(f"Removed duplicate VIN rows   : {before - after}")
    print(f"Final columns                : {merged.columns.tolist()}")

if __name__ == "__main__":
    main()