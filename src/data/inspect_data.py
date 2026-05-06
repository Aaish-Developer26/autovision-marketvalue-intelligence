from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"

FILES = [
    "true_car_listing_01.csv",
    "true_car_listings_02.csv",
    "pakwheels_pakistan_automobile_dataset.csv",
]

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

def describe_df(name: str, df: pd.DataFrame) -> None:
    print("=" * 100)
    print(f"FILE: {name}")
    print(f"SHAPE: {df.shape}")

    print("\nCOLUMNS:")
    print(df.columns.tolist())

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nTOP MISSING VALUES:")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nDUPLICATE ROWS:", df.duplicated().sum())

    if "Vin" in df.columns:
        print("DUPLICATE VINs:", df.duplicated(subset=["Vin"]).sum())
        print("UNIQUE VINs:", df["Vin"].nunique())

    print("\nFIRST 3 ROWS:")
    print(df.head(3))
    print()

def main() -> None:
    for filename in FILES:
        path = DATASET_DIR / filename

        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue

        df = safe_read_csv(path)
        describe_df(filename, df)

if __name__ == "__main__":
    main()