from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"

INPUT_PATH = DATASET_DIR / "truecar_merged.csv"
OUTPUT_PATH = DATASET_DIR / "truecar_clean.csv"
SUMMARY_PATH = DATASET_DIR / "truecar_cleaning_summary.txt"


def print_and_store(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


def main() -> None:
    lines = []
    print_and_store(lines, "=" * 100)
    print_and_store(lines, "TRUECAR CLEANING AND FEATURE ENGINEERING - V2")
    print_and_store(lines, "=" * 100)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print_and_store(lines, f"Original shape: {df.shape}")
    print_and_store(lines, f"Original columns: {df.columns.tolist()}")

    df.columns = [col.strip() for col in df.columns]

    string_cols = ["City", "State", "Vin", "Make", "Model"]
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()

    numeric_cols = ["Price", "Year", "Mileage"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_missing = df.shape[0]
    df = df.dropna(subset=["Price", "Year", "Mileage", "Vin", "Make", "Model"])
    after_missing = df.shape[0]

    before_exact_duplicates = df.shape[0]
    df = df.drop_duplicates()
    after_exact_duplicates = df.shape[0]

    before_vin_duplicates = df.shape[0]
    df = df.drop_duplicates(subset=["Vin"], keep="first")
    after_vin_duplicates = df.shape[0]

    before_filters = df.shape[0]

    df = df[df["Price"] > 500]
    df = df[df["Price"] < 250000]
    df = df[df["Mileage"] >= 0]
    df = df[df["Mileage"] < 500000]
    df = df[df["Year"] >= 1990]
    df = df[df["Year"] <= 2026]

    after_filters = df.shape[0]

    # IMPORTANT:
    # The dataset's latest model year is used as the dataset reference point.
    # We add 1 so the newest cars get age 1 instead of 0, which makes MileagePerYear stable.
    dataset_reference_year = int(df["Year"].max()) + 1

    df["VehicleAge"] = dataset_reference_year - df["Year"]
    df["VehicleAge"] = df["VehicleAge"].clip(lower=1)

    df["MileagePerYear"] = df["Mileage"] / df["VehicleAge"]
    df["LogPrice"] = np.log1p(df["Price"])
    df["LogMileage"] = np.log1p(df["Mileage"])

    df["MakeModel"] = df["Make"] + "_" + df["Model"]

    df["AgeBucket"] = pd.cut(
        df["VehicleAge"],
        bins=[0, 2, 5, 10, 20, 100],
        labels=["0-2 years", "3-5 years", "6-10 years", "11-20 years", "20+ years"],
        include_lowest=True
    )

    df["MileageBucket"] = pd.cut(
        df["Mileage"],
        bins=[-1, 25000, 60000, 100000, 150000, 500000],
        labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )

    df = df.sort_values(["Make", "Model", "Year"]).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print_and_store(lines, f"Rows removed due to missing key values: {before_missing - after_missing}")
    print_and_store(lines, f"Exact duplicate rows removed: {before_exact_duplicates - after_exact_duplicates}")
    print_and_store(lines, f"Duplicate VIN rows removed after merge: {before_vin_duplicates - after_vin_duplicates}")
    print_and_store(lines, f"Rows removed by domain filters: {before_filters - after_filters}")
    print_and_store(lines, f"Dataset reference year used for VehicleAge: {dataset_reference_year}")
    print_and_store(lines, f"Final cleaned shape: {df.shape}")
    print_and_store(lines, f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print_and_store(lines, "")
    print_and_store(lines, "Engineered features added:")
    print_and_store(lines, "- VehicleAge")
    print_and_store(lines, "- MileagePerYear")
    print_and_store(lines, "- LogPrice")
    print_and_store(lines, "- LogMileage")
    print_and_store(lines, "- MakeModel")
    print_and_store(lines, "- AgeBucket")
    print_and_store(lines, "- MileageBucket")
    print_and_store(lines, "")
    print_and_store(lines, "Final missing values:")
    print_and_store(lines, str(df.isna().sum().sort_values(ascending=False).head(15)))
    print_and_store(lines, "")
    print_and_store(lines, "Numeric summary:")
    print_and_store(lines, str(df[["Price", "Year", "Mileage", "VehicleAge", "MileagePerYear"]].describe()))

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCleaning summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()