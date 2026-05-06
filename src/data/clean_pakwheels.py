from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"

INPUT_PATH = DATASET_DIR / "pakwheels_pakistan_automobile_dataset.csv"
OUTPUT_PATH = DATASET_DIR / "pakwheels_clean.csv"
SUMMARY_PATH = DATASET_DIR / "pakwheels_cleaning_summary.txt"


def extract_make(title: str) -> str:
    if pd.isna(title):
        return "Unknown"

    title = str(title).strip()
    if not title:
        return "Unknown"

    return title.split()[0]


def extract_model_name(title: str, make: str, year: int) -> str:
    if pd.isna(title):
        return "Unknown"

    text = str(title)

    # Remove make from beginning if present
    if text.lower().startswith(str(make).lower()):
        text = text[len(str(make)):].strip()

    # Remove year
    text = re.sub(rf"\b{year}\b", "", text).strip()

    return text if text else "Unknown"


def print_and_store(lines: list[str], text: str) -> None:
    print(text)
    lines.append(text)


def main() -> None:
    lines = []

    print_and_store(lines, "=" * 100)
    print_and_store(lines, "PAKWHEELS CLEANING AND LOCAL MARKET FEATURE ENGINEERING")
    print_and_store(lines, "=" * 100)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print_and_store(lines, f"Original shape: {df.shape}")
    print_and_store(lines, f"Original columns: {df.columns.tolist()}")

    df.columns = [col.strip() for col in df.columns]

    rename_map = {
        "title": "Title",
        "price": "Price",
        "city": "City",
        "model": "Year",
        "mileage": "Mileage",
        "fuel_type": "FuelType",
        "transmission": "Transmission",
        "registered": "Registered",
        "color": "Color",
        "assembly": "Assembly",
        "engine_capacity": "EngineCapacity",
        "post_date": "PostDate",
        "price_category": "PriceCategory",
        "mileage_category": "MileageCategory",
        "post_day_of_week": "PostDayOfWeek",
        "vehicle_age": "VehicleAgeProvided",
    }

    df = df.rename(columns=rename_map)

    numeric_cols = ["Price", "Year", "Mileage", "EngineCapacity", "VehicleAgeProvided"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    string_cols = [
        "Title",
        "City",
        "FuelType",
        "Transmission",
        "Registered",
        "Color",
        "Assembly",
        "PostDate",
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    before_missing = df.shape[0]
    df = df.dropna(subset=["Price", "Year", "Mileage", "Title", "City"])
    after_missing = df.shape[0]

    before_duplicates = df.shape[0]
    df = df.drop_duplicates()
    after_duplicates = df.shape[0]

    before_filters = df.shape[0]

    # Domain filters for Pakistan listings
    df = df[df["Price"] > 50000]
    df = df[df["Price"] < 100000000]
    df = df[df["Mileage"] >= 0]
    df = df[df["Mileage"] < 1000000]
    df = df[df["Year"] >= 1990]
    df = df[df["Year"] <= 2026]

    if "EngineCapacity" in df.columns:
        df = df[(df["EngineCapacity"] > 0) & (df["EngineCapacity"] < 10000)]

    after_filters = df.shape[0]

    # PakWheels posts are from 2024 in this dataset, so use post year where possible.
    df["PostDateParsed"] = pd.to_datetime(df["PostDate"], errors="coerce")
    reference_year = int(df["PostDateParsed"].dt.year.dropna().mode().iloc[0]) if df["PostDateParsed"].notna().any() else 2024

    df["VehicleAge"] = reference_year - df["Year"]
    df["VehicleAge"] = df["VehicleAge"].clip(lower=1)

    df["MileagePerYear"] = df["Mileage"] / df["VehicleAge"]
    df["LogPrice"] = np.log1p(df["Price"])
    df["LogMileage"] = np.log1p(df["Mileage"])

    df["Make"] = df["Title"].apply(extract_make)
    df["ModelName"] = df.apply(lambda row: extract_model_name(row["Title"], row["Make"], int(row["Year"])), axis=1)
    df["Market"] = "Pakistan"

    # Do not use PriceCategory as input later because it is directly derived from price.
    # MileageCategory is also derived from mileage, so it is mainly useful for EDA, not baseline modeling.

    output_cols = [
        "Market",
        "Title",
        "Price",
        "Year",
        "VehicleAge",
        "Mileage",
        "MileagePerYear",
        "LogPrice",
        "LogMileage",
        "City",
        "Make",
        "ModelName",
        "FuelType",
        "Transmission",
        "Registered",
        "Color",
        "Assembly",
        "EngineCapacity",
        "PostDate",
    ]

    df = df[output_cols].reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print_and_store(lines, f"Rows removed due to missing key values: {before_missing - after_missing}")
    print_and_store(lines, f"Duplicate rows removed: {before_duplicates - after_duplicates}")
    print_and_store(lines, f"Rows removed by domain filters: {before_filters - after_filters}")
    print_and_store(lines, f"Reference year used for PakWheels VehicleAge: {reference_year}")
    print_and_store(lines, f"Final cleaned shape: {df.shape}")
    print_and_store(lines, f"Saved cleaned PakWheels dataset to: {OUTPUT_PATH}")
    print_and_store(lines, "")
    print_and_store(lines, "Top makes:")
    print_and_store(lines, str(df["Make"].value_counts().head(15)))
    print_and_store(lines, "")
    print_and_store(lines, "Numeric summary:")
    print_and_store(lines, str(df[["Price", "Year", "Mileage", "VehicleAge", "MileagePerYear", "EngineCapacity"]].describe()))

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCleaning summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
