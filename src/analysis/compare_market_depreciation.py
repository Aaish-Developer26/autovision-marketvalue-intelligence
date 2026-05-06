from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
REPORTS_DIR = ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRUECAR_PATH = DATASET_DIR / "truecar_clean.csv"
PAKWHEELS_PATH = DATASET_DIR / "pakwheels_clean.csv"

COMBINED_OUTPUT = REPORTS_DIR / "us_pak_market_comparison_dataset.csv"
AGE_SUMMARY_OUTPUT = REPORTS_DIR / "depreciation_summary_by_age.csv"
MAKE_SUMMARY_OUTPUT = REPORTS_DIR / "common_make_depreciation_summary.csv"


def save_plot(filename: str) -> None:
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def prepare_truecar(df: pd.DataFrame) -> pd.DataFrame:
    use_cols = [
        "Price",
        "Year",
        "VehicleAge",
        "Mileage",
        "MileagePerYear",
        "LogMileage",
        "City",
        "State",
        "Make",
        "Model",
    ]

    df = df[use_cols].copy()
    df["Market"] = "US"
    df["Currency"] = "USD"
    df = df.rename(columns={"Model": "ModelName"})
    return df


def prepare_pakwheels(df: pd.DataFrame) -> pd.DataFrame:
    use_cols = [
        "Price",
        "Year",
        "VehicleAge",
        "Mileage",
        "MileagePerYear",
        "LogMileage",
        "City",
        "Make",
        "ModelName",
    ]

    df = df[use_cols].copy()
    df["Market"] = "Pakistan"
    df["Currency"] = "PKR"
    df["State"] = "N/A"
    return df


def main() -> None:
    print("=" * 100)
    print("CROSS-MARKET DEPRECIATION COMPARISON: US TRUECAR VS PAKISTAN PAKWHEELS")
    print("=" * 100)

    truecar = pd.read_csv(TRUECAR_PATH, low_memory=False)
    pakwheels = pd.read_csv(PAKWHEELS_PATH, low_memory=False)

    us = prepare_truecar(truecar)
    pk = prepare_pakwheels(pakwheels)

    # Keep US data sampled for balanced visualization/report generation
    us_sample = us.sample(min(250000, len(us)), random_state=42)
    combined = pd.concat([us_sample, pk], ignore_index=True)

    # Normalize prices within each market because currencies and purchasing power differ.
    market_median_price = combined.groupby("Market")["Price"].transform("median")
    combined["PriceIndex"] = combined["Price"] / market_median_price

    # Normalize within market + make for make-specific depreciation comparison.
    combined["Make"] = combined["Make"].astype(str).str.title().str.strip()
    market_make_median = combined.groupby(["Market", "Make"])["Price"].transform("median")
    combined["MakePriceIndex"] = combined["Price"] / market_make_median

    combined.to_csv(COMBINED_OUTPUT, index=False)
    print(f"Saved combined comparison dataset to: {COMBINED_OUTPUT}")
    print(f"Combined shape: {combined.shape}")

    age_summary = (
        combined
        .groupby(["Market", "VehicleAge"], as_index=False)
        .agg(
            median_price=("Price", "median"),
            median_price_index=("PriceIndex", "median"),
            median_make_price_index=("MakePriceIndex", "median"),
            median_mileage=("Mileage", "median"),
            listing_count=("Price", "size"),
        )
        .sort_values(["Market", "VehicleAge"])
    )

    age_summary.to_csv(AGE_SUMMARY_OUTPUT, index=False)
    print(f"Saved age summary to: {AGE_SUMMARY_OUTPUT}")

    # Find common makes in both markets
    us_makes = set(combined[combined["Market"] == "US"]["Make"].unique())
    pk_makes = set(combined[combined["Market"] == "Pakistan"]["Make"].unique())
    common_makes = sorted(list(us_makes.intersection(pk_makes)))

    common_df = combined[combined["Make"].isin(common_makes)].copy()

    make_summary = (
        common_df
        .groupby(["Market", "Make", "VehicleAge"], as_index=False)
        .agg(
            median_make_price_index=("MakePriceIndex", "median"),
            listing_count=("Price", "size"),
        )
        .sort_values(["Make", "Market", "VehicleAge"])
    )

    make_summary.to_csv(MAKE_SUMMARY_OUTPUT, index=False)
    print(f"Saved common make summary to: {MAKE_SUMMARY_OUTPUT}")
    print(f"Common makes found: {common_makes[:20]}")

    # Figure 18: Market-level depreciation by vehicle age
    plot_age = age_summary[
        (age_summary["VehicleAge"] <= 25) &
        (age_summary["listing_count"] >= 30)
    ]

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=plot_age,
        x="VehicleAge",
        y="median_price_index",
        hue="Market",
        marker="o"
    )
    plt.title("Normalized Price Index by Vehicle Age: US vs Pakistan")
    plt.xlabel("Vehicle Age")
    plt.ylabel("Median Price Index")
    save_plot("18_price_index_by_vehicle_age_market.png")

    # Figure 19: Mileage per year comparison
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=combined[combined["MileagePerYear"] < combined["MileagePerYear"].quantile(0.98)],
        x="Market",
        y="MileagePerYear"
    )
    plt.title("Mileage Per Year Comparison: US vs Pakistan")
    plt.xlabel("Market")
    plt.ylabel("Mileage Per Year")
    save_plot("19_mileage_per_year_by_market.png")

    # Figure 20: Common make depreciation curves
    common_make_counts = (
        common_df.groupby("Make")["Price"]
        .size()
        .sort_values(ascending=False)
        .head(6)
        .index
    )

    plot_common = make_summary[
        (make_summary["Make"].isin(common_make_counts)) &
        (make_summary["VehicleAge"] <= 20) &
        (make_summary["listing_count"] >= 10)
    ]

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=plot_common,
        x="VehicleAge",
        y="median_make_price_index",
        hue="Make",
        style="Market",
        marker="o"
    )
    plt.title("Make-Normalized Depreciation for Common Makes")
    plt.xlabel("Vehicle Age")
    plt.ylabel("Median Make Price Index")
    save_plot("20_common_make_depreciation_curves.png")

    # Figure 21: Market summary statistics
    market_summary = (
        combined
        .groupby("Market", as_index=False)
        .agg(
            median_vehicle_age=("VehicleAge", "median"),
            median_mileage=("Mileage", "median"),
            median_mileage_per_year=("MileagePerYear", "median"),
            records=("Price", "size"),
        )
    )

    market_summary.to_csv(REPORTS_DIR / "market_summary_statistics.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=market_summary, x="Market", y="median_vehicle_age")
    plt.title("Median Vehicle Age by Market")
    plt.xlabel("Market")
    plt.ylabel("Median Vehicle Age")
    save_plot("21_median_vehicle_age_by_market.png")

    print("\nMarket summary:")
    print(market_summary)
    print("\nComparison figures generated successfully.")


if __name__ == "__main__":
    main()
