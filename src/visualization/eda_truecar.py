from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
FIGURE_DIR = ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATASET_DIR / "truecar_clean.csv"


def save_plot(filename: str) -> None:
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("=" * 100)
    print("TRUECAR EDA VISUALIZATION GENERATION")
    print("=" * 100)
    print(f"Dataset shape: {df.shape}")

    sample_size = min(250000, len(df))
    sample = df.sample(sample_size, random_state=42)

    plt.figure(figsize=(10, 6))
    sns.histplot(sample["Price"], bins=80, kde=True)
    plt.title("Distribution of Used Car Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    save_plot("01_price_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.histplot(sample["LogPrice"], bins=80, kde=True)
    plt.title("Distribution of Log-Transformed Prices")
    plt.xlabel("Log(Price + 1)")
    plt.ylabel("Frequency")
    save_plot("02_log_price_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sample, x="Mileage", y="Price", alpha=0.25, s=8)
    plt.title("Mileage vs Price")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    save_plot("03_mileage_vs_price.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=sample[sample["VehicleAge"] <= 25], x="VehicleAge", y="Price")
    plt.title("Vehicle Age vs Price")
    plt.xlabel("Vehicle Age")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    save_plot("04_vehicle_age_vs_price.png")

    top_makes = df["Make"].value_counts().head(15)
    plt.figure(figsize=(11, 6))
    sns.barplot(x=top_makes.values, y=top_makes.index)
    plt.title("Top 15 Makes by Number of Listings")
    plt.xlabel("Number of Listings")
    plt.ylabel("Make")
    save_plot("05_top_makes_by_count.png")

    top_make_names = top_makes.index.tolist()
    make_price = (
        df[df["Make"].isin(top_make_names)]
        .groupby("Make")["Price"]
        .median()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(x=make_price.values, y=make_price.index)
    plt.title("Median Price by Top Listed Makes")
    plt.xlabel("Median Price")
    plt.ylabel("Make")
    save_plot("06_median_price_by_make.png")

    top_states = df["State"].value_counts().head(20).index
    state_price = (
        df[df["State"].isin(top_states)]
        .groupby("State")["Price"]
        .median()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(x=state_price.values, y=state_price.index)
    plt.title("Median Price by State for Top Listing States")
    plt.xlabel("Median Price")
    plt.ylabel("State")
    save_plot("07_median_price_by_state.png")

    numeric_cols = ["Price", "Year", "Mileage", "VehicleAge", "MileagePerYear", "LogPrice", "LogMileage"]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Numeric Features")
    save_plot("08_numeric_correlation_heatmap.png")

    mileage_bucket_price = df.groupby("MileageBucket", observed=True)["Price"].median()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=mileage_bucket_price.index.astype(str), y=mileage_bucket_price.values)
    plt.title("Median Price by Mileage Bucket")
    plt.xlabel("Mileage Bucket")
    plt.ylabel("Median Price")
    plt.xticks(rotation=30)
    save_plot("09_median_price_by_mileage_bucket.png")

    age_bucket_price = df.groupby("AgeBucket", observed=True)["Price"].median()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=age_bucket_price.index.astype(str), y=age_bucket_price.values)
    plt.title("Median Price by Vehicle Age Bucket")
    plt.xlabel("Age Bucket")
    plt.ylabel("Median Price")
    plt.xticks(rotation=30)
    save_plot("10_median_price_by_age_bucket.png")

    print("\nEDA figures generated successfully.")


if __name__ == "__main__":
    main()