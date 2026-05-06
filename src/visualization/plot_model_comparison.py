from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_RESULTS = REPORTS_DIR / "model_results_baseline.csv"
ADVANCED_RESULTS = REPORTS_DIR / "model_results_advanced.csv"
COMBINED_RESULTS = REPORTS_DIR / "model_results_combined.csv"
OUTPUT_FIGURE_RMSE = FIGURE_DIR / "16_model_comparison_rmse.png"
OUTPUT_FIGURE_R2 = FIGURE_DIR / "17_model_comparison_r2.png"


def main() -> None:
    print("=" * 100)
    print("MODEL COMPARISON VISUALIZATION")
    print("=" * 100)

    baseline_df = pd.read_csv(BASELINE_RESULTS)
    advanced_df = pd.read_csv(ADVANCED_RESULTS)

    baseline_df["stage"] = "Baseline"
    advanced_df["stage"] = "Advanced"

    combined = pd.concat([baseline_df, advanced_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["model"], keep="last")
    combined = combined.sort_values("RMSE")

    combined.to_csv(COMBINED_RESULTS, index=False)

    print(combined)
    print(f"Saved combined results to: {COMBINED_RESULTS}")

    rmse_df = combined.sort_values("RMSE", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(rmse_df["model"], rmse_df["RMSE"])
    plt.title("Model Comparison by RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_RMSE, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved RMSE comparison to: {OUTPUT_FIGURE_RMSE}")

    r2_df = combined.sort_values("R2", ascending=True)
    plt.figure(figsize=(11, 6))
    plt.barh(r2_df["model"], r2_df["R2"])
    plt.title("Model Comparison by R² Score")
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_R2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved R2 comparison to: {OUTPUT_FIGURE_R2}")


if __name__ == "__main__":
    main()
