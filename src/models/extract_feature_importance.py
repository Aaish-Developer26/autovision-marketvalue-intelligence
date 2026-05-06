from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "best_advanced_model.joblib"
OUTPUT_CSV = REPORTS_DIR / "feature_importance.csv"
OUTPUT_FIGURE = FIGURE_DIR / "15_feature_importance.png"

FEATURES = [
    "Year",
    "Mileage",
    "VehicleAge",
    "MileagePerYear",
    "LogMileage",
    "City",
    "State",
    "Make",
    "Model",
    "MakeModel",
]


def main() -> None:
    print("=" * 100)
    print("FEATURE IMPORTANCE EXTRACTION")
    print("=" * 100)

    model_pipeline = joblib.load(MODEL_PATH)
    model = model_pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Best model does not expose feature_importances_.")

    importances = model.feature_importances_

    if len(importances) != len(FEATURES):
        raise ValueError(
            f"Feature count mismatch. Expected {len(FEATURES)} but model returned {len(importances)}."
        )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(OUTPUT_CSV, index=False)

    print(importance_df)
    print(f"Saved feature importance CSV to: {OUTPUT_CSV}")

    top_df = importance_df.head(10).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.title("Top Feature Importances for Used Car Price Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved feature importance figure to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
