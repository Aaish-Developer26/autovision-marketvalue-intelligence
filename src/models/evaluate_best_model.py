from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
REPORTS_DIR = ROOT / "reports"
FIGURE_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATASET_DIR / "truecar_clean.csv"
MODEL_PATH = MODELS_DIR / "best_advanced_model.joblib"
REPORT_PATH = REPORTS_DIR / "best_model_evaluation.json"
PREDICTIONS_PATH = REPORTS_DIR / "sample_predictions.csv"

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

TARGET = "Price"


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def save_plot(filename: str) -> None:
    output_path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    print("=" * 100)
    print("BEST MODEL EVALUATION AND DIAGNOSTICS")
    print("=" * 100)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df[FEATURES + [TARGET]].dropna().copy()

    sample_size = min(500000, len(df))
    df_model = df.sample(sample_size, random_state=42).reset_index(drop=True)

    X = df_model[FEATURES]
    y = df_model[TARGET]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)
    residuals = y_test - preds

    metrics = {
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(rmse_score(y_test, preds), 4),
        "R2": round(r2_score(y_test, preds), 4),
        "test_rows": int(len(y_test)),
    }

    print(json.dumps(metrics, indent=2))
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    prediction_df = X_test.copy()
    prediction_df["ActualPrice"] = y_test.values
    prediction_df["PredictedPrice"] = preds
    prediction_df["AbsoluteError"] = np.abs(prediction_df["ActualPrice"] - prediction_df["PredictedPrice"])
    prediction_df = prediction_df.sort_values("AbsoluteError").head(1000)
    prediction_df.to_csv(PREDICTIONS_PATH, index=False)

    plot_sample = pd.DataFrame({
        "ActualPrice": y_test.values,
        "PredictedPrice": preds,
        "Residual": residuals.values
    }).sample(min(50000, len(y_test)), random_state=42)

    plt.figure(figsize=(8, 8))
    plt.scatter(plot_sample["ActualPrice"], plot_sample["PredictedPrice"], alpha=0.25, s=8)
    min_value = min(plot_sample["ActualPrice"].min(), plot_sample["PredictedPrice"].min())
    max_value = max(plot_sample["ActualPrice"].max(), plot_sample["PredictedPrice"].max())
    plt.plot([min_value, max_value], [min_value, max_value])
    plt.title("Actual vs Predicted Used Car Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    save_plot("11_actual_vs_predicted.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_sample["PredictedPrice"], plot_sample["Residual"], alpha=0.25, s=8)
    plt.axhline(0)
    plt.title("Residual Plot")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    save_plot("12_residual_plot.png")

    plt.figure(figsize=(10, 6))
    plt.hist(plot_sample["Residual"], bins=80)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    save_plot("13_residual_distribution.png")

    print(f"Saved evaluation report to: {REPORT_PATH}")
    print(f"Saved sample predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
