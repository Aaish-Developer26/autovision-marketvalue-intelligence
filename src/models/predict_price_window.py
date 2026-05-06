from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

INPUT_PATH = DATASET_DIR / "truecar_clean.csv"
MODEL_PATH = MODELS_DIR / "best_advanced_model.joblib"
OUTPUT_PATH = REPORTS_DIR / "price_window_sample_prediction.csv"

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


def build_price_window(
    predicted_price: float,
    lower_error: float,
    upper_error: float
) -> dict:
    lower_bound = max(0, predicted_price + lower_error)
    upper_bound = max(0, predicted_price + upper_error)

    return {
        "PredictedFairValue": round(predicted_price, 2),
        "LowerPriceBound": round(lower_bound, 2),
        "UpperPriceBound": round(upper_bound, 2),
        "PricingWindow": f"{round(lower_bound, 2)} - {round(upper_bound, 2)}"
    }


def main() -> None:
    print("=" * 100)
    print("PRICE WINDOW PREDICTION")
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

    test_predictions = model.predict(X_test)
    residuals = y_test.values - test_predictions

    lower_error = np.quantile(residuals, 0.05)
    upper_error = np.quantile(residuals, 0.95)

    print(f"Lower residual bound 5% : {lower_error:.2f}")
    print(f"Upper residual bound 95%: {upper_error:.2f}")

    sample_car = pd.DataFrame([
        {
            "Year": 2017,
            "Mileage": 35000,
            "VehicleAge": 2,
            "MileagePerYear": 17500,
            "LogMileage": np.log1p(35000),
            "City": "Houston",
            "State": "TX",
            "Make": "Toyota",
            "Model": "Camry",
            "MakeModel": "Toyota_Camry",
        }
    ])

    predicted_price = model.predict(sample_car)[0]

    window = build_price_window(
        predicted_price=predicted_price,
        lower_error=lower_error,
        upper_error=upper_error
    )

    result = sample_car.copy()
    result["PredictedFairValue"] = window["PredictedFairValue"]
    result["LowerPriceBound"] = window["LowerPriceBound"]
    result["UpperPriceBound"] = window["UpperPriceBound"]
    result["PricingWindow"] = window["PricingWindow"]

    result.to_csv(OUTPUT_PATH, index=False)

    print("\nSample car prediction with pricing window:")
    print(result)

    print(f"\nSaved price window result to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()