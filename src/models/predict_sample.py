from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "best_baseline_model.joblib"

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
    model = joblib.load(MODEL_PATH)

    examples = pd.DataFrame([
        {
            "Year": 2017,
            "Mileage": 35000,
            "VehicleAge": 2,
            "MileagePerYear": 17500,
            "LogMileage": 10.463132,
            "City": "Houston",
            "State": "TX",
            "Make": "Toyota",
            "Model": "Camry",
            "MakeModel": "Toyota_Camry",
        },
        {
            "Year": 2015,
            "Mileage": 75000,
            "VehicleAge": 4,
            "MileagePerYear": 18750,
            "LogMileage": 11.225257,
            "City": "Los Angeles",
            "State": "CA",
            "Make": "Honda",
            "Model": "Civic",
            "MakeModel": "Honda_Civic",
        },
    ])

    predictions = model.predict(examples)
    examples["PredictedPrice"] = predictions.round(2)

    print("=" * 100)
    print("SAMPLE PRICE PREDICTIONS")
    print("=" * 100)
    print(examples)


if __name__ == "__main__":
    main()