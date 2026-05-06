from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

INPUT_PATH = DATASET_DIR / "truecar_clean.csv"
RESULTS_PATH = REPORTS_DIR / "model_results_advanced.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_advanced_model.joblib"

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

NUMERIC_FEATURES = [
    "Year",
    "Mileage",
    "VehicleAge",
    "MileagePerYear",
    "LogMileage",
]

CATEGORICAL_FEATURES = [
    "City",
    "State",
    "Make",
    "Model",
    "MakeModel",
]


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(model_name: str, model, X_train, X_test, y_train, y_test) -> tuple[dict, object]:
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test)

    result = {
        "model": model_name,
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(rmse_score(y_test, preds), 4),
        "R2": round(r2_score(y_test, preds), 4),
        "train_time_seconds": round(train_time, 2),
    }

    return result, model


def build_preprocessor() -> ColumnTransformer:
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_preprocessor, NUMERIC_FEATURES),
            ("cat", categorical_preprocessor, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )


def main() -> None:
    print("=" * 100)
    print("ADVANCED REGRESSION MODEL TRAINING")
    print("=" * 100)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = df[FEATURES + [TARGET]].dropna().copy()

    # Keep this stable for laptop training. Increase later if machine handles it.
    sample_size = min(500000, len(df))
    df_model = df.sample(sample_size, random_state=42).reset_index(drop=True)

    print(f"Loaded dataset: {INPUT_PATH}")
    print(f"Full shape: {df.shape}")
    print(f"Modeling sample size: {df_model.shape[0]}")

    X = df_model[FEATURES]
    y = df_model[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = build_preprocessor()

    models = {
        "Hist Gradient Boosting Tuned": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=45,
            l2_regularization=0.05,
            random_state=42
        ),
        "Random Forest Tuned": RandomForestRegressor(
            n_estimators=180,
            max_depth=22,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=180,
            max_depth=24,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
    }

    # Optional LightGBM, if installed and working.
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.045,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            objective="regression"
        )
    except Exception as exc:
        print(f"[WARN] LightGBM unavailable, skipping. Reason: {exc}")

    results = []
    trained_pipelines = {}

    for model_name, regressor in models.items():
        print(f"\nTraining: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", regressor),
            ]
        )

        result, trained_model = evaluate_model(
            model_name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)
        trained_pipelines[model_name] = trained_model

        print(json.dumps(result, indent=2))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(RESULTS_PATH, index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_pipelines[best_model_name]
    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\n" + "=" * 100)
    print("ADVANCED MODEL COMPARISON")
    print("=" * 100)
    print(results_df)
    print(f"\nSaved results to: {RESULTS_PATH}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
