from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"

INPUT_PATH = REPORTS_DIR / "us_pak_market_comparison_dataset.csv"
OUTPUT_PATH = REPORTS_DIR / "depreciation_hypothesis_summary.txt"


def fit_depreciation_model(df: pd.DataFrame, market: str) -> dict:
    data = df[df["Market"] == market].copy()

    # Remove extreme normalized prices for a stable simple comparison
    data = data[
        (data["PriceIndex"] > data["PriceIndex"].quantile(0.01)) &
        (data["PriceIndex"] < data["PriceIndex"].quantile(0.99))
    ]

    # Simple interpretable depreciation model:
    # log(PriceIndex) = beta0 + beta1*VehicleAge + beta2*LogMileage
    X = data[["VehicleAge", "LogMileage"]].copy()
    y = np.log(data["PriceIndex"])

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    r2 = r2_score(y, preds)

    age_coef = model.coef_[0]
    mileage_coef = model.coef_[1]

    # Convert log-linear coefficient to approximate percentage change
    yearly_depreciation_pct = (math.exp(age_coef) - 1) * 100
    mileage_effect_pct = (math.exp(mileage_coef) - 1) * 100

    return {
        "market": market,
        "rows": len(data),
        "age_coefficient": age_coef,
        "yearly_depreciation_pct": yearly_depreciation_pct,
        "log_mileage_coefficient": mileage_coef,
        "log_mileage_effect_pct": mileage_effect_pct,
        "r2": r2,
    }


def main() -> None:
    print("=" * 100)
    print("DEPRECIATION HYPOTHESIS TEST: US VS PAKISTAN")
    print("=" * 100)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    results = [
        fit_depreciation_model(df, "US"),
        fit_depreciation_model(df, "Pakistan"),
    ]

    us = results[0]
    pk = results[1]

    summary = f"""Depreciation Hypothesis Summary

Hypothesis:
Do used cars in Pakistan depreciate at the same rate as used cars in the US?

Important note:
Raw prices are not directly comparable because TrueCar uses USD while PakWheels uses PKR. Therefore, this comparison uses a normalized Price Index within each market.

Method:
A simple interpretable regression model was fitted separately for each market:

log(PriceIndex) = beta0 + beta1*VehicleAge + beta2*LogMileage

Results:

US:
- Rows used: {us['rows']}
- VehicleAge coefficient: {us['age_coefficient']:.6f}
- Approx yearly depreciation effect: {us['yearly_depreciation_pct']:.2f}%
- LogMileage coefficient: {us['log_mileage_coefficient']:.6f}
- R2: {us['r2']:.4f}

Pakistan:
- Rows used: {pk['rows']}
- VehicleAge coefficient: {pk['age_coefficient']:.6f}
- Approx yearly depreciation effect: {pk['yearly_depreciation_pct']:.2f}%
- LogMileage coefficient: {pk['log_mileage_coefficient']:.6f}
- R2: {pk['r2']:.4f}

Interpretation:
If the yearly depreciation percentages are meaningfully different, then the two markets do not depreciate at the same rate. Differences can be caused by market demand, import duties, brand availability, local preferences, spare-parts availability, fuel economy concerns, and macroeconomic conditions.

Conclusion:
The comparison should be presented as an exploratory market analysis rather than a strict causal claim. It adds strong local relevance to the project while keeping TrueCar as the main large-scale modeling dataset.
"""

    print(summary)
    OUTPUT_PATH.write_text(summary, encoding="utf-8")
    print(f"Saved hypothesis summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
