from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "auto_generated_report_sections.md"

ADVANCED_RESULTS = REPORTS_DIR / "model_results_advanced.csv"
BEST_EVAL = REPORTS_DIR / "best_model_evaluation.json"
PRICE_POSITIONING = REPORTS_DIR / "price_positioning_classification_report.txt"
FEATURE_IMPORTANCE = REPORTS_DIR / "feature_importance.csv"


def main() -> None:
    advanced_df = pd.read_csv(ADVANCED_RESULTS)
    best = advanced_df.sort_values("RMSE").iloc[0].to_dict()

    with open(BEST_EVAL, "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)

    feature_df = pd.read_csv(FEATURE_IMPORTANCE)
    top_features = feature_df.head(5)

    price_positioning_text = PRICE_POSITIONING.read_text(encoding="utf-8")

    markdown = f"""# Auto-Generated Report Sections

## Model Training Summary

Several regression models were trained to predict used-car market value from structured vehicle listing attributes. The final model comparison showed that **{best['model']}** achieved the best performance among the tested models.

## Best Model Performance

| Metric | Value |
|---|---:|
| MAE | {eval_metrics['MAE']} |
| RMSE | {eval_metrics['RMSE']} |
| R² | {eval_metrics['R2']} |
| Test Rows | {eval_metrics['test_rows']} |

The MAE value means that, on average, the model prediction differs from the actual listing price by approximately **{eval_metrics['MAE']}** currency units. The R² score indicates that the model explains approximately **{round(eval_metrics['R2'] * 100, 2)}%** of the variation in used-car prices within the test sample.

## Feature Importance Summary

The most influential features identified by the best model were:

{top_features.to_markdown(index=False)}

These features are consistent with real-world used-car valuation logic, where vehicle identity, usage, age, and market location all contribute to final price.

## Business Interpretation Layer

A derived price-positioning layer was created by comparing actual listing price with predicted fair market value.

- **Underpriced** if the actual price was more than 10% below predicted fair value
- **Fair** if the actual price was within ±10% of predicted fair value
- **Overpriced** if the actual price was more than 10% above predicted fair value

Distribution:

```text
{price_positioning_text}
```

## Key Takeaway

The model shows strong predictive capability for used-car market valuation. The project demonstrates a complete machine learning workflow including large-scale data loading, data cleaning, exploratory data analysis, feature engineering, model selection, model training, evaluation, prediction, and business interpretation.
"""

    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Saved report sections to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
