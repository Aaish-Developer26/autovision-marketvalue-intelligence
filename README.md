# CPP Project — Used Car Market Value Prediction

## Project Goal
Build a machine learning system to predict the fair market price of used cars from vehicle specifications and listing attributes.

## Primary Dataset
- `true_car_listing_01.csv`
- `true_car_listings_02.csv`

## Secondary Dataset
- `pakwheels_pakistan_automobile_dataset.csv`

## Recommended Strategy
- Use **TrueCar** as the main dataset for the core regression pipeline.
- Use **PakWheels** as a secondary experiment for:
  - localized comparison,
  - hypothesis testing on depreciation behavior,
  - optional fine-tuning / transfer-learning extension.

## Current Dataset Understanding
### TrueCar
Confirmed shared columns:
- `Price`, `Year`, `Mileage`, `City`, `State`, `Vin`, `Make`, `Model`
- `true_car_listing_01.csv` also contains `Id`

### PakWheels
Columns:
- `title`, `price`, `city`, `model`, `mileage`, `fuel_type`, `transmission`,
  `registered`, `color`, `assembly`, `engine_capacity`, `post_date`,
  `price_category`, `mileage_category`, `post_day_of_week`, `vehicle_age`

## Important Note on Leakage
For PakWheels:
- `price_category` must **not** be used as an input feature for price prediction.
- `vehicle_age` should be treated as derived from `model` if `model` is actually the year field.
- `mileage_category` is derived from mileage and should be excluded from the main baseline unless used only for analysis.

## Step-by-Step Plan
1. Load and validate the datasets
2. Merge and standardize the TrueCar files
3. Perform EDA and visualize pricing patterns
4. Build preprocessing and feature engineering pipeline
5. Train regression models
6. Evaluate using MAE, RMSE, and R²
7. We will later add classification metrics only if we create a secondary label like:
Underpriced / Fairly Priced / Overpriced
Then we can use:
Confusion Matrix
Precision
Recall
F1-score
ROC-AUC
8. Generate predictions and examples
9. Document everything in README and PDF report
10. Extend with PakWheels comparative experiment

## Suggested Main Task
**Predict used car market value from structured listing attributes**

## Suggested Optional Extension
**Do cars in Pakistan depreciate at the same rate as cars in the US?**
- Train a baseline on TrueCar
- Build a comparable subset on PakWheels
- Compare price-age-mileage relationships
- Optional deep learning / transfer learning as an advanced extension

# PakWheels Extension: Cross-Market Depreciation Comparison

## Goal

Use the PakWheels Pakistan dataset as a localized extension to compare used-car depreciation behavior between:

- US market: TrueCar
- Pakistan market: PakWheels

## Why not merge PakWheels directly into TrueCar?

The two datasets come from different markets and have different currencies, structures, and feature availability. Therefore, PakWheels should not be directly merged into the main TrueCar training dataset.

Instead, we use PakWheels for a market-comparison extension.

## Core question

Do cars in Pakistan depreciate at the same rate as cars in the US?

## Key idea

Raw prices are not directly comparable because:

- TrueCar prices are in USD
- PakWheels prices are in PKR

So we use:

```text
PriceIndex = car price / market median price
```

This allows relative comparison inside each market.

## VS Code Setup
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Install packages:
```bash
pip install -r requirements.txt
```

Run the first dataset audit:
```bash
python src/data/inspect_data.py
```
# Step 1 Files

## Files included
- requirements.txt
- src/data/inspect_data.py
- src/data/merge_truecar.py

## Run order

### 1) Install packages
```bash
pip install -r requirements.txt
```

### 2) Inspect all datasets
```bash
python src/data/inspect_data.py
```

### 3) Merge the TrueCar files
```bash
python src/data/merge_truecar.py
```
### 4) This creates a clean model-ready dataset:
```bash
python src/data/clean_truecar.py
``` 
It adds these engineered features:

VehicleAge
MileagePerYear
LogPrice
LogMileage
MakeModel
AgeBucket
MileageBucket

### 5) This creates EDA plots inside: reports/figures/
```bash
python src/visualization/eda_truecar.py
``` 
It generates:

01_price_distribution.png
02_log_price_distribution.png
03_mileage_vs_price.png
04_vehicle_age_vs_price.png
05_top_makes_by_count.png
06_median_price_by_make.png
07_median_price_by_state.png
08_numeric_correlation_heatmap.png
09_median_price_by_mileage_bucket.png
10_median_price_by_age_bucket.png

### 6) This script trains stronger models:
Hist Gradient Boosting Tuned
Random Forest Tuned
Extra Trees
LightGBM, if available
```bash
python src/models/train_advanced_models.py
``` 
This will create:
1) reports/model_results_advanced.csv
2) models/best_advanced_model.joblib
Why this step matters
Our current Random Forest is good, but now we need to check whether more advanced tree-based models can reduce error further.
Expected possibilities:
1) Extra Trees may train faster than Random Forest and perform similarly or better.
2) LightGBM may perform very well on tabular data.
3) Hist Gradient Boosting is fast and reliable, but may be slightly less accurate than Random Forest.

### 6) This creates diagnostic outputs:
reports/best_model_evaluation.json
reports/sample_predictions.csv
reports/figures/11_actual_vs_predicted.png
reports/figures/12_residual_plot.png
reports/figures/13_residual_distribution.png
```bash
python src/models/evaluate_best_model.py
``` 
### 7) This creates:
reports/price_positioning_classification_report.txt
reports/figures/14_price_positioning_confusion_matrix.png
```bash
python src/models/evaluate_best_model.py
``` 
This script converts the regression model into a business interpretation layer:
Underpriced
Fair
Overpriced

It compares:
Actual listing price vs Predicted fair market value

### 8) This creates:
reports/feature_importance.csv
reports/figures/15_feature_importance.png
```bash
python src/models/extract_feature_importance.py
``` 
This tells us which features contributed most to price prediction.
Expected important features will likely include:
MakeModel
Model
Make
Mileage
VehicleAge
MileagePerYear
State
City

### 9) This creates:
reports/model_results_combined.csv
reports/figures/16_model_comparison_rmse.png
reports/figures/17_model_comparison_r2.png
```bash
python src/visualization/plot_model_comparison.py
``` 
### 10) This creates:
reports/auto_generated_report_sections.md
```bash
python src/models/generate_report_sections.py
``` 
This file will contain ready-to-use report content for:
model training summary
best model performance
feature importance summary
business interpretation layer
key takeaway

### 11) This creates:
dataset/pakwheels_clean.csv
dataset/pakwheels_cleaning_summary.txt
```bash
python src/data/clean_pakwheels.py
``` 
It standardizes the PakWheels dataset and creates comparable features:
Price
Year
VehicleAge
Mileage
MileagePerYear
LogPrice
LogMileage
City
Make
ModelName
FuelType
Transmission
EngineCapacity
Market

It also extracts Make from the title.

Example:
Honda N One Premium 2014

becomes:
Make = Honda
ModelName = N One Premium
Year = 2014

Important: it removes rows where price = 0, because that is not valid for price/depreciation comparison.

### 12) This creates:
reports/us_pak_market_comparison_dataset.csv
reports/depreciation_summary_by_age.csv
reports/common_make_depreciation_summary.csv
reports/market_summary_statistics.csv
reports/figures/18_price_index_by_vehicle_age_market.png
reports/figures/19_mileage_per_year_by_market.png
reports/figures/20_common_make_depreciation_curves.png
reports/figures/21_median_vehicle_age_by_market.png
```bash
python src/analysis/compare_market_depreciation.py
``` 
Key concept: Price Index

Because USD and PKR cannot be directly compared, we create:

PriceIndex = vehicle price / market median vehicle price

So instead of saying:

US car = $18,000
Pakistan car = PKR 4,000,000

we compare relative value:

US car price index = 1.2x market median
Pakistan car price index = 1.2x market median

This makes the comparison fairer and more meaningful.

### 13) This creates:
reports/depreciation_hypothesis_summary.txt
```bash
python src/analysis/depreciation_hypothesis_test.py
``` 
It fits a simple interpretable model separately for each market:

log(PriceIndex) = beta0 + beta1 * VehicleAge + beta2 * LogMileage

Then it compares the vehicle-age coefficient.

In simple words:

If age reduces price faster in Pakistan, Pakistan has stronger depreciation.
If age reduces price faster in the US, the US market has stronger depreciation.
If both are close, depreciation patterns are somewhat similar.

This gives you a proper hypothesis-testing style extension.

