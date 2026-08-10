# Industrial IoT Pump Failure Prediction

A production-style ML system that predicts water pump equipment failure from
industrial sensor telemetry, built around the **Water Pump Industrial
Telemetry Dataset** schema (timestamp, `sensor_00..sensor_NN`,
`machine_status`).

Sensors disconnect and send corrupted or missing readings in the real
world; this project's core design goal is a pipeline that keeps working
(imputing, scaling, predicting) instead of crashing when that happens.

## Project structure

```
industrial_iot_pump_failure_prediction/
├── streamlit_app.py           # Streamlit dashboard (all 7 pages)
├── config.py                  # Single source of truth: paths, schema, hyperparameters
├── generate_sample_data.py    # Synthetic telemetry generator (schema-compatible stand-in)
├── requirements.txt
├── data/raw/                  # Place pump_sensor_data.csv here
├── models/                    # best_model.joblib + JSON reports (created by training)
└── src/
    ├── data_loader.py         # CSV loading, time-feature engineering, target derivation, stats
    ├── preprocessing.py       # ColumnTransformer: median/most-frequent imputation, scaling, one-hot
    ├── train.py                # 3-model CV comparison, best-model selection, joblib persistence
    ├── evaluate.py              # Metrics, confusion matrix, ROC data, feature importance
    ├── predict.py                # Single/batch prediction, resilient to missing/malformed input
    └── utils.py                  # Logging + JSON I/O helpers
```

## Setup

```bash
pip install -r requirements.txt

# No Kaggle dataset on hand? Generate a schema-compatible synthetic one:
python generate_sample_data.py

# Train and compare Logistic Regression, Random Forest, and Gradient Boosting:
python -m src.train

# Launch the dashboard:
streamlit run streamlit_app.py
```

## Using the real dataset

Download the Kaggle "Pump Sensor Data" CSV and save it as
`data/raw/pump_sensor_data.csv` with its original columns
(`timestamp`, `sensor_00` ... `sensor_51`, `machine_status`). Nothing else
needs to change - `src/preprocessing.py` detects numeric vs. categorical
columns dynamically, and `src/data_loader.py` maps `machine_status`
(NORMAL / RECOVERING / BROKEN) to a binary `equipment_failure` label via
`config.STATUS_TO_FAILURE_MAP`. If your CSV already ships a binary
`equipment_failure` column, it's used as-is.

## Dashboard pages

| Page | What it shows |
|---|---|
| Overview | Headline stats, pipeline summary, current best model |
| Dataset Explorer | Shape, missing values, class balance, descriptive stats |
| Preprocessing | Numeric/categorical split, imputation & encoding strategy |
| Model Comparison | Stratified k-fold CV results across 3 models, 5 metrics |
| Evaluation | Held-out test metrics, confusion matrix, ROC curve, classification report |
| Feature Importance | Tree-model feature importances (or coefficients for linear models) |
| Live Prediction | Manual sensor-value entry, or batch CSV scoring, with failure probability |

## Design notes

- **No leakage**: the `ColumnTransformer` is fit only inside each
  cross-validation fold / on the training split, never on the full dataset.
- **Model selection**: the candidate with the best mean cross-validated
  F1-score (`config.SELECTION_METRIC`) is refit on the full training split,
  evaluated once on the untouched test split, and persisted with Joblib.
- **Resilience**: `src/predict.py` coerces missing fields to `NaN` rather
  than raising, so the fitted imputers absorb disconnected-sensor scenarios
  at inference time exactly as they did during training.
