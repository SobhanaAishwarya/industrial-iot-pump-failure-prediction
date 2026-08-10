"""
Prediction module: loads the persisted best model and serves single-record
or batch predictions.

This is the module the "Sensor Cleaning" reliability goal is really about:
whatever malformed, partial, or extra-column input arrives from a live
sensor feed or a manual dashboard entry, it must be coerced into something
the pipeline can score - or fail with a clear, caught error - rather than
crashing the app.
"""

from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from src.utils import get_logger

logger = get_logger(__name__)


class ModelNotFoundError(Exception):
    """Raised when a prediction is requested before any model has been trained."""


class PredictionInputError(Exception):
    """Raised when input data cannot be coerced into a valid model input row."""


def load_model(path=None) -> Pipeline:
    """Load the persisted best pipeline from disk.

    Raises ModelNotFoundError with actionable guidance instead of letting
    joblib's FileNotFoundError bubble up unexplained.
    """
    model_path = path or config.BEST_MODEL_PATH
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Train one first via `python -m src.train` or the dashboard's "
            "'Train / Retrain Models' button."
        )
    try:
        return joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - joblib can raise several exception types
        raise ModelNotFoundError(f"Failed to load model from '{model_path}': {exc}") from exc


def _safe_numeric_coerce(series: pd.Series) -> pd.Series:
    """Convert an object-dtype series to numeric only if every non-null value
    parses cleanly; otherwise return it unchanged (it's genuinely categorical).

    Avoids the deprecated `pd.to_numeric(..., errors="ignore")` API while
    keeping the same intent: don't corrupt a categorical column like
    `operation_mode` into all-NaN just because it isn't numeric.
    """
    converted = pd.to_numeric(series, errors="coerce")
    became_nan = converted.isna() & series.notna()
    if became_nan.any():
        return series
    return converted


def _expected_columns(pipeline: Pipeline) -> list:
    """Recover the exact column list/order the fitted ColumnTransformer expects."""
    preprocessor = pipeline.named_steps["preprocessor"]
    columns = []
    for _, _, cols in preprocessor.transformers_:
        if cols == "remainder":
            continue
        columns.extend(cols)
    return columns


def _coerce_input_row(input_dict: Dict, expected_columns: list) -> pd.DataFrame:
    """Build a single-row DataFrame matching the model's expected schema.

    - Missing keys become NaN (the fitted median/most-frequent imputers
      handle them at transform time - this is the core "never crash on a
      disconnected sensor" behaviour).
    - Non-numeric junk in a numeric field (e.g. an empty string from a form)
      is coerced to NaN rather than raising, again deferring to imputation.
    - Extra keys not in the schema are silently dropped.
    """
    row = {}
    for col in expected_columns:
        value = input_dict.get(col, np.nan)
        if isinstance(value, str) and value.strip() == "":
            value = np.nan
        row[col] = value

    df = pd.DataFrame([row], columns=expected_columns)

    for col in expected_columns:
        if df[col].dtype == object:
            df[col] = _safe_numeric_coerce(df[col])

    return df


def predict_single(pipeline: Pipeline, input_dict: Dict) -> Dict:
    """Predict failure class + probability for one sensor reading.

    Returns {"prediction": 0/1, "label": "Normal"/"Failure", "failure_probability": float}.
    """
    if not isinstance(input_dict, dict) or len(input_dict) == 0:
        raise PredictionInputError("Input must be a non-empty dictionary of sensor readings.")

    expected_columns = _expected_columns(pipeline)
    row_df = _coerce_input_row(input_dict, expected_columns)

    try:
        proba = pipeline.predict_proba(row_df)[0, 1]
        pred = int(proba >= 0.5)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly UI error
        raise PredictionInputError(f"Could not generate a prediction from the given input: {exc}") from exc

    return {
        "prediction": pred,
        "label": config.TARGET_CLASS_NAMES[pred],
        "failure_probability": float(proba),
    }


def predict_batch(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Predict for every row of an arbitrary DataFrame (e.g. an uploaded CSV).

    Missing expected columns are added as all-NaN (imputed away); unexpected
    columns are dropped. Rows that are entirely unusable still get a row in
    the output with NaN probability rather than crashing the whole batch.
    """
    expected_columns = _expected_columns(pipeline)
    working = df.copy()

    for col in expected_columns:
        if col not in working.columns:
            working[col] = np.nan
    working = working[expected_columns]

    for col in expected_columns:
        if working[col].dtype == object:
            working[col] = _safe_numeric_coerce(working[col])

    try:
        proba = pipeline.predict_proba(working)[:, 1]
    except Exception as exc:  # noqa: BLE001
        raise PredictionInputError(f"Batch prediction failed: {exc}") from exc

    result = df.copy()
    result["failure_probability"] = proba
    result["prediction"] = (proba >= 0.5).astype(int)
    result["label"] = result["prediction"].map(dict(enumerate(config.TARGET_CLASS_NAMES)))
    return result
