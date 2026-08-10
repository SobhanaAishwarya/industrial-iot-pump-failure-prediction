"""
Data loading and dataset-level inspection utilities.

Responsible for:
    - Reading the raw Water Pump Industrial Telemetry CSV safely.
    - Engineering simple time features from the timestamp column.
    - Deriving the binary `equipment_failure` target from `machine_status`
      (or passing an already-binary target through untouched).
    - Producing summary statistics consumed by the Streamlit "Dataset
      statistics" page.

Kept deliberately free of any modeling logic so it can be reused by the
training pipeline, the prediction pipeline, and the dashboard alike.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger(__name__)


class DataLoadError(Exception):
    """Raised when the raw dataset cannot be read or does not match the expected schema."""


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the raw telemetry CSV from disk.

    Parameters
    ----------
    path: optional override of config.RAW_DATA_PATH (useful for a Streamlit
        file-uploader flow that reads from an in-memory buffer path).

    Raises
    ------
    DataLoadError if the file is missing, unreadable, or empty. Callers
    (CLI scripts, Streamlit pages) should catch this and show a friendly
    message rather than letting a raw traceback surface.
    """
    csv_path = Path(path) if path is not None else config.RAW_DATA_PATH

    if not csv_path.exists():
        raise DataLoadError(
            f"No dataset found at '{csv_path}'. Run `python generate_sample_data.py` "
            "to create a synthetic dataset, or place the real Water Pump Industrial "
            "Telemetry CSV at that path."
        )

    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Failed to parse '{csv_path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Dataset at '{csv_path}' contains zero rows.")

    # Drop a stray pandas index column some CSV exports include.
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    logger.info("Loaded raw dataset: %s rows x %s columns from %s", df.shape[0], df.shape[1], csv_path)
    return df


def engineer_time_features(df: pd.DataFrame, timestamp_col: str = config.TIMESTAMP_COLUMN) -> pd.DataFrame:
    """Replace a raw timestamp column with a handful of learnable time features.

    Raw timestamps are high-cardinality identifiers, not useful model inputs;
    hour-of-day and day-of-week, however, often carry real signal for
    equipment that runs on shift patterns. Leaves the frame untouched if the
    timestamp column is absent (e.g. already engineered, or dataset lacks one).
    """
    if timestamp_col not in df.columns:
        return df

    df = df.copy()
    parsed = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["hour_of_day"] = parsed.dt.hour
    df["day_of_week"] = parsed.dt.dayofweek
    df["is_weekend"] = parsed.dt.dayofweek.isin([5, 6]).astype(int)
    return df


def derive_target(
    df: pd.DataFrame,
    raw_target_col: str = config.RAW_TARGET_COLUMN,
    target_col: str = config.TARGET_COLUMN,
    status_map: Optional[dict] = None,
) -> pd.DataFrame:
    """Ensure a binary `target_col` (0/1) exists on the returned frame.

    If `raw_target_col` (e.g. `machine_status`) is present, it is mapped to
    a binary failure flag using `status_map`. If `target_col` already exists
    (dataset already ships a binary label), it is used as-is. Otherwise an
    informative error is raised so the caller doesn't silently train on
    garbage.
    """
    status_map = status_map or config.STATUS_TO_FAILURE_MAP
    df = df.copy()

    if target_col in df.columns:
        unique_vals = set(pd.unique(df[target_col].dropna()))
        if not unique_vals.issubset({0, 1}):
            raise DataLoadError(
                f"Column '{target_col}' exists but is not binary (0/1); found values {unique_vals}."
            )
        return df

    if raw_target_col not in df.columns:
        raise DataLoadError(
            f"Could not find a target. Expected either a binary column '{target_col}' "
            f"or a raw status column '{raw_target_col}' mappable via {status_map}."
        )

    unmapped = set(df[raw_target_col].dropna().unique()) - set(status_map.keys())
    if unmapped:
        raise DataLoadError(
            f"'{raw_target_col}' contains values not covered by STATUS_TO_FAILURE_MAP: {unmapped}."
        )

    df[target_col] = df[raw_target_col].map(status_map).astype(int)
    return df


def prepare_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """Convenience wrapper: load raw data, engineer time features, derive target."""
    df = load_raw_data(path)
    df = engineer_time_features(df)
    df = derive_target(df)
    return df


def get_dataset_overview(df: pd.DataFrame, target_col: str = config.TARGET_COLUMN) -> dict:
    """Compute the summary statistics shown on the Streamlit dataset page.

    Returns a plain dict of JSON-serialisable values (no numpy scalars) so it
    can be cached, logged, or persisted without extra conversion.
    """
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    overview = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(v) for col, v in missing_counts.items() if v > 0},
        "missing_pct": {col: float(v) for col, v in missing_pct.items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if target_col in df.columns:
        counts = df[target_col].value_counts().sort_index()
        overview["class_counts"] = {int(k): int(v) for k, v in counts.items()}
        overview["class_balance_pct"] = {
            int(k): round(float(v) / len(df) * 100, 2) for k, v in counts.items()
        }

    numeric_df = df.select_dtypes(include=[np.number])
    overview["numeric_summary"] = numeric_df.describe().round(3).to_dict()

    return overview


def get_feature_input_specs(df: pd.DataFrame, numeric_features: list, categorical_features: list) -> dict:
    """Derive dashboard input-widget bounds/defaults straight from the data.

    Keeps the Streamlit "Live Prediction" form in sync with whatever dataset
    is loaded (real or synthetic) instead of hardcoding sensor ranges.
    """
    specs = {}
    for col in numeric_features:
        values = df[col].dropna()
        specs[col] = {
            "type": "numeric",
            "min": float(values.min()) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 1.0,
            "median": float(values.median()) if len(values) else 0.0,
        }
    for col in categorical_features:
        options = sorted(df[col].dropna().unique().tolist())
        mode = df[col].mode(dropna=True)
        specs[col] = {
            "type": "categorical",
            "options": options,
            "default": mode.iloc[0] if not mode.empty else (options[0] if options else None),
        }
    return specs
