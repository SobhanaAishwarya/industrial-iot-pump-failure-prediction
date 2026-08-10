"""
Preprocessing: feature-type detection and the Scikit-Learn ColumnTransformer
that makes the rest of the pipeline resilient to messy sensor data.

Design goal (per the "Industrial IoT Sensor Cleaning" brief): a live system
that predicts on real telemetry must never crash because a sensor dropped a
reading. All missing-value handling is expressed as sklearn transformers
fitted on training data, so the exact same imputation/scaling/encoding is
replayed identically at prediction time - including on rows the model has
never seen, via `.transform()` alone (no re-fitting, no leakage).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.utils import get_logger

logger = get_logger(__name__)


def identify_feature_types(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    exclude_cols: List[str] = None,
) -> Tuple[List[str], List[str]]:
    """Split feature columns into numeric and categorical lists by dtype.

    Purely dtype-driven so the same code works whether the dataset is the
    52-sensor real Kaggle CSV (no categoricals at all) or the enriched
    synthetic sample used for the demo (which adds operational categoricals
    to exercise the one-hot branch).
    """
    exclude = set(exclude_cols or []) | set(config.NON_FEATURE_COLUMNS) | {target_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    logger.info(
        "Identified %d numeric features and %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build the ColumnTransformer shared by every candidate model.

    Numeric branch: median imputation (robust to outlier sensor spikes) then
    standard scaling. Categorical branch: most-frequent imputation then
    one-hot encoding with unknown categories ignored at inference time
    (so an unseen operational mode at prediction time degrades gracefully
    to an all-zero encoding instead of raising).
    """
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns found to build a preprocessor from.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_output_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Return the flat list of feature names produced after transformation.

    Needed to map tree-model `feature_importances_` (which are indexed by
    the transformed, one-hot-expanded feature space) back to human-readable
    names for the dashboard's feature-importance chart.
    """
    return list(preprocessor.get_feature_names_out())


def split_train_test(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    feature_cols: List[str] = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """Stratified train/test split so the (typically imbalanced) failure
    class keeps the same proportion in both splits."""
    y = df[target_col]
    X = df[feature_cols] if feature_cols is not None else df.drop(columns=[target_col])

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocessing_summary(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> dict:
    """Human-readable summary of what the pipeline will do, for the
    Streamlit "Preprocessing" page - computed from data, not hardcoded."""
    missing_numeric = {
        c: int(df[c].isna().sum()) for c in numeric_features if df[c].isna().sum() > 0
    }
    missing_categorical = {
        c: int(df[c].isna().sum()) for c in categorical_features if df[c].isna().sum() > 0
    }
    categorical_cardinality = {c: int(df[c].nunique(dropna=True)) for c in categorical_features}

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_numeric": len(numeric_features),
        "n_categorical": len(categorical_features),
        "numeric_imputation_strategy": "median",
        "categorical_imputation_strategy": "most_frequent",
        "scaling_strategy": "StandardScaler (zero mean, unit variance)",
        "encoding_strategy": "OneHotEncoder (unknown categories ignored at inference)",
        "missing_before_imputation_numeric": missing_numeric,
        "missing_before_imputation_categorical": missing_categorical,
        "categorical_cardinality": categorical_cardinality,
        "total_missing_cells": int(df[numeric_features + categorical_features].isna().sum().sum()),
    }
