"""
Evaluation utilities: point metrics, confusion matrix, classification
report, ROC curve data, and feature importance extraction for tree models.

All functions take an already-fitted sklearn Pipeline (preprocessor +
classifier) plus held-out data, and return plain Python / numpy structures
that are easy to render in Streamlit or serialize to JSON.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

import config
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Core classification metrics used everywhere in the dashboard.

    `zero_division=0` avoids sklearn warnings/crashes on the (rare, but
    possible) fold where a class is entirely absent from predictions.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Run a fitted pipeline on held-out data and collect every artefact the
    dashboard's "Evaluation" page needs in one pass."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=config.TARGET_CLASS_NAMES, output_dict=True, zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def get_feature_importance(
    pipeline: Pipeline,
    top_n: int = config.TOP_N_FEATURE_IMPORTANCE,
) -> Optional[pd.DataFrame]:
    """Extract and rank feature importances for tree-based models.

    Returns None (rather than raising) for models without
    `feature_importances_` - e.g. Logistic Regression - so callers can
    simply skip rendering the chart for those models.
    """
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df
