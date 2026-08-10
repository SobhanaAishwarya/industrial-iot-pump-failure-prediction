"""
Model training: builds three candidate pipelines (Logistic Regression,
Random Forest, Gradient Boosting), compares them with stratified k-fold
cross-validation, refits the best one on the full training split, evaluates
it on the held-out test set, and persists everything needed by the
Streamlit dashboard and the prediction module.

Run directly to (re)train from the command line:

    python -m src.train
"""

import argparse
import time
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

import config
from src.data_loader import prepare_dataset
from src.evaluate import evaluate_pipeline
from src.preprocessing import build_preprocessor, identify_feature_types, split_train_test
from src.utils import get_logger, save_json

logger = get_logger(__name__)


def get_candidate_models(random_state: int = config.RANDOM_STATE) -> Dict[str, object]:
    """The three classifiers the brief asks us to compare.

    `class_weight="balanced"` on the linear/forest models counteracts the
    typical failure-class imbalance in telemetry data without needing a
    separate resampling step. GradientBoostingClassifier has no such knob,
    so it relies on the CV comparison to reveal if imbalance hurts it.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }


def build_full_pipeline(preprocessor, estimator) -> Pipeline:
    """Wrap a preprocessor + classifier into a single fit/predict unit so
    cross-validation never leaks test-fold statistics into imputation or
    scaling (each fold refits the preprocessor independently)."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])


def cross_validate_models(
    models: Dict[str, object],
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = config.CV_FOLDS,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Stratified k-fold CV for every candidate, scored on all five metrics.

    Returns one row per model with `{metric}_mean` / `{metric}_std` columns,
    ready to render as the "Model comparison" table/chart.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows = []

    for name, estimator in models.items():
        logger.info("Cross-validating %s ...", name)
        pipeline = build_full_pipeline(preprocessor, estimator)
        start = time.time()
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring=config.CV_SCORING,
            n_jobs=-1,
            return_train_score=False,
        )
        elapsed = time.time() - start

        row = {"model": name, "fit_time_sec": round(elapsed, 2)}
        for metric in config.CV_SCORING:
            key = f"test_{metric}"
            row[f"{metric}_mean"] = float(np.mean(scores[key]))
            row[f"{metric}_std"] = float(np.std(scores[key]))
        rows.append(row)
        logger.info(
            "%s: f1=%.4f (+/- %.4f), roc_auc=%.4f",
            name,
            row["f1_mean"],
            row["f1_std"],
            row["roc_auc_mean"],
        )

    return pd.DataFrame(rows).sort_values(
        f"{config.SELECTION_METRIC}_mean", ascending=False
    ).reset_index(drop=True)


def select_best_model(cv_results: pd.DataFrame, metric: str = config.SELECTION_METRIC) -> str:
    """Pick the model name with the highest mean CV score on `metric`."""
    best_row = cv_results.sort_values(f"{metric}_mean", ascending=False).iloc[0]
    return str(best_row["model"])


def run_training_pipeline(
    data_path: Optional[str] = None,
    save_artifacts: bool = True,
) -> dict:
    """End-to-end pipeline: load data -> split -> compare models via CV ->
    refit + evaluate the best one -> persist model and reports.

    Returns a summary dict (also what gets written to disk as JSON) so both
    the CLI entry point and the Streamlit "Retrain" button can consume it
    without re-reading files immediately after writing them.
    """
    logger.info("Loading and preparing dataset ...")
    df = prepare_dataset(data_path)

    numeric_features, categorical_features = identify_feature_types(df)
    feature_cols = numeric_features + categorical_features

    X_train, X_test, y_train, y_test = split_train_test(df, feature_cols=feature_cols)
    logger.info(
        "Train/test split: %d / %d rows, failure rate train=%.2f%% test=%.2f%%",
        len(X_train), len(X_test), y_train.mean() * 100, y_test.mean() * 100,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = get_candidate_models()

    cv_results = cross_validate_models(models, preprocessor, X_train, y_train)
    best_model_name = select_best_model(cv_results)
    logger.info("Best model by %s: %s", config.SELECTION_METRIC, best_model_name)

    best_pipeline = build_full_pipeline(preprocessor, models[best_model_name])
    best_pipeline.fit(X_train, y_train)

    test_evaluation = evaluate_pipeline(best_pipeline, X_test, y_test)
    logger.info("Held-out test metrics for %s: %s", best_model_name, test_evaluation["metrics"])

    summary = {
        "best_model_name": best_model_name,
        "selection_metric": config.SELECTION_METRIC,
        "cv_results": cv_results.to_dict(orient="records"),
        "test_evaluation": test_evaluation,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    if save_artifacts:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, config.BEST_MODEL_PATH)
        save_json(summary["cv_results"], config.MODEL_COMPARISON_PATH)
        save_json(test_evaluation, config.EVALUATION_REPORT_PATH)
        save_json(
            {
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "best_model_name": best_model_name,
                "trained_at": summary["trained_at"],
                # X_test/y_test are stashed as records so the dashboard can
                # redraw evaluation charts (e.g. re-derive a fresh ROC curve)
                # without needing to keep the split reproducible via seed alone.
                "sample_test_rows": X_test.head(50).assign(**{config.TARGET_COLUMN: y_test.head(50)}).to_dict(orient="records"),
            },
            config.FEATURE_METADATA_PATH,
        )
        logger.info("Saved best model to %s", config.BEST_MODEL_PATH)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and compare pump failure prediction models.")
    parser.add_argument("--data-path", default=None, help="Optional override for the raw CSV path.")
    args = parser.parse_args()

    try:
        summary = run_training_pipeline(data_path=args.data_path)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        logger.error("Training failed: %s", exc)
        raise SystemExit(1) from exc

    print("\n=== Model comparison (cross-validated) ===")
    print(pd.DataFrame(summary["cv_results"]).to_string(index=False))
    print(f"\nBest model: {summary['best_model_name']}")
    print(f"Test metrics: {summary['test_evaluation']['metrics']}")


if __name__ == "__main__":
    main()
