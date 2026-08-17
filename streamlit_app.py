"""
Streamlit dashboard for the Industrial IoT Pump Failure Prediction system.

Pages:
    Overview            - headline stats and how the system works
    Dataset Explorer     - dataset statistics (shape, missingness, class balance)
    Preprocessing        - what the ColumnTransformer pipeline does, and why
    Model Comparison      - stratified cross-validation results for all 3 models
    Evaluation            - held-out test metrics, confusion matrix, ROC curve
    Feature Importance   - tree-model feature importances
    Live Prediction       - manual sensor entry + batch CSV scoring

Every page degrades gracefully: if no model has been trained yet, the
model-dependent pages show a "Train Models" call to action instead of
crashing on a missing file.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.data_loader import DataLoadError, get_dataset_overview, get_feature_input_specs, prepare_dataset
from src.evaluate import get_feature_importance
from src.predict import ModelNotFoundError, PredictionInputError, load_model, predict_batch, predict_single
from src.preprocessing import identify_feature_types, preprocessing_summary
from src.train import run_training_pipeline
from src.utils import load_json

PALETTE = config.PALETTE
MODEL_COLOR_MAP = {
    "Logistic Regression": PALETTE["categorical"][0],
    "Random Forest": PALETTE["categorical"][1],
    "Gradient Boosting": PALETTE["categorical"][2],
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

st.set_page_config(
    page_title="Industrial IoT Sensor Cleaning & Equipment Classifier Pipeline",
    layout="wide",
    page_icon="🛠",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(1100px 560px at 8% -12%, rgba(124,92,196,0.16), transparent 60%),
            radial-gradient(900px 480px at 96% -6%, rgba(216,140,190,0.12), transparent 55%),
            {PALETTE['page']};
    }}
    [data-testid="stSidebar"] {{
        background: {PALETTE['surface']}; border-right: 1px solid {PALETTE['border']};
    }}
    [data-testid="stSidebar"] * {{ color: {PALETTE['ink_secondary']}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {PALETTE['ink_primary']};
    }}
    .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}
    h1, h2, h3 {{ color: {PALETTE['ink_primary']}; font-weight: 700; }}
    p, li, span, label {{ color: {PALETTE['ink_secondary']}; }}
    .eyebrow {{
        font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: {PALETTE['accent']}; margin-bottom: 2px;
    }}
    .section-title {{
        font-size: 12.5px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; color: {PALETTE['ink_muted']}; margin: 26px 0 10px;
    }}
    [data-testid="stMetric"] {{
        background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']}; border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 1px 10px rgba(76,58,140,0.10);
        transition: border-color 0.15s ease, transform 0.15s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: rgba(74,58,167,0.45); transform: translateY(-1px);
    }}
    [data-testid="stMetricLabel"] {{ color: {PALETTE['ink_muted']}; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {PALETTE['ink_primary']}; }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {PALETTE['border']}; border-radius: 10px; overflow: hidden;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{ color: {PALETTE['ink_muted']}; }}
    .stTabs [aria-selected="true"] {{ color: {PALETTE['ink_primary']}; }}
    div.stButton > button, div.stDownloadButton > button {{
        background: {PALETTE['surface_raised']}; color: {PALETTE['ink_primary']};
        border: 1px solid {PALETTE['border']}; border-radius: 8px; font-weight: 600;
        transition: border-color 0.15s ease;
    }}
    div.stButton > button *, div.stDownloadButton > button * {{ color: inherit !important; }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        border-color: {PALETTE['accent']}; color: {PALETTE['accent']};
    }}
    div.stButton > button[kind="primary"] {{
        background: {PALETTE['accent']}; color: #ffffff; border: none;
    }}
    div.stButton > button[kind="primary"]:hover {{ background: {PALETTE['accent_hover']}; color: #ffffff; }}
    .result-card {{
        border-radius: 14px; padding: 22px 26px; margin-top: 10px; margin-bottom: 10px;
        border: 1px solid {PALETTE['border']}; box-shadow: 0 4px 18px rgba(76,58,140,0.12);
    }}
    .result-card.normal {{ background: rgba(12,163,12,0.14); border-color: {PALETTE['status_good']}; }}
    .result-card.failure {{ background: rgba(208,59,59,0.14); border-color: {PALETTE['status_critical']}; }}
    .result-card h2 {{ margin: 0 0 4px; }}
    .result-card.normal h2 {{ color: {PALETTE['status_good']}; }}
    .result-card.failure h2 {{ color: {PALETTE['status_critical']}; }}
    .result-card p {{ color: {PALETTE['ink_secondary']}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders - keep expensive I/O and model loading off the hot path of
# every Streamlit rerun.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return prepare_dataset()


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model()


@st.cache_data(show_spinner=False)
def cached_comparison_results():
    if not config.MODEL_COMPARISON_PATH.exists():
        return None
    return load_json(config.MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def cached_evaluation_report():
    if not config.EVALUATION_REPORT_PATH.exists():
        return None
    return load_json(config.EVALUATION_REPORT_PATH)


@st.cache_data(show_spinner=False)
def cached_feature_metadata():
    if not config.FEATURE_METADATA_PATH.exists():
        return None
    return load_json(config.FEATURE_METADATA_PATH)


def clear_all_caches():
    cached_dataset.clear()
    cached_model.clear()
    cached_comparison_results.clear()
    cached_evaluation_report.clear()
    cached_feature_metadata.clear()


def train_models_flow():
    """Shared "Train / Retrain Models" action used by the sidebar and by the
    inline call-to-action shown on model-dependent pages before a model exists."""
    with st.spinner("Comparing Logistic Regression, Random Forest, and Gradient Boosting via 5-fold CV..."):
        try:
            summary = run_training_pipeline()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
            st.error(f"Training failed: {exc}")
            return
    clear_all_caches()
    st.success(f"Training complete. Best model: {summary['best_model_name']}")
    st.rerun()


def render_train_prompt(subject: str):
    st.warning(f"No trained model yet - train the candidate models to unlock {subject}.")
    if st.button("Train / Retrain Models", type="primary", key=f"train_btn_{subject}"):
        train_models_flow()
    st.stop()


def blue_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_title: str):
    df = df.sort_values(x_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df[x_col], y=df[y_col], orientation="h",
            marker_color=PALETTE["categorical"][0],
            hovertemplate=f"%{{y}}: %{{x}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=None,
        plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["ink_secondary"], margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=PALETTE["gridline"]), height=max(300, 22 * len(df)),
    )
    return fig


# --------------------------------------------------------------------------
# Load data (fatal errors here are shown as a friendly page, not a traceback)
# --------------------------------------------------------------------------
try:
    df = cached_dataset()
except DataLoadError as exc:
    st.error(str(exc))
    st.info("From the project root, run:  `python generate_sample_data.py`")
    st.stop()

numeric_features, categorical_features = identify_feature_types(df)
model_ready = config.BEST_MODEL_PATH.exists()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.markdown("### Industrial IoT Sensor Cleaning & Equipment Classifier Pipeline")
st.sidebar.caption("Industrial IoT sensor cleaning & equipment classifier")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Dataset Explorer",
        "Preprocessing",
        "Model Comparison",
        "Evaluation",
        "Feature Importance",
        "Live Prediction",
    ],
)
st.sidebar.divider()
st.sidebar.caption(f"Dataset: `{config.RAW_DATA_PATH.name}`  ·  {df.shape[0]:,} rows")
if st.sidebar.button("Train / Retrain Models"):
    train_models_flow()
if model_ready:
    meta = cached_feature_metadata()
    if meta:
        st.sidebar.success(f"Active model: {meta.get('best_model_name', 'unknown')}")
else:
    st.sidebar.info("No model trained yet.")


# --------------------------------------------------------------------------
# Page: Overview
# --------------------------------------------------------------------------
def page_overview():
    st.title("Industrial IoT Sensor Cleaning & Equipment Classifier Pipeline")
    st.markdown('<div class="eyebrow">Industrial IoT · Predictive Maintenance</div>', unsafe_allow_html=True)
    st.write(
        "End-to-end pipeline that cleans messy telemetry (missing/disconnected sensor "
        "readings), compares three classifiers under stratified cross-validation, and "
        "serves failure predictions that never crash on malformed input."
    )

    overview = get_dataset_overview(df, config.TARGET_COLUMN)
    failure_pct = overview.get("class_balance_pct", {}).get(1, 0.0)
    total_missing = sum(overview["missing_counts"].values())

    n_sensors = len([c for c in numeric_features if c.startswith("sensor_")])
    cols = st.columns(4)
    cols[0].metric("Telemetry rows", f"{overview['n_rows']:,}")
    cols[1].metric("Sensors monitored", f"{n_sensors}")
    cols[2].metric("Historical failure rate", f"{failure_pct:.2f}%")
    cols[3].metric("Missing sensor cells", f"{total_missing:,}")

    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Load & clean** raw telemetry, engineer time features, derive the binary failure label.\n"
        "2. **Preprocess** via a scikit-learn `ColumnTransformer` - median imputation + scaling for "
        "sensors, most-frequent imputation + one-hot encoding for categorical context.\n"
        "3. **Compare models** - Logistic Regression, Random Forest, Gradient Boosting - with "
        f"stratified {config.CV_FOLDS}-fold cross-validation across 5 metrics.\n"
        "4. **Select & persist** the best model (by " + config.SELECTION_METRIC.upper() + ") with Joblib.\n"
        "5. **Predict** on a single manual reading or a batch CSV upload, resilient to missing fields."
    )

    if model_ready:
        report = cached_evaluation_report()
        meta = cached_feature_metadata()
        st.markdown('<div class="section-title">Current best model</div>', unsafe_allow_html=True)
        m = report["metrics"]
        cols = st.columns(5)
        cols[0].metric("Model", meta.get("best_model_name", "-"))
        cols[1].metric("F1-score", f"{m['f1']:.3f}")
        cols[2].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        cols[3].metric("Precision", f"{m['precision']:.3f}")
        cols[4].metric("Recall", f"{m['recall']:.3f}")
    else:
        st.info("Train the models from the sidebar to see live performance here.")


# --------------------------------------------------------------------------
# Page: Dataset Explorer
# --------------------------------------------------------------------------
def page_dataset_explorer():
    st.header("Dataset Statistics")
    overview = get_dataset_overview(df, config.TARGET_COLUMN)

    cols = st.columns(4)
    cols[0].metric("Rows", f"{overview['n_rows']:,}")
    cols[1].metric("Columns", overview["n_columns"])
    cols[2].metric("Duplicate rows", overview["duplicate_rows"])
    cols[3].metric("Columns with missing values", len(overview["missing_counts"]))

    st.markdown('<div class="section-title">Target class balance</div>', unsafe_allow_html=True)
    class_counts = overview.get("class_counts", {})
    if class_counts:
        labels = [config.TARGET_CLASS_NAMES[k] for k in sorted(class_counts)]
        values = [class_counts[k] for k in sorted(class_counts)]
        colors = [PALETTE["status_good"], PALETTE["status_critical"]]
        fig = go.Figure(go.Bar(
            x=labels, y=values, marker_color=colors[: len(labels)],
            text=values, textposition="outside",
        ))
        fig.update_layout(
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["ink_secondary"], margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(gridcolor=PALETTE["gridline"], title="Rows"), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Missing values by column</div>', unsafe_allow_html=True)
    if overview["missing_pct"]:
        missing_df = pd.DataFrame(
            {"column": list(overview["missing_pct"].keys()), "missing_pct": list(overview["missing_pct"].values())}
        ).sort_values("missing_pct", ascending=False).head(20)
        fig = blue_bar_chart(missing_df, "missing_pct", "column", "", "% missing")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No missing values detected in the current dataset.")

    st.markdown('<div class="section-title">Column types</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({"column": overview["dtypes"].keys(), "dtype": overview["dtypes"].values()})
    st.dataframe(dtype_df, use_container_width=True, height=250)

    st.markdown('<div class="section-title">Descriptive statistics (numeric columns)</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(overview["numeric_summary"]).T, use_container_width=True)

    st.markdown('<div class="section-title">Sample rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)


# --------------------------------------------------------------------------
# Page: Preprocessing
# --------------------------------------------------------------------------
def page_preprocessing():
    st.header("Preprocessing Summary")
    st.write(
        "A single `ColumnTransformer` handles every feature. It is fit only on the "
        "training split and then reused unchanged for validation, testing, and live "
        "predictions - so no statistic ever leaks from test data into training."
    )
    summary = preprocessing_summary(df, numeric_features, categorical_features)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### Numeric branch")
        st.markdown(
            f"- **{summary['n_numeric']} features** (sensor readings + engineered time features)\n"
            f"- Missing values -> **{summary['numeric_imputation_strategy']} imputation**\n"
            f"- Then -> **{summary['scaling_strategy']}**"
        )
    with cols[1]:
        st.markdown("##### Categorical branch")
        st.markdown(
            f"- **{summary['n_categorical']} features** (operational context)\n"
            f"- Missing values -> **{summary['categorical_imputation_strategy']} imputation**\n"
            f"- Then -> **{summary['encoding_strategy']}**"
        )

    st.markdown('<div class="section-title">Missing values before imputation</div>', unsafe_allow_html=True)
    missing = {**summary["missing_before_imputation_numeric"], **summary["missing_before_imputation_categorical"]}
    if missing:
        miss_df = pd.DataFrame({"column": missing.keys(), "missing_count": missing.values()})
        st.dataframe(miss_df.sort_values("missing_count", ascending=False), use_container_width=True, height=250)
    else:
        st.success("No missing values in the current dataset - the imputers will simply act as a no-op.")
    st.caption(f"Total missing cells across model features: {summary['total_missing_cells']:,}")

    st.markdown('<div class="section-title">Categorical feature cardinality</div>', unsafe_allow_html=True)
    if summary["categorical_cardinality"]:
        card_df = pd.DataFrame(
            {"column": summary["categorical_cardinality"].keys(), "unique_categories": summary["categorical_cardinality"].values()}
        )
        st.dataframe(card_df, use_container_width=True)
        for col in categorical_features:
            st.caption(f"**{col}**: {', '.join(map(str, sorted(df[col].dropna().unique())))}")
    else:
        st.info("No categorical features detected in the current dataset.")

    with st.expander("Full feature list"):
        st.write("Numeric:", numeric_features)
        st.write("Categorical:", categorical_features)


# --------------------------------------------------------------------------
# Page: Model Comparison
# --------------------------------------------------------------------------
def page_model_comparison():
    st.header("Model Comparison (Stratified Cross-Validation)")
    results = cached_comparison_results()
    if results is None:
        render_train_prompt("model comparison")

    cv_df = pd.DataFrame(results)
    meta = cached_feature_metadata()
    best_name = meta.get("best_model_name") if meta else None

    st.markdown(
        f"Each candidate is scored with stratified {config.CV_FOLDS}-fold cross-validation on the "
        f"training split; the model with the highest mean **{config.SELECTION_METRIC.upper()}** is selected."
    )
    if best_name:
        st.success(f"Selected model: **{best_name}**")

    display_cols = ["model"] + [f"{m}_mean" for m in config.CV_SCORING] + ["fit_time_sec"]
    pretty = cv_df[display_cols].copy()
    pretty.columns = ["Model"] + [METRIC_LABELS[m] for m in config.CV_SCORING] + ["Fit time (s)"]
    st.dataframe(pretty.round(4), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Metric comparison</div>', unsafe_allow_html=True)
    metrics = list(config.CV_SCORING.keys())
    metric_labels = [METRIC_LABELS[m] for m in metrics]
    fig = go.Figure()
    for _, row in cv_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["model"],
            x=metric_labels,
            y=[row[f"{m}_mean"] for m in metrics],
            error_y=dict(type="data", array=[row[f"{m}_std"] for m in metrics], visible=True),
            marker_color=MODEL_COLOR_MAP.get(row["model"], PALETTE["categorical"][0]),
        ))
    fig.update_layout(
        barmode="group", plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
        font_color=PALETTE["ink_secondary"], yaxis=dict(gridcolor=PALETTE["gridline"], title="Score", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# Page: Evaluation
# --------------------------------------------------------------------------
def page_evaluation():
    st.header("Held-out Test Evaluation")
    report = cached_evaluation_report()
    meta = cached_feature_metadata()
    if report is None:
        render_train_prompt("evaluation metrics")

    st.caption(f"Metrics below are for the selected best model: **{meta.get('best_model_name', '-')}**, "
               "measured on a stratified test split the model never trained on.")

    m = report["metrics"]
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    cols[1].metric("Precision", f"{m['precision']:.3f}")
    cols[2].metric("Recall", f"{m['recall']:.3f}")
    cols[3].metric("F1-score", f"{m['f1']:.3f}")
    cols[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
        cm = np.array(report["confusion_matrix"])
        seq = PALETTE["sequential_blue"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=config.TARGET_CLASS_NAMES, y=config.TARGET_CLASS_NAMES,
            colorscale=[[0, seq[0]], [0.33, seq[1]], [0.66, seq[2]], [1, seq[3]]],
            showscale=False,
        ))
        # Confusion matrices are typically dominated by one large cell (true
        # negatives); a single fixed text color would be unreadable against
        # the resulting light-to-dark color spread, so each cell gets its own
        # annotation with a color picked from how intense that cell's fill is.
        intensity = cm / max(cm.max(), 1)
        for i, actual_label in enumerate(config.TARGET_CLASS_NAMES):
            for j, predicted_label in enumerate(config.TARGET_CLASS_NAMES):
                fig.add_annotation(
                    x=predicted_label, y=actual_label, text=str(cm[i, j]), showarrow=False,
                    font=dict(size=16, color=PALETTE["surface"] if intensity[i, j] > 0.55 else PALETTE["ink_primary"]),
                )
        fig.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["ink_secondary"], margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">ROC curve</div>', unsafe_allow_html=True)
        roc = report["roc_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            line=dict(color=PALETTE["categorical"][0], width=2),
            name=f"{meta.get('best_model_name', 'Model')} (AUC={m['roc_auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=PALETTE["ink_muted"], width=1, dash="dash"),
            name="Random classifier",
        ))
        fig.update_layout(
            xaxis_title="False positive rate", yaxis_title="True positive rate",
            plot_bgcolor=PALETTE["surface"], paper_bgcolor=PALETTE["surface"],
            font_color=PALETTE["ink_secondary"],
            xaxis=dict(gridcolor=PALETTE["gridline"]), yaxis=dict(gridcolor=PALETTE["gridline"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=10, r=10, t=40, b=10), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Classification report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(report["classification_report"]).T.round(3)
    st.dataframe(report_df, use_container_width=True)


# --------------------------------------------------------------------------
# Page: Feature Importance
# --------------------------------------------------------------------------
def page_feature_importance():
    st.header("Feature Importance")
    if not model_ready:
        render_train_prompt("feature importance")

    pipeline = cached_model()
    fi_df = get_feature_importance(pipeline)

    if fi_df is not None:
        st.write("Relative importance of each (preprocessed) feature for the selected tree-based model.")
        fig = blue_bar_chart(fi_df, "importance", "feature", "", "Importance")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fi_df, use_container_width=True, hide_index=True)
    else:
        classifier = pipeline.named_steps["classifier"]
        if hasattr(classifier, "coef_"):
            st.info(
                "Feature importance charts apply to tree-based models (Random Forest, Gradient "
                "Boosting). The selected model is Logistic Regression, so coefficient magnitudes "
                "are shown instead - larger magnitude means stronger influence on the prediction."
            )
            preprocessor = pipeline.named_steps["preprocessor"]
            names = preprocessor.get_feature_names_out()
            coef_df = pd.DataFrame({"feature": names, "importance": np.abs(classifier.coef_[0])})
            coef_df = coef_df.sort_values("importance", ascending=False).head(config.TOP_N_FEATURE_IMPORTANCE)
            fig = blue_bar_chart(coef_df, "importance", "feature", "", "|Coefficient|")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("The selected model does not expose a feature importance measure.")


# --------------------------------------------------------------------------
# Page: Live Prediction
# --------------------------------------------------------------------------
def page_live_prediction():
    st.header("Live Prediction")
    if not model_ready:
        render_train_prompt("live prediction")

    pipeline = cached_model()
    specs = get_feature_input_specs(df, numeric_features, categorical_features)
    sensor_cols = [c for c in numeric_features if c.startswith("sensor_")]
    other_numeric = [c for c in numeric_features if c not in sensor_cols and c != "is_weekend"]

    tab_single, tab_batch = st.tabs(["Single Reading", "Batch CSV Upload"])

    with tab_single:
        st.write("Enter sensor values manually, or load a random historical reading to explore the demo.")
        if st.button("Load random sample from dataset"):
            sample = df.sample(1, random_state=None).iloc[0]
            for col in sensor_cols + other_numeric:
                st.session_state[f"in_{col}"] = float(sample[col]) if pd.notna(sample[col]) else specs[col]["median"]
            for col in categorical_features:
                st.session_state[f"in_{col}"] = sample[col] if pd.notna(sample[col]) else specs[col]["default"]
            st.rerun()

        # Streamlit forbids passing both `value=`/`index=` and writing to a
        # widget's session_state key (as "Load random sample" does above) in
        # the same run - so every widget below is seeded via setdefault and
        # then created with `key=` only, letting session_state own its value.
        with st.expander(f"Sensor readings ({len(sensor_cols)})", expanded=True):
            grid_cols = st.columns(5)
            for i, col in enumerate(sensor_cols):
                spec = specs[col]
                step = max((spec["max"] - spec["min"]) / 100, 0.01)
                st.session_state.setdefault(f"in_{col}", round(spec["median"], 2))
                grid_cols[i % 5].number_input(
                    col, min_value=None, max_value=None,
                    step=round(step, 3), key=f"in_{col}", format="%.3f",
                )

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        with st.expander("Time context", expanded=False):
            c1, c2 = st.columns(2)
            if "hour_of_day" in specs:
                st.session_state.setdefault("in_hour_of_day", int(specs["hour_of_day"]["median"]))
                c1.slider("Hour of day", 0, 23, key="in_hour_of_day")
            if "day_of_week" in specs:
                default_day = min(max(int(specs["day_of_week"]["median"]), 0), 6)
                st.session_state.setdefault("in_day_of_week", default_day)
                c2.selectbox("Day of week", options=list(range(7)), format_func=lambda x: day_names[x], key="in_day_of_week")

        if categorical_features:
            with st.expander("Operational context", expanded=False):
                op_cols = st.columns(len(categorical_features))
                for i, col in enumerate(categorical_features):
                    spec = specs[col]
                    options = spec["options"] or [spec["default"]]
                    st.session_state.setdefault(f"in_{col}", spec["default"] if spec["default"] in options else options[0])
                    op_cols[i].selectbox(col, options=options, key=f"in_{col}")

        st.markdown("")
        if st.button("Predict Equipment Status", type="primary"):
            input_dict = {col: st.session_state[f"in_{col}"] for col in sensor_cols + other_numeric if f"in_{col}" in st.session_state}
            for col in categorical_features:
                if f"in_{col}" in st.session_state:
                    input_dict[col] = st.session_state[f"in_{col}"]
            if "day_of_week" in input_dict:
                input_dict["is_weekend"] = 1 if input_dict["day_of_week"] in (5, 6) else 0

            try:
                result = predict_single(pipeline, input_dict)
            except PredictionInputError as exc:
                st.error(f"Could not score this input: {exc}")
            else:
                css_class = "failure" if result["prediction"] == 1 else "normal"
                st.markdown(
                    f"""<div class="result-card {css_class}">
                        <h2>{result['label']}</h2>
                        <p>Predicted failure probability: <b>{result['failure_probability']*100:.1f}%</b></p>
                        </div>""",
                    unsafe_allow_html=True,
                )
                gauge_color = PALETTE["status_critical"] if result["prediction"] == 1 else PALETTE["status_good"]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["failure_probability"] * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "bgcolor": PALETTE["surface"],
                        "steps": [
                            {"range": [0, 50], "color": "rgba(12,163,12,0.12)"},
                            {"range": [50, 100], "color": "rgba(208,59,59,0.12)"},
                        ],
                    },
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor=PALETTE["surface"])
                st.plotly_chart(fig, use_container_width=True)

    with tab_batch:
        st.write(
            "Upload a CSV of telemetry rows to score in bulk. Missing or extra columns are handled "
            "automatically - the pipeline imputes what it can and never crashes on malformed rows."
        )
        uploaded = st.file_uploader("Telemetry CSV", type="csv", key="batch_uploader")
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                results = predict_batch(pipeline, batch_df)
            except (PredictionInputError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Batch prediction failed: {exc}")
            else:
                n_failures = int(results["prediction"].sum())
                cols = st.columns(3)
                cols[0].metric("Rows scored", len(results))
                cols[1].metric("Predicted failures", n_failures)
                cols[2].metric("Failure rate", f"{n_failures / len(results) * 100:.2f}%")
                st.dataframe(results, use_container_width=True)
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="pump_predictions.csv", mime="text/csv")


PAGES = {
    "Overview": page_overview,
    "Dataset Explorer": page_dataset_explorer,
    "Preprocessing": page_preprocessing,
    "Model Comparison": page_model_comparison,
    "Evaluation": page_evaluation,
    "Feature Importance": page_feature_importance,
    "Live Prediction": page_live_prediction,
}

PAGES[page]()
