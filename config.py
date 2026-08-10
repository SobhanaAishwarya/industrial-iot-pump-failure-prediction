"""
Central configuration for the Industrial IoT Pump Failure Prediction system.

Every path, column name, and hyperparameter that other modules need is defined
here so there is a single source of truth. Changing the dataset schema or
retraining behaviour should only ever require edits to this file.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "pump_sensor_data.csv"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison_results.json"
FEATURE_METADATA_PATH = MODEL_DIR / "feature_metadata.json"
EVALUATION_REPORT_PATH = MODEL_DIR / "evaluation_report.json"

for _dir in (RAW_DATA_DIR, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
# The Water Pump Industrial Telemetry Dataset ships with a `machine_status`
# column carrying NORMAL / RECOVERING / BROKEN. We collapse it to a binary
# failure flag. If your CSV already has a binary target, set
# RAW_TARGET_COLUMN = None and make sure TARGET_COLUMN exists as 0/1.
TIMESTAMP_COLUMN = "timestamp"
RAW_TARGET_COLUMN = "machine_status"
TARGET_COLUMN = "equipment_failure"
TARGET_CLASS_NAMES = ["Normal", "Failure"]

# machine_status -> binary failure flag. RECOVERING is treated as an
# abnormal/failure state because the pump is not operating normally.
STATUS_TO_FAILURE_MAP = {
    "NORMAL": 0,
    "RECOVERING": 1,
    "BROKEN": 1,
}

# Columns that should never be used as model features (identifiers, raw
# timestamp once time features are engineered, raw status column, etc.)
NON_FEATURE_COLUMNS = [TIMESTAMP_COLUMN, RAW_TARGET_COLUMN, TARGET_COLUMN, "row_id"]

# --------------------------------------------------------------------------
# Train / evaluate configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Metric used to pick the "best" model out of the compared candidates.
SELECTION_METRIC = "f1"

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

TOP_N_FEATURE_IMPORTANCE = 15

# --------------------------------------------------------------------------
# Synthetic sample data generator (used when no real dataset is available)
# --------------------------------------------------------------------------
N_SAMPLES = 20000
N_SENSORS = 30
MISSING_VALUE_RATE = 0.04

# --------------------------------------------------------------------------
# Visual palette (validated categorical / sequential / status colors)
# --------------------------------------------------------------------------
PALETTE = {
    "categorical": ["#2a78d6", "#eb6834", "#1baf7a"],  # blue, orange, aqua
    "sequential_blue": ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95"],
    "status_good": "#0ca30c",
    "status_critical": "#d03b3b",
    "ink_primary": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "gridline": "#e1e0d9",
    "surface": "#fcfcfb",
}
