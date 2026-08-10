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
# Visual palette (validated categorical / sequential / status colors - dark
# theme: near-black page, dark card surface, white/light-gray text)
# --------------------------------------------------------------------------
PALETTE = {
    "categorical": ["#3987e5", "#d95926", "#199e70"],  # blue, orange, aqua (dark-mode steps)
    "sequential_blue": ["#184f95", "#2a78d6", "#6da7ec", "#cde2fb"],  # low -> high, recedes toward the dark surface
    "status_good": "#0ca30c",
    "status_critical": "#d03b3b",
    "ink_primary": "#ffffff",
    "ink_secondary": "#c3c2b7",
    "ink_muted": "#898781",
    "gridline": "#2c2c2a",
    "surface": "#1a1a19",  # card / chart surface
    "page": "#0d0d0d",  # app background, near-black
}
