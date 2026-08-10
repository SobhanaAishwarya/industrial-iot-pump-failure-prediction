"""
Synthetic Water Pump Industrial Telemetry dataset generator.

The real-world reference for this project is the Kaggle "Pump Sensor Data"
dataset: a timestamp, ~50 sensor_XX numeric readings, and a `machine_status`
column of NORMAL / RECOVERING / BROKEN. That dataset requires a Kaggle
account to download, so this script fabricates a schema-compatible stand-in
you can run the whole pipeline against immediately.

To use the real dataset instead: download it and save it as
`data/raw/pump_sensor_data.csv` with the same column names
(timestamp, sensor_00 ... sensor_NN, machine_status) - every downstream
module reads columns dynamically, so nothing else needs to change.

Simulation model
-----------------
Each row belongs to a hidden "risk" process that is 0 during normal
operation and ramps toward 1 in the run-up to a failure event, triggering a
short BROKEN window followed by a longer RECOVERING window before falling
back to 0. A handful of "critical" sensors drift with risk; the rest are
mostly noise - giving the classifiers a genuine (not trivially linear)
pattern to learn, and a realistic minority failure class.

We also inject two operational categorical columns (`operation_mode`,
`pump_station`) that the real dataset lacks, purely to exercise the
one-hot-encoding branch of the preprocessing pipeline end to end.
"""

import argparse

import numpy as np
import pandas as pd

import config
from src.utils import get_logger

logger = get_logger(__name__)


def _generate_risk_profile(n_samples: int, rng: np.random.Generator) -> tuple:
    """Build the hidden risk timeline and the aligned machine_status labels."""
    risk = np.zeros(n_samples)
    status = np.array(["NORMAL"] * n_samples, dtype=object)

    avg_gap = 2200
    n_events = max(1, n_samples // avg_gap)
    cursor = int(rng.integers(400, avg_gap))

    for _ in range(n_events):
        ramp_len = int(rng.integers(100, 300))
        broken_len = int(rng.integers(5, 15))
        recover_len = int(rng.integers(30, 80))
        event_len = ramp_len + broken_len + recover_len

        if cursor + event_len >= n_samples:
            break

        ramp_end = cursor + ramp_len
        broken_end = ramp_end + broken_len
        recover_end = broken_end + recover_len

        risk[cursor:ramp_end] = np.linspace(0, 1, ramp_len, endpoint=False)
        risk[ramp_end:broken_end] = 1.0
        status[ramp_end:broken_end] = "BROKEN"
        risk[broken_end:recover_end] = np.linspace(1, 0, recover_len, endpoint=False)
        status[broken_end:recover_end] = "RECOVERING"

        cursor = recover_end + int(rng.integers(500, avg_gap))

    return risk, status


def generate_dataset(
    n_samples: int = config.N_SAMPLES,
    n_sensors: int = config.N_SENSORS,
    missing_rate: float = config.MISSING_VALUE_RATE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Generate a full synthetic telemetry DataFrame, missing values included."""
    rng = np.random.default_rng(random_state)

    timestamps = pd.date_range(start="2025-01-01", periods=n_samples, freq="1min")
    risk, status = _generate_risk_profile(n_samples, rng)

    n_critical = min(5, n_sensors)
    sensor_means = rng.uniform(20, 80, size=n_sensors)
    sensor_stds = rng.uniform(1.5, 8.0, size=n_sensors)
    drift_directions = rng.choice([-1, 1], size=n_sensors)
    drift_magnitudes = rng.uniform(8, 25, size=n_sensors)

    data = {}
    for i in range(n_sensors):
        col_name = f"sensor_{i:02d}"
        is_critical = i < n_critical
        drift = drift_directions[i] * drift_magnitudes[i] * risk if is_critical else 0.0
        noise_scale = sensor_stds[i] * (1 + 0.25 * risk)
        values = rng.normal(loc=sensor_means[i] + drift, scale=noise_scale, size=n_samples)
        data[col_name] = values

    df = pd.DataFrame(data)
    df.insert(0, config.TIMESTAMP_COLUMN, timestamps)
    df["operation_mode"] = rng.choice(
        ["AUTOMATIC", "MANUAL", "MAINTENANCE"], size=n_samples, p=[0.75, 0.20, 0.05]
    )
    df["pump_station"] = rng.choice(
        ["STATION_A", "STATION_B", "STATION_C"], size=n_samples, p=[0.4, 0.35, 0.25]
    )
    df[config.RAW_TARGET_COLUMN] = status

    # Simulate disconnected/corrupted sensors: missing-completely-at-random
    # cells across the numeric sensor columns only.
    sensor_cols = [f"sensor_{i:02d}" for i in range(n_sensors)]
    mask = rng.random(size=(n_samples, n_sensors)) < missing_rate
    sensor_values = df[sensor_cols].to_numpy(copy=True)
    sensor_values[mask] = np.nan
    df[sensor_cols] = sensor_values

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic Water Pump telemetry dataset.")
    parser.add_argument("--n-samples", type=int, default=config.N_SAMPLES)
    parser.add_argument("--output", type=str, default=str(config.RAW_DATA_PATH))
    args = parser.parse_args()

    df = generate_dataset(n_samples=args.n_samples)

    failure_rate = (df[config.RAW_TARGET_COLUMN] != "NORMAL").mean() * 100
    missing_rate = df.filter(like="sensor_").isna().mean().mean() * 100
    logger.info(
        "Generated %d rows | failure-related rows: %.2f%% | missing sensor cells: %.2f%%",
        len(df), failure_rate, missing_rate,
    )

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved synthetic dataset to %s", args.output)


if __name__ == "__main__":
    main()
