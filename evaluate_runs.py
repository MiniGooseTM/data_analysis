# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas>=2", "scipy", "tabulate"]
# ///
"""Evaluation metrics for each run of a robot state log with accelerometer columns.

A log (e.g. ``experiments/balancing_carpet_original_policy.csv``) holds several runs
back to back; each starts at ``time_s == 0``. Every run is trimmed by ``--trim`` seconds
at both ends (the robot settling after start and being stopped), then evaluated:

- lin MAE: |v - v_cmd|, v the forward velocity from integrating the accelerometer
- ang MAE: |yaw_rate - omega_cmd|
- pitch rate MAE: |pitch_rate| (the target is 0)
- |acc y|: mean absolute forward acceleration (gravity removed, see below)
- |wheel vel|, |action|, |wheel acc|: mean absolute value over both wheels

The forward acceleration is the accelerometer's y (forward) axis with gravity removed
using the logged pitch, ``acc_y cos(pitch) - acc_z sin(pitch)``. Integrating it drifts
(a 0.3 deg pitch error leaks 0.05 m/s^2 of gravity), so the integrated velocity is
high-pass filtered at ``--highpass`` Hz; slower velocity changes are not seen.

Usage:
    uv run data_analysis/evaluate_runs.py LOG.csv [LOG.csv ...]
    uv run data_analysis/evaluate_runs.py LOG.csv --runs 6          # expect 6 runs
    uv run data_analysis/evaluate_runs.py LOG.csv --csv metrics.csv # also save the table
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Scales of the normalized log fields (POLICY_* in software_stack/models/policy_weights.h)
MAX_LINEAR_COMMAND = 0.3  # m/s
MAX_ANGULAR_COMMAND = 1.5  # rad/s
WHEEL_NO_LOAD_SPEED = 19.4  # rad/s

# (column, unit)
METRICS = {
    "lin MAE": "m/s",
    "ang MAE": "rad/s",
    "pitch rate MAE": "rad/s",
    "|acc y|": "m/s²",
    "|wheel vel|": "rad/s",
    "|action|": "",
    "|wheel acc|": "rad/s²",
}


def split_runs(df: pd.DataFrame) -> list[pd.DataFrame]:
    run_id = (df["time_s"] == 0).cumsum()
    return [run.reset_index(drop=True) for _, run in df.groupby(run_id)]


def forward_acceleration(run: pd.DataFrame) -> np.ndarray:
    """Forward acceleration in m/s^2: accelerometer y with gravity removed."""
    pitch = run["pitch"].to_numpy()
    return run["acc_y"].to_numpy() * np.cos(pitch) - run["acc_z"].to_numpy() * np.sin(pitch)


def forward_velocity(run: pd.DataFrame, highpass_hz: float) -> np.ndarray:
    """Forward velocity in m/s from the accelerometer, drift removed by a high-pass."""
    t, acc = run["time_s"].to_numpy(), forward_acceleration(run)
    v = np.concatenate([[0.0], np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(t))])
    rate = 1.0 / np.median(np.diff(t))
    b, a = butter(1, highpass_hz, "highpass", fs=rate)
    return filtfilt(b, a, v)


def evaluate_run(run: pd.DataFrame, trim_s: float, highpass_hz: float) -> dict[str, float]:
    t = run["time_s"].to_numpy()
    wheels = run[["left_wheel_velocity", "right_wheel_velocity"]].to_numpy() * WHEEL_NO_LOAD_SPEED
    wheel_acc = np.gradient(wheels, t, axis=0)
    v = forward_velocity(run, highpass_hz)

    # Velocity and accelerations are computed on the whole run so the filter and
    # derivative edges fall into the trimmed parts
    keep = (t >= t[0] + trim_s) & (t <= t[-1] - trim_s)
    run = run[keep]
    v_cmd = run["v_cmd"].to_numpy() * MAX_LINEAR_COMMAND
    omega_cmd = run["omega_cmd"].to_numpy() * MAX_ANGULAR_COMMAND
    actions = run[["action_left", "action_right"]].to_numpy()
    return {
        "duration": t[keep][-1] - t[keep][0],
        "lin MAE": np.mean(np.abs(v[keep] - v_cmd)),
        "ang MAE": np.mean(np.abs(run["yaw_rate"].to_numpy() - omega_cmd)),
        "pitch rate MAE": np.mean(np.abs(run["pitch_rate"].to_numpy())),
        "|acc y|": np.mean(np.abs(forward_acceleration(run))),
        "|wheel vel|": np.mean(np.abs(wheels[keep])),
        "|action|": np.mean(np.abs(actions)),
        "|wheel acc|": np.mean(np.abs(wheel_acc[keep])),
    }


def evaluate_log(path: Path, expected_runs: int, trim_s: float, highpass_hz: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    runs = split_runs(df)
    if len(runs) != expected_runs:
        raise ValueError(f"{len(runs)} runs (time_s == 0 rows), expected {expected_runs}")
    rows = [evaluate_run(run, trim_s, highpass_hz) for run in runs]
    table = pd.DataFrame(rows, index=[f"run {i}" for i in range(1, len(rows) + 1)])
    table.loc["average"] = table.mean()
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="robot state logs (CSV)")
    parser.add_argument("--runs", type=int, default=3, help="runs expected per log (default 3)")
    parser.add_argument("--trim", type=float, default=0.5, help="seconds cut from each run's start and end (default 0.5)")
    parser.add_argument("--highpass", type=float, default=0.1, help="velocity drift high-pass cutoff in Hz (default 0.1)")
    parser.add_argument("--csv", type=Path, help="also write all tables to this CSV")
    args = parser.parse_args()

    tables = {}
    for path in args.logs:
        try:
            tables[path.stem] = table = evaluate_log(path, args.runs, args.trim, args.highpass)
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            continue
        # Escape the | of |x| so it doesn't split the column when pasted into Markdown
        headers = ["", "duration [s]"] + [
            (f"{m} [{u}]" if u else m).replace("|", r"\|") for m, u in METRICS.items()
        ]
        print(f"\n## {path.name}\n")
        print(table.to_markdown(headers=headers, floatfmt=".4f"))

    if args.csv and tables:
        pd.concat(tables, names=["log", "run"]).to_csv(args.csv, float_format="%.6f")
        print(f"\n{args.csv}")
    return 0 if len(tables) == len(args.logs) else 1


if __name__ == "__main__":
    sys.exit(main())
