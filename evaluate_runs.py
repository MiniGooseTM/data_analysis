# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas>=2", "tabulate"]
# ///
"""Quantitative metrics for each run of a robot state log, per experiment.

A log (e.g. ``experiments/runs/goose_exp_default_forward_3runs.csv``) holds several runs
back to back; ``step`` restarts at 0 for each. The experiment (``balancing``, ``forward``,
``spin`` or ``slope``) is taken from the file name unless ``--experiment`` is given; it
selects which metrics are reported:

- all: duration, fell, pitch MAE / RMSE / max (from upright), pitch rate MAE / RMSE, mean
  |action|, action saturation (share of |a| >= 0.98), action smoothness (mean
  |a - previous a|), mean |wheel acceleration|
- balancing: lin / ang MAE and RMSE (the command is 0), position drift, heading drift
- forward: lin MAE / RMSE / steady-state error / rise time / overshoot, ang MAE / RMSE,
  distance
- spin: ang MAE / RMSE / steady-state error / rise time / overshoot, lin MAE / RMSE,
  position drift
- slope: lin / ang MAE, RMSE and steady-state errors, distance

Every metric is one number per run; the tables give each run and the mean and standard
deviation across runs.

The forward velocity on the robot is wheel odometry, ``r (mean wheel speed - pitch rate)``,
so it misses wheel slip; a simulated run (from ``simulate_log.py``) carries its true velocity
in ``v_true``, which is used instead. Commands are scaled to m/s and rad/s with the limits
of the policy that ran (``--policy`` reads them from its policy.bin).

Each run is trimmed by ``--trim`` seconds at both ends (the robot settling after start and
being stopped). Steady-state errors are taken over the stretches with a constant, nonzero
command, without their first ``--settle`` seconds (always 0 for slope, where the command is
held while the robot drives onto the ramp); rise time (to 90 %) and overshoot are
averaged over the command steps. A run fell if |pitch| reached 0.5 rad in its last 0.2 s.

Usage:
    uv run data_analysis/evaluate_runs.py LOG.csv [LOG.csv ...]
    uv run data_analysis/evaluate_runs.py LOG.csv --policy software_stack/models_share/default/policy_baseline.bin
    uv run data_analysis/evaluate_runs.py LOG.csv --experiment spin --runs 1,2,5
    uv run data_analysis/evaluate_runs.py LOG.csv --csv metrics.csv     # also save the table
"""

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = ("balancing", "forward", "spin", "slope")
WHEEL_RADIUS = 0.0275  # m, wheel collision cylinder in software_stack's goose.xml
# Sign of the pitch rate in the odometry: ground speed = r (wheel speed + sign * pitch rate),
# the wheel encoders measuring relative to the chassis. Checked against the true velocity
# in simulation: 0.005 m/s RMSE over a forward run (0.012 with +1, 0.007 without the term).
ODOMETRY_PITCH_SIGN = -1.0
FALL_PITCH = 0.5  # rad; the firmware cuts the motors at 0.6
FALL_WINDOW_S = 0.2
SATURATION = 0.98
RISE_FRACTION = 0.9
# Settle time per experiment, in place of --settle. On the slope the command is held while the
# robot drives onto the ramp, so command changes don't mark its transients (and the manually
# driven runs flicker between commands): every sample of a nonzero command counts.
EXPERIMENT_SETTLE_S = {"slope": 0.0}

COMMON_METRICS = [
    "duration_s", "fell", "pitch_mae_deg", "pitch_rmse_deg", "pitch_max_abs_deg",
    "pitch_rate_mae", "pitch_rate_rmse", "action_abs_mean", "action_saturation",
    "action_smoothness", "wheel_acc_abs_mean",
]
EXPERIMENT_METRICS = {
    "balancing": [
        "lin_mae", "lin_rmse", "ang_mae", "ang_rmse", "position_drift_m", "heading_drift_rad",
    ],
    "forward": [
        "lin_mae", "lin_rmse", "lin_ss_error", "lin_rise_time_s", "lin_overshoot", "ang_mae",
        "ang_rmse", "distance_m",
    ],
    "spin": [
        "ang_mae", "ang_rmse", "ang_ss_error", "ang_rise_time_s", "ang_overshoot", "lin_mae",
        "lin_rmse", "position_drift_m",
    ],
    "slope": [
        "lin_mae", "lin_rmse", "ang_mae", "ang_rmse", "lin_ss_error", "ang_ss_error",
        "distance_m",
    ],
}
UNITS = {
    "duration_s": "s", "pitch_mae_deg": "deg", "pitch_rmse_deg": "deg",
    "pitch_max_abs_deg": "deg", "pitch_rate_mae": "rad/s", "pitch_rate_rmse": "rad/s",
    "wheel_acc_abs_mean": "rad/s²",
    "lin_mae": "m/s", "lin_rmse": "m/s", "lin_ss_error": "m/s", "lin_rise_time_s": "s",
    "ang_mae": "rad/s", "ang_rmse": "rad/s", "ang_ss_error": "rad/s", "ang_rise_time_s": "s",
    "position_drift_m": "m", "heading_drift_rad": "rad", "distance_m": "m",
}


def metric_names(experiment: str) -> list[str]:
    return COMMON_METRICS + EXPERIMENT_METRICS[experiment]


@dataclass(frozen=True)
class Scales:
    """Physical units of the normalized log fields, as exported with the policy."""

    max_linear_command: float = 0.4  # m/s
    max_angular_command: float = 2.5  # rad/s
    wheel_no_load_speed: float = 19.4  # rad/s

    # Header of a policy.bin (export.py in software_stack): magic, version, sizes and
    # shapes, frame field CRC, then control period, max linear and angular command, half
    # track, wheel no-load speed, tilt limit, name.
    _HEADER = struct.Struct("<4sHHIHHHHHHIffffff44s")

    @classmethod
    def from_policy(cls, path: Path) -> "Scales":
        fields = cls._HEADER.unpack_from(Path(path).read_bytes())
        if fields[0] != b"GPOL":
            raise ValueError(f"{path}: not a policy file")
        _, max_linear, max_angular, _, no_load_speed, _ = fields[11:17]
        return cls(max_linear, max_angular, no_load_speed)


def experiment_of(path: Path) -> str:
    """The experiment named in a log's file name (``..._forward_3runs.csv``)."""
    found = re.findall(rf"(?:^|_)({'|'.join(EXPERIMENTS)})(?=_|$)", Path(path).stem)
    if len(set(found)) != 1:
        raise ValueError(f"{Path(path).name}: can't tell the experiment, give --experiment")
    return found[0]


def read_robot_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"step", "time_s", "v_cmd", "omega_cmd", "pitch"} <= set(df.columns):
        raise ValueError(f"not a robot state log: {','.join(df.columns)}")
    # A browser that reconnected mid-log may have appended a second header line
    return df[pd.to_numeric(df["step"], errors="coerce").notna()].apply(pd.to_numeric)


def split_runs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """One frame per run, cut where `step` goes back down."""
    run_id = (df["step"].diff() < 0).cumsum()
    return [run.reset_index(drop=True) for _, run in df.groupby(run_id)]


def select_runs(runs: list[pd.DataFrame], numbers: list[int] | None) -> dict[int, pd.DataFrame]:
    """The runs with the given 1-based numbers (all if None)."""
    if numbers is None:
        return dict(enumerate(runs, start=1))
    missing = [n for n in numbers if not 1 <= n <= len(runs)]
    if missing:
        raise ValueError(f"no run {missing} (the log has {len(runs)})")
    return {n: runs[n - 1] for n in numbers}


def prepare_signals(run: pd.DataFrame, scales: Scales) -> pd.DataFrame:
    """Physical signals of a robot or simulated run: time since start, commands, velocities."""
    t = run["time_s"] - run["time_s"].iloc[0] if "time_s" in run else run["t"]
    wheels = run[["left_wheel_velocity", "right_wheel_velocity"]] * scales.wheel_no_load_speed
    if "v_true" in run:
        v = run["v_true"]
    else:
        v = WHEEL_RADIUS * (wheels.mean(axis=1) + ODOMETRY_PITCH_SIGN * run["pitch_rate"])
    return pd.DataFrame({
        "t": t.to_numpy(),
        "v_cmd": run["v_cmd"].to_numpy() * scales.max_linear_command,
        "w_cmd": run["omega_cmd"].to_numpy() * scales.max_angular_command,
        "v": v.to_numpy(),
        "w": run["yaw_rate"].to_numpy(),
        "pitch": run["pitch"].to_numpy(),
        "pitch_rate": run["pitch_rate"].to_numpy(),
        "wheel_left": wheels.iloc[:, 0].to_numpy(),
        "wheel_right": wheels.iloc[:, 1].to_numpy(),
        "action_left": run["action_left"].to_numpy(),
        "action_right": run["action_right"].to_numpy(),
        "previous_action_left": run["previous_action_left"].to_numpy(),
        "previous_action_right": run["previous_action_right"].to_numpy(),
    })


def integral(x: np.ndarray, t: np.ndarray) -> float:
    return float(np.sum(0.5 * (x[1:] + x[:-1]) * np.diff(t))) if len(t) > 1 else 0.0


def command_segments(command: np.ndarray) -> list[tuple[int, int]]:
    """[begin, end) index ranges of constant command."""
    changes = np.flatnonzero(np.diff(command) != 0) + 1
    bounds = np.concatenate([[0], changes, [len(command)]])
    return list(zip(bounds[:-1], bounds[1:]))


def tracking(y: np.ndarray, target: np.ndarray, t: np.ndarray, keep: np.ndarray, settle_s: float) -> dict:
    """MAE, RMSE, steady-state error, rise time and overshoot of `y` following `target`,
    counted over the samples in `keep`."""
    error = (y - target)[keep]
    steady, rise_times, overshoots = [], [], []
    segments = command_segments(target)
    for k, (begin, end) in enumerate(segments):
        in_segment = np.zeros(len(t), bool)
        in_segment[begin:end] = True
        window = in_segment & keep
        if not window.any():
            continue
        value = target[begin]
        if value != 0 and t[end - 1] - t[begin] >= settle_s:
            settled = window & (t >= t[begin] + settle_s)
            steady.append(np.abs(y[settled] - value))
        # Step response: from the previous command to this one, inside this segment
        if k == 0 or not keep[begin]:
            continue
        previous = target[begin - 1]
        step = value - previous
        progress = (y[window] - previous) / step
        reached = np.flatnonzero(progress >= RISE_FRACTION)
        rise_times.append(t[window][reached[0]] - t[begin] if reached.size else np.nan)
        overshoots.append(max(0.0, progress.max() - 1.0))
    steady = np.concatenate(steady) if steady else np.array([])
    return {
        "mae": np.mean(np.abs(error)),
        "rmse": np.sqrt(np.mean(error**2)),
        "ss_error": np.mean(steady) if steady.size else np.nan,
        "rise_time_s": np.nanmean(rise_times) if np.isfinite(rise_times).any() else np.nan,
        "overshoot": np.mean(overshoots) if overshoots else np.nan,
    }


def evaluate_run(
    run: pd.DataFrame,
    experiment: str,
    scales: Scales,
    trim_s: float = 0.5,
    settle_s: float = 1.0,
    fell: bool | None = None,
) -> dict[str, float]:
    """The metrics of `experiment` for one run of a robot log or a simulated run. `fell`
    overrides the detection from the final pitch (a simulation knows when it fell)."""
    s = prepare_signals(run, scales)
    t, pitch = s["t"].to_numpy(), s["pitch"].to_numpy()
    if fell is None:
        fell = bool(np.abs(pitch[t >= t[-1] - FALL_WINDOW_S]).max() >= FALL_PITCH)
    wheels = s[["wheel_left", "wheel_right"]].to_numpy()
    # Derivatives on the whole run, so their edges fall into the trimmed parts
    wheel_acc = np.gradient(wheels, t, axis=0) if len(t) > 1 else np.zeros_like(wheels)
    keep = (t >= t[0] + trim_s) & (t <= t[-1] - trim_s)
    if keep.sum() < 2:
        raise ValueError(f"run of {t[-1] - t[0]:.1f} s is too short for --trim {trim_s}")

    tk = t[keep]
    actions = s[["action_left", "action_right"]].to_numpy()[keep]
    previous = s[["previous_action_left", "previous_action_right"]].to_numpy()[keep]
    settle_s = EXPERIMENT_SETTLE_S.get(experiment, settle_s)
    lin = tracking(s["v"].to_numpy(), s["v_cmd"].to_numpy(), t, keep, settle_s)
    ang = tracking(s["w"].to_numpy(), s["w_cmd"].to_numpy(), t, keep, settle_s)
    distance = integral(s["v"].to_numpy()[keep], tk)
    # Pitch and pitch rate are errors from upright and still (target 0)
    pitch_rate = s["pitch_rate"].to_numpy()[keep]
    values = {
        "duration_s": t[-1] - t[0],
        "fell": float(fell),
        "pitch_mae_deg": np.degrees(np.mean(np.abs(pitch[keep]))),
        "pitch_rmse_deg": np.degrees(np.sqrt(np.mean(pitch[keep] ** 2))),
        "pitch_max_abs_deg": np.degrees(np.abs(pitch[keep]).max()),
        "pitch_rate_mae": np.mean(np.abs(pitch_rate)),
        "pitch_rate_rmse": np.sqrt(np.mean(pitch_rate**2)),
        "action_abs_mean": np.mean(np.abs(actions)),
        "action_saturation": np.mean(np.abs(actions) >= SATURATION),
        "action_smoothness": np.mean(np.abs(actions - previous)),
        "wheel_acc_abs_mean": np.mean(np.abs(wheel_acc[keep])),
        "distance_m": distance,
        "position_drift_m": abs(distance),
        "heading_drift_rad": abs(integral(s["w"].to_numpy()[keep], tk)),
        **{f"lin_{k}": v for k, v in lin.items()},
        **{f"ang_{k}": v for k, v in ang.items()},
    }
    return {name: float(values[name]) for name in metric_names(experiment)}


def evaluate_runs(
    runs: dict[int, pd.DataFrame], experiment: str, scales: Scales, **options
) -> pd.DataFrame:
    """One row of metrics per run, indexed by run number."""
    rows = {n: evaluate_run(run, experiment, scales, **options) for n, run in runs.items()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "run"
    return table


def evaluate_log(
    path: Path,
    scales: Scales,
    experiment: str | None = None,
    runs: list[int] | None = None,
    **options,
) -> tuple[str, pd.DataFrame]:
    """(experiment, metrics per run) of a robot state log."""
    experiment = experiment or experiment_of(path)
    selected = select_runs(split_runs(read_robot_log(path)), runs)
    return experiment, evaluate_runs(selected, experiment, scales, **options)


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Metrics as rows, runs as columns, plus their mean and standard deviation, as text."""
    out = table.T.rename(columns=lambda n: f"run {n}")
    out["mean"] = table.mean()
    out["std"] = table.std(ddof=1) if len(table) > 1 else np.nan
    out = out.map(lambda v: "–" if np.isnan(v) else f"{v:.4f}")
    # Falls: 0 or 1 per run, and their number in place of a mean
    out.loc["fell"] = [*(str(int(f)) for f in table["fell"]), str(int(table["fell"].sum())), ""]
    out = out.rename(index={"fell": "falls"})
    out.index = [f"{m} [{UNITS[m]}]" if m in UNITS else m for m in out.index]
    return out


def parse_runs(text: str) -> list[int]:
    return [int(n) for n in text.split(",") if n.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="robot state logs (CSV)")
    parser.add_argument("--experiment", choices=EXPERIMENTS, help="default: from the file name")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--policy", type=Path, help="policy.bin that ran, for its command limits")
    scale.add_argument(
        "--scales", type=float, nargs=2, metavar=("LIN", "ANG"),
        help="max linear (m/s) and angular (rad/s) command (default 0.4 2.5)",
    )
    parser.add_argument("--runs", type=parse_runs, help="runs to evaluate, 1-based, e.g. 1,2,5 (default all)")
    parser.add_argument("--trim", type=float, default=0.5, help="seconds cut from each run's start and end (default 0.5)")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds after a command change left out of the steady-state error (default 1; always 0 for slope)")
    parser.add_argument("--csv", type=Path, help="also write all tables to this CSV")
    args = parser.parse_args()

    if args.policy:
        scales = Scales.from_policy(args.policy)
    elif args.scales:
        scales = Scales(*args.scales)
    else:
        scales = Scales()

    tables = {}
    for path in args.logs:
        try:
            experiment, table = evaluate_log(
                path, scales, args.experiment, args.runs, trim_s=args.trim, settle_s=args.settle
            )
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            continue
        tables[path.stem] = table.assign(experiment=experiment)
        print(f"\n## {path.name} ({experiment})\n")
        print(summary_table(table).to_markdown(
            colalign=["left"] + ["right"] * (len(table) + 2), disable_numparse=True
        ))

    if args.csv and tables:
        pd.concat(tables, names=["log", "run"]).to_csv(args.csv, float_format="%.6f")
        print(f"\n{args.csv}")
    return 0 if len(tables) == len(args.logs) else 1


if __name__ == "__main__":
    sys.exit(main())
