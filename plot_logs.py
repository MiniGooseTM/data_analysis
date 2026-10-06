# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.7", "pandas>=2"]
# ///
"""Plot CSV logs downloaded from the ESP32 web pages.

Two formats are recognised by their header:

- robot state log (``goose_state_*.csv``, "Download" in the robot page's state log):
  ``step,time_s,<OBS_FRAME_FIELDS>,action_left,action_right``, one row per control step.
  A file can hold several runs back to back; ``step`` restarts at 0 for each. Each run
  gets its own figure.
- PID test log (``pid_*.csv``, "Download CSV" in the pid_test page):
  ``t_ms,tilt_deg,setpoint_deg,rate_dps,p,i,d,u``.

Usage:
    uv run data_analysis/plot_logs.py LOG.csv [LOG.csv ...]           # show windows
    uv run data_analysis/plot_logs.py LOG.csv --save                  # PNG next to the CSV
    uv run data_analysis/plot_logs.py LOG.csv --save --out plots/     # PNG into plots/
    uv run data_analysis/plot_logs.py LOG.csv --run 2                 # only the 2nd run
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Categorical slots 1-3 of the validated default palette, always in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# (title, y label, [(column, legend label)], scale applied to the values)
ROBOT_PANELS = [
    ("Velocity command", "normalized", [("v_cmd", "linear"), ("omega_cmd", "angular")], 1.0),
    ("Tilt", "deg", [("pitch", "pitch (fwd lean < 0)"), ("roll", "roll")], np.degrees(1.0)),
    (
        "Body rates",
        "deg/s",
        [("pitch_rate", "pitch"), ("roll_rate", "roll"), ("yaw_rate", "yaw")],
        np.degrees(1.0),
    ),
    (
        "Wheel velocity",
        "normalized",
        [("left_wheel_velocity", "left"), ("right_wheel_velocity", "right")],
        1.0,
    ),
    ("Motor action", "[-1, 1]", [("action_left", "left"), ("action_right", "right")], 1.0),
]

PID_PANELS = [
    ("Tilt", "deg", [("tilt_deg", "tilt"), ("setpoint_deg", "setpoint")], 1.0),
    ("Tilt rate", "deg/s", [("rate_dps", "rate")], 1.0),
    ("PID terms", "", [("p", "P"), ("i", "I"), ("d", "D")], 1.0),
    ("Output", "u", [("u", "u")], 1.0),
]


def split_runs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Robot log: one frame per run, cut where `step` goes back down."""
    run_id = (df["step"].diff() < 0).cumsum()
    return [run.reset_index(drop=True) for _, run in df.groupby(run_id)]


def break_gaps(run: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Insert a NaN row after every skipped step so lines don't bridge dropped rows."""
    gaps = run["step"].diff().fillna(1) > 1
    missing = int((run["step"].diff() - 1)[gaps].sum())
    if not gaps.any():
        return run, 0
    breaks = pd.DataFrame({"t": run.loc[gaps, "t"] - 1e-9})
    return pd.concat([run, breaks]).sort_values("t", kind="stable").reset_index(drop=True), missing


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT, fontweight="bold")
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_panels(df: pd.DataFrame, panels, title: str, xlabel: str, overlay=None, styles=None):
    """One panel per entry of `panels` that has data in `df`. `overlay` is a second frame
    drawn dashed on top in the same series colors; `styles` names the solid and the dashed
    lines in a key at the bottom right."""
    panels = [p for p in panels if any(col in df for col, _ in p[2])]
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True, figsize=(11, 1.9 * len(panels) + 0.8), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, (panel_title, ylabel, series, scale) in zip(axes[:, 0], panels):
        style_axis(ax, panel_title, ylabel)
        for color, (col, label) in zip(SERIES, series):
            if col in df:
                ax.plot(df["t"], df[col] * scale, color=color, linewidth=1.5, label=label)
            if overlay is not None and col in overlay:
                ax.plot(overlay["t"], overlay[col] * scale, color=color, linewidth=1.5,
                        linestyle=(0, (4, 2)))
        if len(series) > 1:
            ax.legend(
                loc="upper right", fontsize=8, frameon=False, ncol=len(series),
                labelcolor=TEXT_SECONDARY, handlelength=1.5,
            )
    axes[-1, 0].set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    if styles:
        key = [
            Line2D([], [], color=TEXT_SECONDARY, linewidth=1.5, linestyle=style, label=name)
            for name, style in zip(styles, ("-", (0, (4, 2))))
        ]
        fig.legend(
            handles=key, loc="lower right", bbox_to_anchor=(0.995, 0.0), fontsize=8,
            frameon=False, ncol=len(key), labelcolor=TEXT_SECONDARY, handlelength=2.5,
        )
    return fig


def prepare_run(run: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Robot run with `t` (time since run start) and gaps broken; also the missing steps."""
    return break_gaps(run.assign(t=run["time_s"] - run["time_s"].iloc[0]))


def robot_figures(df: pd.DataFrame, name: str, only_run: int | None):
    runs = split_runs(df)
    for i, run in enumerate(runs, start=1):
        if only_run is not None and i != only_run:
            continue
        run, missing = prepare_run(run)
        duration = run["t"].iloc[-1]
        steps = int(run["step"].count())
        title = f"{name} - run {i}/{len(runs)}: {duration:.1f} s, {steps} steps"
        if missing:
            title += f", {missing} missing"
        yield f"run{i}", plot_panels(run, ROBOT_PANELS, title, "time since run start [s]")


def pid_figures(df: pd.DataFrame, name: str):
    df = df.assign(t=(df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0)
    title = f"{name}: {df['t'].iloc[-1]:.1f} s, {len(df)} samples"
    yield "", plot_panels(df, PID_PANELS, title, "time since recording start [s]")


def figures_for(path: Path, only_run: int | None):
    df = pd.read_csv(path)
    if {"step", "time_s"} <= set(df.columns):
        # A browser that reconnected mid-log may have appended a second header line
        df = df[pd.to_numeric(df["step"], errors="coerce").notna()].apply(pd.to_numeric)
        return robot_figures(df, path.name, only_run)
    if "t_ms" in df.columns:
        return pid_figures(df, path.name)
    raise ValueError(f"unrecognised header: {','.join(df.columns)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="CSV files downloaded from the robot")
    parser.add_argument("--save", action="store_true", help="write PNGs instead of showing windows")
    parser.add_argument("--out", type=Path, help="directory for the PNGs (default: next to the CSV)")
    parser.add_argument("--run", type=int, help="robot log: plot only this run (1-based)")
    args = parser.parse_args()

    shown = 0
    for path in args.logs:
        try:
            figs = list(figures_for(path, args.run))
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            continue
        for suffix, fig in figs:
            if args.save:
                out_dir = args.out or path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{path.stem}{'_' + suffix if suffix else ''}.png"
                fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(out)
            shown += 1
    if not shown:
        print("nothing plotted", file=sys.stderr)
        return 1
    if not args.save:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
