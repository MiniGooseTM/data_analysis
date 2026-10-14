# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["matplotlib>=3.7", "pandas>=2", "self-balancing-robot"]
#
# [tool.uv.sources]
# self-balancing-robot = { path = "../software_stack", editable = true }
# ///
"""Overlay a robot state log and its replay in simulation, run by run.

Each run of a robot state log (``goose_state_*.csv``) is plotted as by ``plot_logs.py``
(solid lines) and simulated as by ``simulate_log.py``, whose result is drawn dashed on
the same panels in the same colors. The simulated lines end early if the simulated robot
falls (the title says so).

Usage:
    uv run data_analysis/compare_log.py LOG.csv                         # show windows
    uv run data_analysis/compare_log.py LOG.csv --save                  # PNG next to the CSV
    uv run data_analysis/compare_log.py LOG.csv --save --out plots/     # PNG into plots/
    uv run data_analysis/compare_log.py LOG.csv --run 2                 # only the 2nd run
    uv run data_analysis/compare_log.py LOG.csv --stage -1 --seed 3     # noisy, randomized
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from plot_logs import ROBOT_PANELS, plot_panels, prepare_run
from simulate_log import add_simulation_arguments, make_replay, simulated_runs


def comparison_figures(path: Path, replay, args, label: str):
    for i, count, run, sim, fell, _ in simulated_runs(path, replay, args):
        robot, missing = prepare_run(run)
        title = (
            f"{path.name} - run {i}/{count}: robot {robot['t'].iloc[-1]:.1f} s"
            + (f" ({missing} steps missing)" if missing else "")
            + f" vs. simulated ({label}) {sim['t'].iloc[-1]:.1f} s"
        )
        if fell:
            title += ", fell"
        fig = plot_panels(
            robot, ROBOT_PANELS, title, "time since run start [s]",
            overlay=sim, styles=("robot", "simulation"),
        )
        yield f"run{i}_compare", fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="robot state logs (goose_state_*.csv)")
    add_simulation_arguments(parser)
    parser.add_argument("--save", action="store_true", help="write PNGs instead of showing windows")
    parser.add_argument("--out", type=Path, help="directory for the PNGs (default: next to the CSV)")
    parser.add_argument("--run", type=int, help="plot only this run (1-based)")
    args = parser.parse_args()

    replay, label = make_replay(args, parser)

    shown = 0
    for path in args.logs:
        try:
            figs = list(comparison_figures(path, replay, args, label))
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            continue
        for suffix, fig in figs:
            if args.save:
                out_dir = args.out or path.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{path.stem}_{suffix}.png"
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
