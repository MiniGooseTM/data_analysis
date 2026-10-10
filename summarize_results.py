# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2", "tabulate"]
# ///
"""Summarize the metric CSVs written by ``evaluate_runs.py --csv``: one row per file,
holding the average of its runs.

Reads every ``*.csv`` in a folder (default ``experiments/results/``) and writes the
summary next to it (default ``experiments/results_summary.csv``), so a rerun doesn't
read its own output.

Usage:
    uv run data_analysis/summarize_results.py
    uv run data_analysis/summarize_results.py --dir other/results --out summary.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def summarize(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    runs = df[df["run"].astype(str).str.startswith("run")]
    if runs.empty:
        raise ValueError("no 'run N' rows")
    means = runs.drop(columns=["log", "run"], errors="ignore").mean()
    return pd.concat([pd.Series({"runs": len(runs)}), means]).rename(path.stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dir", type=Path, default=HERE / "experiments" / "results",
                        help="folder with the metric CSVs (default experiments/results/)")
    parser.add_argument("--out", type=Path, help="summary CSV (default <dir>_summary.csv)")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.dir.glob("*.csv")):
        try:
            rows.append(summarize(path))
        except (OSError, ValueError, KeyError) as e:
            print(f"{path}: {e}", file=sys.stderr)
    if not rows:
        print(f"{args.dir}: no metric CSVs", file=sys.stderr)
        return 1

    summary = pd.DataFrame(rows).astype({"runs": int})
    summary.index.name = "log"
    out = args.out or args.dir.with_name(args.dir.name + "_summary.csv")
    summary.to_csv(out, float_format="%.6f")
    # Escape the | of |x| so it doesn't split the column when pasted into Markdown
    print(summary.to_markdown(headers=["log", *(c.replace("|", r"\|") for c in summary.columns)],
                              floatfmt=["", ".0f"] + [".4f"] * (len(summary.columns) - 1)))
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
