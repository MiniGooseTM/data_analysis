# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["matplotlib>=3.7", "pandas>=2", "tabulate", "self-balancing-robot"]
#
# [tool.uv.sources]
# self-balancing-robot = { path = "../software_stack", editable = true }
# ///
"""Full quantitative evaluation: every policy's robot runs and their replays in simulation.

A config (default ``experiments/evaluation.toml``) names each policy's ``policy.bin`` and,
per experiment (balancing, forward, spin, slope), the robot log it was run in. For every
selected run of every log the metrics of ``evaluate_runs.py`` are computed for the robot,
and the run is replayed in simulation as by ``simulate_log.py`` (same policy, same command
sequence, same start pitch) and scored the same way. The simulated velocity is the true
one, the robot's is wheel odometry.

The config can name several simulations, each a curriculum stage and the seeds to replay
with, e.g. ``clean`` (stage "balance": no noise, latency or randomized physics) and ``noisy``
(stage "commands 0.4 m/s": the sensor noise and latency the policies trained with). Each one
is a source next to ``real`` in the results.

Writes to ``experiments/results/`` (or ``--out``):

- ``evaluation_runs.csv``: one row per policy, experiment, source (real or a simulation),
  run and seed
- ``evaluation_summary.csv``: per policy, experiment and metric the mean and standard
  deviation over the runs of every source (simulations averaged over their seeds first), and
  each simulation's minus real gap (paired by run, as each replay follows that run's
  commands)
- ``evaluation_summary.md``: those as Markdown tables, one per experiment

Usage:
    uv run data_analysis/run_evaluation.py
    uv run data_analysis/run_evaluation.py other.toml --out results/
    uv run data_analysis/run_evaluation.py --policy default --experiment forward spin
    uv run data_analysis/run_evaluation.py --no-sim                    # robot metrics only
"""

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_runs import (
    EXPERIMENTS,
    Scales,
    evaluate_run,
    evaluate_runs,
    metric_names,
    read_robot_log,
    select_runs,
    split_runs,
)

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "experiments" / "evaluation.toml"
EXPECTED_RUNS = 3
KEYS = ["policy", "experiment", "source", "run", "seed"]


@dataclass(frozen=True)
class Task:
    policy: str
    experiment: str
    model: Path
    log: Path
    runs: list[int] | None


@dataclass(frozen=True)
class Simulation:
    name: str  # its source label in the results
    stage: str | int
    seeds: list[int]


@dataclass(frozen=True)
class Config:
    simulations: list[Simulation]
    skip: set[str]
    trim_s: float
    settle_s: float
    tasks: list[Task]


def load_config(path: Path) -> Config:
    raw = tomllib.loads(path.read_text())
    simulation, evaluation = raw.get("simulation", {}), raw.get("evaluation", {})
    tasks, problems = [], []
    for policy, entries in raw.get("policies", {}).items():
        entries = dict(entries)
        if "model" not in entries:
            problems.append(f"policies.{policy}: no model")
            continue
        model = path.parent / entries.pop("model")
        if not model.is_file():
            problems.append(f"policies.{policy}: no model {model}")
        for experiment, entry in entries.items():
            if experiment not in EXPERIMENTS:
                problems.append(f"policies.{policy}.{experiment}: not one of {', '.join(EXPERIMENTS)}")
                continue
            log, runs = (entry["log"], entry.get("runs")) if isinstance(entry, dict) else (entry, None)
            log = path.parent / log
            if not log.is_file():
                problems.append(f"policies.{policy}.{experiment}: no log {log}")
            tasks.append(Task(policy, experiment, model, log, runs))
    skip = set(simulation.get("skip", []))
    problems += [f"simulation.skip: no experiment {e!r}" for e in skip - set(EXPERIMENTS)]
    simulations = []
    for name, entry in simulation.items():
        if not isinstance(entry, dict):
            continue
        if name == "real" or "stage" not in entry:
            problems.append(f"simulation.{name}: needs a stage, and can't be called real")
            continue
        simulations.append(Simulation(name, entry["stage"], list(entry.get("seeds", [0]))))
    if problems:
        raise ValueError("\n".join(problems))
    return Config(
        simulations=simulations or [Simulation("sim", "balance", [0])],
        skip=skip,
        trim_s=float(evaluation.get("trim_s", 0.5)),
        settle_s=float(evaluation.get("settle_s", 1.0)),
        tasks=tasks,
    )


def make_replay(simulations: list[Simulation]):
    """A log replay, its policy and stage set per task (`use`, `in_stage`); checks that the
    simulations' stages exist. Imported here so that `--no-sim` doesn't load JAX."""
    from self_balancing_robot.actor import Actor
    from self_balancing_robot.envs.balance_env import SelfBalancingRobotEnv
    from simulate_log import LogReplay

    env = SelfBalancingRobotEnv(fixed_difficulty=True, max_episode_steps=2**31 - 1)
    replay = LogReplay(env, Actor([]), 0.0)
    for simulation in simulations:
        stage = replay.in_stage(simulation.stage)
        print(f'simulation {simulation.name}: stage "{stage}", seeds {simulation.seeds}')
    return replay


def evaluate_task(task: Task, config: Config, replay) -> pd.DataFrame:
    """Rows of metrics for the robot runs of `task` and, with a `replay`, their simulations."""
    scales = Scales.from_policy(task.model)
    all_runs = split_runs(read_robot_log(task.log))
    if task.runs is None and len(all_runs) != EXPECTED_RUNS:
        print(f"  warning: {task.log.name} has {len(all_runs)} runs, expected {EXPECTED_RUNS} "
              "(give runs = [...] to choose)", file=sys.stderr)
    runs = select_runs(all_runs, task.runs)
    options = {"trim_s": config.trim_s, "settle_s": config.settle_s}

    real = evaluate_runs(runs, task.experiment, scales, **options).reset_index()
    tables = [real.assign(source="real", seed=np.nan)]
    summary = f"  {task.policy} {task.experiment}: {len(runs)} robot runs ({int(real['fell'].sum())} fell)"
    if replay is not None:
        from self_balancing_robot.actor import Actor
        from simulate_log import logged_commands

        replay.use(Actor.load(task.model), 0.0)
        for simulation in config.simulations:
            replay.in_stage(simulation.stage)
            rows, falls = [], 0
            for number, run in runs.items():
                for seed in simulation.seeds:
                    sim, fell, _ = replay.run(logged_commands(run), float(run["pitch"].iloc[0]), seed)
                    falls += fell
                    metrics = evaluate_run(sim, task.experiment, scales, fell=fell, **options)
                    rows.append({"run": number, "seed": seed, **metrics})
            tables.append(pd.DataFrame(rows).assign(source=simulation.name))
            summary += f", {simulation.name} {len(rows)} ({falls} fell)"
    print(summary)
    table = pd.concat(tables, ignore_index=True)
    return table.assign(
        policy=task.policy,
        experiment=task.experiment,
        log=task.log.name,
        model=f"{task.model.parent.name}/{task.model.name}",
    )


def summarize(results: pd.DataFrame, simulations: list[str]) -> pd.DataFrame:
    """Per policy, experiment and metric: the mean and std over the runs of the robot and of
    each simulation, and the mean and std of each simulation's per-run minus real gap."""
    rows = []
    for (policy, experiment), group in results.groupby(["policy", "experiment"], sort=False):
        metrics = metric_names(experiment)
        real = group[group["source"] == "real"].set_index("run")[metrics]
        # Seeds are repeated simulations of the same run: average them first
        sims = {
            name: group[group["source"] == name].groupby("run")[metrics].mean()
            for name in simulations
        }
        for metric in metrics:
            row = {
                "policy": policy,
                "experiment": experiment,
                "metric": metric,
                "runs": len(real),
                "real_mean": real[metric].mean(),
                "real_std": real[metric].std(),
            }
            for name, sim in sims.items():
                gap = (sim[metric] - real[metric]).dropna() if not sim.empty else pd.Series(dtype=float)
                row |= {
                    f"{name}_mean": sim[metric].mean() if not sim.empty else np.nan,
                    f"{name}_std": sim[metric].std() if not sim.empty else np.nan,
                    f"{name}_gap_mean": gap.mean(),
                    f"{name}_gap_std": gap.std(),
                }
            if metric == "fell":
                # The mean is the fall rate; the spread of yes/no outcomes means nothing
                row |= {k: np.nan for k in row if k.endswith("_std")}
            rows.append(row)
    return pd.DataFrame(rows)


def fall_counts(results: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """Number of falls per policy, experiment and source, over all of a simulation's
    replays (runs x seeds)."""
    counts = results.groupby(["policy", "experiment", "source"])["fell"].sum()
    return {key: str(int(count)) for key, count in counts.items()}


def mean_std(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "–"
    return f"{mean:.3g}" + (f" ± {std:.2g}" if not np.isnan(std) else "")


def markdown(summary: pd.DataFrame, results: pd.DataFrame, simulations: list[Simulation]) -> str:
    """One table per experiment: metrics as rows, real and simulated per policy as columns."""
    falls = fall_counts(results)
    described = "; ".join(
        f'{s.name}: stage "{s.stage}", {len(s.seeds)} seed{"s" * (len(s.seeds) > 1)}'
        for s in simulations
    )
    parts = ["# Evaluation summary\n", "Every metric is one number per run (an MAE or RMSE "
             "is over that run's samples; pitch and pitch rate against upright and still). "
             "Cells are the mean ± standard deviation of it across runs, simulations averaged "
             "over their seeds first. Falls: the number of robot runs, or of a simulation's "
             "replays (runs × seeds), that fell. Robot velocity from wheel odometry, "
             f"simulated velocity true. Simulations: {described}.\n"]
    for experiment in EXPERIMENTS:
        group = summary[summary["experiment"] == experiment]
        if group.empty:
            continue
        columns = {}
        for policy, rows in group.groupby("policy", sort=False):
            rows = rows.set_index("metric")

            def column(source: str) -> list[str]:
                cells = [mean_std(m, s) for m, s in zip(rows[f"{source}_mean"], rows[f"{source}_std"])]
                fell = falls.get((policy, experiment, source), "–")
                return [fell if metric == "fell" else c for metric, c in zip(rows.index, cells)]

            columns[f"{policy} real (n={int(rows['runs'].iloc[0])})"] = column("real")
            for simulation in simulations:
                means = rows.get(f"{simulation.name}_mean")
                if means is not None and means.notna().any():
                    columns[f"{policy} {simulation.name}"] = column(simulation.name)
        table = pd.DataFrame(columns, index=metric_names(experiment)).rename(index={"fell": "falls"})
        parts += [f"\n## {experiment}\n", table.to_markdown(), ""]
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG,
                        help="evaluation config (default experiments/evaluation.toml)")
    parser.add_argument("--out", type=Path, default=HERE / "experiments" / "results",
                        help="output folder (default experiments/results/)")
    parser.add_argument("--policy", nargs="+", help="only these policies (config names)")
    parser.add_argument("--experiment", nargs="+", choices=EXPERIMENTS, help="only these experiments")
    parser.add_argument("--no-sim", action="store_true", help="evaluate the robot logs only")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as e:
        print(f"{args.config}: {e}", file=sys.stderr)
        return 1
    tasks = [
        t for t in config.tasks
        if (not args.policy or t.policy in args.policy)
        and (not args.experiment or t.experiment in args.experiment)
    ]
    if not tasks:
        print("nothing to evaluate", file=sys.stderr)
        return 1

    simulate = not args.no_sim and any(t.experiment not in config.skip for t in tasks)
    try:
        replay = make_replay(config.simulations) if simulate else None
    except ValueError as e:
        print(f"{args.config}: {e}", file=sys.stderr)
        return 1
    tables, failed = [], 0
    for task in tasks:
        try:
            tables.append(evaluate_task(task, config, None if task.experiment in config.skip else replay))
        except (OSError, ValueError, KeyError) as e:
            print(f"  {task.policy} {task.experiment}: {e}", file=sys.stderr)
            failed += 1
    if not tables:
        return 1

    results = pd.concat(tables, ignore_index=True)
    metrics = [m for e in EXPERIMENTS for m in metric_names(e)]
    metrics = list(dict.fromkeys(m for m in metrics if m in results))
    results = results[KEYS + ["log", "model"] + metrics].astype({"seed": "Int64"})
    names = [s.name for s in config.simulations]
    summary = summarize(results, names)

    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "evaluation_runs.csv": lambda p: results.to_csv(p, index=False, float_format="%.6f"),
        "evaluation_summary.csv": lambda p: summary.to_csv(p, index=False, float_format="%.6f"),
        "evaluation_summary.md": lambda p: p.write_text(markdown(summary, results, config.simulations)),
    }
    print()
    for name, write in outputs.items():
        write(args.out / name)
        print(args.out / name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
