# Data analysis

Scripts for plotting logs downloaded from the robot's web pages and for comparing them with simulation. Run them from `mini_goose/` with [uv](https://docs.astral.sh/uv/); each script declares its own dependencies, so no setup is needed.

## Log formats

- **Robot state log** (`goose_state_*.csv`, "Download" in the state log on the robot page): `step,time_s,<11 observation fields>,action_left,action_right`, one row per 10 ms control step. The observation fields are the policy's input frame (`ACTOR_FRAME_FIELDS` in `software_stack`): commands normalized to [-1, 1], tilt in rad, body rates in rad/s, wheel velocities divided by the no-load speed, previous action. A file can hold several runs; `step` restarts at 0 for each.
- **PID test log** (`pid_*.csv`, "Download CSV" on the pid_test page): `t_ms,tilt_deg,setpoint_deg,rate_dps,p,i,d,u`.

## `plot_logs.py`: plot a log

One figure per run (robot log) or per file (PID log): velocity command, tilt, body rates, wheel velocity and motor action over time.

```bash
uv run data_analysis/plot_logs.py LOG.csv [LOG.csv ...]         # show windows
uv run data_analysis/plot_logs.py LOG.csv --save                # PNG next to the CSV
uv run data_analysis/plot_logs.py LOG.csv --save --out plots/   # PNG into plots/
uv run data_analysis/plot_logs.py LOG.csv --run 2               # only the 2nd run
```

## `simulate_log.py`: replay a robot log in simulation

Runs the policy in the MuJoCo environment of `../software_stack` with the same command sequence as each run in a robot state log, and plots the result with the same panels as `plot_logs.py`, so the two figures compare 1:1. The simulation starts at the run's first logged pitch, standing still, and ends early if the simulated robot falls (the title says so). With `--video` it also renders each simulated run to an MP4.

```bash
uv run data_analysis/simulate_log.py LOG.csv                    # show windows
uv run data_analysis/simulate_log.py LOG.csv --save             # <log>_run<N>_sim.png next to the CSV
uv run data_analysis/simulate_log.py LOG.csv --run 4 --stage -1 --seed 3
uv run data_analysis/simulate_log.py LOG.csv --run 4 --save --video   # also <log>_run4_sim.mp4
```

Options:

- `--model`: `policy.bin` or a training run directory (default `software_stack/models/policy.bin`).
- `--stage`: curriculum stage to simulate, by name or index. The default `"balance"` is a clean simulation (the logged commands are replayed in full in any stage; a stage's command range only limits the commands it draws itself); the `"commands ..."` stages add sensor noise and latency (`"commands 0.4 m/s"` the full amount, up to 2 steps), and `-1` (the last stage) also randomized mass, friction, CoM, pushes and slopes.
- `--seed`: seed for that noise and randomization.
- `--slope`: ground slope in rad (default 0, flat).
- `--save`, `--out`, `--run`: as in `plot_logs.py`.
- `--video`: also render each simulated run with MuJoCo's tracking camera to `<log>_run<N>_sim.mp4` (next to the CSV or into `--out`). Needs OpenGL: a display, or `MUJOCO_GL=egl` on a headless machine.

The first run installs the simulation dependencies (JAX, MuJoCo, Brax) into uv's cache, which takes a minute; the simulation runs on the CPU.

## `compare_log.py`: overlay robot log and simulation

Does both of the above for every run of a robot state log and draws them on the same panels: the robot log solid, the simulation dashed in the same color, so deviations stand out directly. The title gives both durations and says if the simulated robot fell.

```bash
uv run data_analysis/compare_log.py LOG.csv                     # show windows
uv run data_analysis/compare_log.py LOG.csv --save              # <log>_run<N>_compare.png next to the CSV
uv run data_analysis/compare_log.py LOG.csv --run 4 --stage -1 --seed 3
```

It takes the same options as `simulate_log.py`, except `--video`.

## `evaluate_runs.py`: metrics per run and experiment

For the quantitative experiments in `experiments/runs/` (`goose_exp_<policy>_<experiment>_*.csv`). Splits a log into runs, cuts `--trim` seconds (default 0.5) from each run's start and end, and prints the metrics of the experiment (taken from the file name, or `--experiment`) with one column per run plus mean and std:

- all experiments: duration, fell (|pitch| ≥ 0.5 rad in the last 0.2 s), pitch MAE / RMSE / max (from upright), pitch rate MAE / RMSE, mean |action|, action saturation (share of |a| ≥ 0.98), action smoothness (mean |a − previous a|), mean |wheel acceleration|
- `balancing`: linear and angular MAE and RMSE (the command is 0), position drift, heading drift
- `forward`: linear MAE, RMSE, steady-state error, rise time (to 90 %), overshoot; angular MAE and RMSE (heading hold); distance
- `spin`: angular MAE, RMSE, steady-state error, rise time, overshoot; linear MAE, RMSE and position drift
- `slope`: linear and angular MAE, RMSE and steady-state errors, distance

The robot's forward velocity is wheel odometry, `r (mean wheel speed − pitch rate)` with r = 27.5 mm, so it doesn't see wheel slip. (Integrating the accelerometer was dropped: the firmware's tilt filter absorbs sustained acceleration and the result drifts.) Commands are scaled to m/s and rad/s with the limits stored in the policy file (`--policy`; default 0.4 m/s, 2.5 rad/s). Steady-state errors use the stretches of constant nonzero command without their first `--settle` seconds (default 1).

```bash
uv run data_analysis/evaluate_runs.py LOG.csv [LOG.csv ...]
uv run data_analysis/evaluate_runs.py LOG.csv --policy software_stack/models_share/default/policy_baseline.bin
uv run data_analysis/evaluate_runs.py LOG.csv --runs 1,2,5        # only these runs
uv run data_analysis/evaluate_runs.py LOG.csv --csv metrics.csv   # also save the tables
```

## `run_evaluation.py`: robot vs. simulation for all policies

Evaluates every policy and experiment listed in `experiments/evaluation.toml`: the robot runs as by `evaluate_runs.py`, and each run replayed in simulation as by `simulate_log.py` (same policy, command sequence and start pitch; the simulated velocity is the true one). The config names each policy's `policy.bin` and per experiment its log, optionally with the runs to use:

```toml
[simulation]
skip = ["slope"]             # robot only

[simulation.clean]           # each simulation is a source next to "real" in the results
stage = "balance"            # no noise, latency or randomized physics
seeds = [0]                  # one replay per seed and run

[simulation.noisy]
stage = "commands 0.4 m/s"   # the sensor noise and latency the policies trained with
seeds = [0, 1, 2, 3, 4]

[policies.default]
model = "../../software_stack/models_share/default/policy_baseline.bin"
forward = "runs/goose_exp_default_forward_3runs.csv"
slope = { log = "runs/goose_exp_default_slope_3runs_12XX3.csv", runs = [1, 2, 5] }
```

```bash
uv run data_analysis/run_evaluation.py                                       # whole config
uv run data_analysis/run_evaluation.py --policy default --experiment forward
uv run data_analysis/run_evaluation.py --no-sim                              # robot logs only
```

Writes to `experiments/results/` (`--out` to change):

- `evaluation_runs.csv`: one row per policy, experiment, source (`real` or a simulation's name), run and seed, with all metrics
- `evaluation_summary.csv`: per policy, experiment and metric the mean and std over the runs of the robot and of each simulation (averaged over its seeds first), and each simulation's − real gap paired by run
- `evaluation_summary.md`: the summary as Markdown tables, one per experiment
