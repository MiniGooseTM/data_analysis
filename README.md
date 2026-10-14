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
- `--stage`: curriculum stage to simulate, by name or index. The default `"commands 0.3 m/s"` is a clean simulation; `-1` (the last stage) adds sensor noise, latency and randomized mass, friction, CoM and pushes.
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

## `evaluate_runs.py`: evaluation metrics per run

For robot state logs with accelerometer columns (`acc_x,acc_y,acc_z`, e.g. `experiments/*.csv`). Splits the log into runs at `time_s == 0` (3 expected, `--runs` to change), cuts `--trim` seconds (default 0.5) from each run's start and end, and prints a table with one row per run plus the average: velocity tracking MAE (linear from the integrated, gravity-compensated forward acceleration, high-pass filtered at `--highpass` Hz against drift; angular from the yaw rate), pitch rate MAE, and the mean absolute wheel velocity, action and wheel acceleration over both wheels.

```bash
uv run data_analysis/evaluate_runs.py LOG.csv [LOG.csv ...]
uv run data_analysis/evaluate_runs.py LOG.csv --csv metrics.csv   # also save the tables
```
