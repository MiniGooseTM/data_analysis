# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["matplotlib>=3.7", "pandas>=2", "self-balancing-robot"]
#
# [tool.uv.sources]
# self-balancing-robot = { path = "../software_stack", editable = true }
# ///
"""Replay the commands of a robot state log in simulation and plot the simulated run.

For every run in a robot state log (``goose_state_*.csv``, see ``plot_logs.py``) the policy
is run in the MuJoCo environment of ``software_stack`` with the same ``v_cmd`` /
``omega_cmd`` sequence, step for step (missing log rows keep the previous command). The
simulated run starts at the run's first logged pitch, standing still, with the history
filled with copies of the first frame like the firmware does on Start. Each step records
the frame the policy saw and its action, i.e. the same columns the robot logs, and the
result is drawn with the panels of ``plot_logs.py``, so the two figures compare 1:1.

The simulation ends at the first fall (pitch or roll past 0.6 rad, spinning, or the
chassis touching the ground), where the robot firmware cuts the motors as well.

The ground is flat unless ``--slope`` is given; ``--stage`` selects what else the
simulation randomizes (default: a clean simulation, no noise, latency or randomized
physics; e.g. ``--stage -1`` for the last training stage with all of them).

Usage:
    uv run data_analysis/simulate_log.py LOG.csv                        # show windows
    uv run data_analysis/simulate_log.py LOG.csv --save                 # PNG next to the CSV
    uv run data_analysis/simulate_log.py LOG.csv --save --out plots/    # PNG into plots/
    uv run data_analysis/simulate_log.py LOG.csv --run 2                # only the 2nd run
    uv run data_analysis/simulate_log.py LOG.csv --model ../software_stack/models/runs/<run>
    uv run data_analysis/simulate_log.py LOG.csv --stage -1 --seed 3    # noisy, randomized
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_logs import ROBOT_PANELS, plot_panels, split_runs
from self_balancing_robot.actor import Actor, forward
from self_balancing_robot.curriculum import CURRICULUM, FULL_COMMAND
from self_balancing_robot.envs.balance_env import (
    ACTOR_FRAME_FIELDS,
    ACTOR_FRAME_SIZE,
    ACTOR_KEY,
    SelfBalancingRobotEnv,
    axis_angle_quat,
    quat_mul,
)

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "software_stack/models/policy.bin"
DEFAULT_STAGE = "commands 0.3 m/s"
# Steps per jitted rollout call: one compilation serves runs of any length.
CHUNK_STEPS = 500


class LogReplay:
    """One simulated robot, driven by the policy with a given command per step."""

    def __init__(self, env: SelfBalancingRobotEnv, actor: Actor, slope: float):
        self.env = env
        self.layers = [(jnp.asarray(w), jnp.asarray(b)) for w, b in actor.layers]
        self.slope = slope
        self._start = jax.jit(self._start_state)
        self._rollout = jax.jit(self._rollout_chunk)

    def _start_state(self, key, command, pitch):
        """Episode at `command`, standing still at `pitch` (rad) on the ground."""
        env = self.env
        reset_key, observe_key = jax.random.split(key)
        persistent = {
            **env.initial_persistent(),
            "command_held": jnp.array(True),
            "command": command,
            "override_slope": jnp.float32(self.slope),
        }
        state = env.reset_episode(reset_key, persistent)
        # Replace the random initial lean and velocities with the logged start. The slope's
        # axis is pinned along world x, so a lean about the axle is a pitch on it as well.
        lean = axis_angle_quat(jnp.array([1.0, 0.0, 0.0]), pitch - jnp.float32(self.slope))
        quat = quat_mul(state.data.mocap_quat[0], lean)
        base = env._base_qpos
        data = state.data.replace(
            qpos=state.data.qpos.at[base + 3 : base + 7].set(quat),
            qvel=jnp.zeros_like(state.data.qvel),
        )
        obs, info = env._observe(data, state.info, observe_key, first=True)
        return state.replace(data=data, obs=obs, info=info)

    def _rollout_chunk(self, state, next_commands):
        """Run one step per entry of `next_commands` (the command of the following frame)."""

        def body(state, command):
            observation = state.obs[ACTOR_KEY]
            action = jnp.clip(forward(self.layers, observation), -1.0, 1.0)
            # The command set now is the one the next observation (frame) carries.
            state = self.env.step(state.replace(info={**state.info, "command": command}), action)
            record = {
                "frame": observation[:ACTOR_FRAME_SIZE],
                "action": action,
                "terminated": state.info["terminated"],
            }
            return state, record

        return jax.lax.scan(body, state, next_commands)

    def run(self, commands: np.ndarray, start_pitch: float, seed: int) -> tuple[pd.DataFrame, bool]:
        """Simulate one step per row of `commands` ([v m/s, omega rad/s]); returns the robot-log
        columns per step and whether the robot fell (the frame ends with the falling step)."""
        steps = len(commands)
        padded = np.concatenate([commands[1:], np.repeat(commands[-1:], CHUNK_STEPS, axis=0)])
        state = self._start(
            jax.random.PRNGKey(seed), jnp.asarray(commands[0]), jnp.float32(start_pitch)
        )
        chunks = []
        for begin in range(0, steps, CHUNK_STEPS):
            state, chunk = self._rollout(state, jnp.asarray(padded[begin : begin + CHUNK_STEPS]))
            chunk = jax.device_get(chunk)
            chunks.append(chunk)
            if chunk["terminated"].any():
                break
        record = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
        fall_steps = np.flatnonzero(record["terminated"][:steps])
        fell = fall_steps.size > 0
        length = fall_steps[0] + 1 if fell else steps
        df = pd.DataFrame(record["frame"][:length], columns=list(ACTOR_FRAME_FIELDS))
        df["action_left"] = record["action"][:length, 0]
        df["action_right"] = record["action"][:length, 1]
        df.insert(0, "step", np.arange(length))
        df.insert(1, "t", df["step"] * self.env.dt)
        return df, fell


def read_robot_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"step", "time_s", "v_cmd", "omega_cmd", "pitch"} <= set(df.columns):
        raise ValueError(f"not a robot state log: {','.join(df.columns)}")
    # A browser that reconnected mid-log may have appended a second header line
    return df[pd.to_numeric(df["step"], errors="coerce").notna()].apply(pd.to_numeric)


def logged_commands(run: pd.DataFrame) -> np.ndarray:
    """[v m/s, omega rad/s] for every step of the run, held over missing rows."""
    steps = pd.RangeIndex(int(run["step"].max()) + 1)
    scaled = run.set_index("step")[["v_cmd", "omega_cmd"]].reindex(steps).ffill()
    return (scaled.to_numpy() * np.asarray(FULL_COMMAND)).astype(np.float32)


def simulated_runs(path: Path, replay: LogReplay, args):
    """(run number, run count, logged run, simulated run, fell) for every selected run."""
    runs = split_runs(read_robot_log(path))
    for i, run in enumerate(runs, start=1):
        if args.run is not None and i != args.run:
            continue
        commands = logged_commands(run)
        sim, fell = replay.run(commands, float(run["pitch"].iloc[0]), args.seed)
        print(f"{path.name} run {i}: simulated {len(sim)}/{len(commands)} steps"
              + (", fell" if fell else ""))
        yield i, len(runs), run, sim, fell


def simulated_figures(path: Path, replay: LogReplay, args, label: str):
    for i, count, run, sim, fell in simulated_runs(path, replay, args):
        duration = sim["t"].iloc[-1]
        title = (
            f"{path.name} - run {i}/{count} simulated ({label}): "
            f"{duration:.1f} s, {len(sim)} steps"
        )
        if fell:
            title += f", fell (robot log: {int(run['step'].max()) + 1} steps)"
        yield f"run{i}_sim", plot_panels(sim, ROBOT_PANELS, title, "time since run start [s]")


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL,
        help="policy.bin or training run directory (default: software_stack/models/policy.bin)",
    )
    parser.add_argument(
        "--stage", default=DEFAULT_STAGE,
        help=f'curriculum stage to simulate, by name or index (default "{DEFAULT_STAGE}"). '
        "Stages: " + ", ".join(f'{i} "{s.name}"' for i, s in enumerate(CURRICULUM)),
    )
    parser.add_argument("--slope", type=float, default=0.0, help="ground slope in rad (default 0)")
    parser.add_argument("--seed", type=int, default=0, help="seed for noise and randomized physics")


def make_replay(args, parser: argparse.ArgumentParser) -> tuple[LogReplay, str]:
    """The replay for the simulation arguments, and a label describing it for titles."""
    actor = Actor.load(args.model)
    try:
        env = SelfBalancingRobotEnv(
            start_stage=args.stage, fixed_difficulty=True, max_episode_steps=2**31 - 1
        )
    except ValueError as error:
        parser.error(str(error))
    stage = env.curriculum[env.start_stage].name
    label = f'{actor.name or args.model.name}, stage "{stage}"'
    if args.slope:
        label += f", slope {args.slope:g} rad"
    return LogReplay(env, actor, args.slope), label


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
            figs = list(simulated_figures(path, replay, args, label))
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
