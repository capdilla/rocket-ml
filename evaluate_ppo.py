"""Evaluates or shows a PPO model saved by train_ppo.py."""
import argparse
from datetime import datetime
import math
from pathlib import Path
import random

import torch

from evaluate import run_episode, summarize
from rocket_lab.data.recovery_curriculum import load_recovery_starts, reset_recovery_scenario
from rocket_lab.ml.ppo_agent import PPOAgent
from rocket_lab.physics import (RocketEnv, GRAVITY, ENGINE_ACCEL, EARTH_GRAVITY,
                                EARTH_ENGINE_ACCEL, policy_observation)


def choose_model(root):
    """Lists only PPO snapshots, newest first."""
    candidates = []
    for path in root.rglob("model_episode_*.pt"):
        try:
            if torch.load(path, map_location="cpu", weights_only=True).get("algorithm") == "ppo":
                candidates.append(path)
        except (OSError, RuntimeError, ValueError):
            continue
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find PPO snapshots in {root.resolve()}")
    print("Available PPO models (newest first):")
    for index, path in enumerate(candidates, 1):
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"  {index}. {path}  ({modified})")
    while True:
        choice = input("Choose a number, or q to exit: ").strip().lower()
        if choice == "q":
            raise SystemExit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print(f"Enter a number between 1 and {len(candidates)}, or q to exit.")


def evaluate_episode(env, agent, render, mode, reset_fn=None):
    """Evaluates a flight and preserves the craft's final state to show it."""
    result = run_episode(env, lambda state: agent.select_action(
        policy_observation(state), deterministic=True)[0],
        render, mode, show_final_state=True, reset_fn=reset_fn)
    state = env.state
    observation = env.observation()
    final_state = {
        "x": state.x, "y": state.y, "vx": state.vx, "vy": state.vy,
        "angle": state.angle, "angular_velocity": state.angular_velocity,
        "dx": observation[6], "dy": observation[7], "pad_x": env.pad_x,
    }
    return (*result, final_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="PPO model; if omitted, a snapshot selector appears")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--planet", choices=("moon", "earth"),
                        help="Physics for evaluation; by default uses the one saved in the model")
    parser.add_argument("--pad-range", type=float, default=0.0, metavar="METERS",
                        help="Randomizes the pad's X position between -METERS and +METERS in each flight")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Folder where to search for PPO snapshots")
    parser.add_argument("--recovery-log", type=Path, action="append", default=[],
                        help="Evaluates from this log's manual-to-AI handovers; can be repeated")
    parser.add_argument("--recovery-step", type=int,
                        help="Specific step of each log instead of the manual-to-AI handover")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if not math.isfinite(args.pad_range) or args.pad_range < 0:
        parser.error("pad-range must be a finite, non-negative number")
    if args.recovery_step is not None and not args.recovery_log:
        parser.error("--recovery-step requires --recovery-log")
    if args.recovery_log and args.pad_range:
        parser.error("--pad-range is not used together with --recovery-log")
    model_path = Path(args.model) if args.model else choose_model(args.runs_dir)
    agent, mode = PPOAgent.from_model(model_path)
    planet = args.planet or agent.planet
    gravity = EARTH_GRAVITY if planet == "earth" else GRAVITY
    engine_accel = EARTH_ENGINE_ACCEL if planet == "earth" else ENGINE_ACCEL
    if args.recovery_log and mode != "xy-angle":
        parser.error("--recovery-log requires an xy-angle model")
    try:
        recovery_starts = load_recovery_starts(args.recovery_log, args.recovery_step,
                                               gravity=gravity, engine_accel=engine_accel)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Could not load the recoveries: {error}")
    env = RocketEnv(seed=args.seed,
                    max_horizontal_distance=8.0 if mode in ("xy", "xy-angle") else None,
                    max_height=14.0 if mode in ("xy", "xy-angle") else None,
                    gravity=gravity, engine_accel=engine_accel)
    pad_random = random.Random(args.seed)
    results = []
    for index in range(args.episodes):
        if recovery_starts:
            start = recovery_starts[index % len(recovery_starts)]
            reset_fn = lambda current_env, start=start, augment=index >= len(recovery_starts): (
                reset_recovery_scenario(current_env, start, 1.0, augment=augment))
        else:
            env.pad_x = pad_random.uniform(-args.pad_range, args.pad_range) if args.pad_range else 0.0
            reset_fn = None
        results.append(evaluate_episode(env, agent, args.render, mode, reset_fn=reset_fn))
    print(f"PPO ({mode}, {planet}): {summarize([result[:3] for result in results])}")
    for episode, (outcome, _reward, steps, final) in enumerate(results, 1):
        print(
            f"Episode {episode}: {outcome} in {steps} steps | "
            f"pad x={final['pad_x']:+.2f} m | "
            f"position final x={final['x']:+.2f} m, y={final['y']:+.2f} m "
            f"(dx={final['dx']:+.2f} m, dy={final['dy']:+.2f} m) | "
            f"speed vx={final['vx']:+.2f} m/s, vy={final['vy']:+.2f} m/s | "
            f"angle={final['angle']:+.3f} rad, rotation={final['angular_velocity']:+.3f} rad/s"
        )
    print(f"Model tested: {model_path}")


if __name__ == "__main__":
    main()
