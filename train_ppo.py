"""Trains PPO in the same environment as DQN and saves comparable metrics.

In Y, XY and XY-angle, the dense descent signal and the timeout penalty help
avoid flights that drag on indefinitely. XY adds a horizontal progress signal;
XY-angle teaches correcting tilt and rotation.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch.distributions import Categorical

from rocket_lab.data.recovery_curriculum import load_recovery_starts, reset_recovery_scenario
from rocket_lab.ml.ppo_agent import PPOAgent
from rocket_lab.physics import (RocketEnv, OBSERVATION_SIZE, SAFE_VERTICAL_SPEED, PAD_HALF_WIDTH,
                                XY_ALLOWED_ACTIONS, XY_ANGLE_ALLOWED_ACTIONS,
                                GRAVITY, ENGINE_ACCEL, EARTH_GRAVITY, EARTH_ENGINE_ACCEL,
                                policy_observation)
from rocket_lab.reporting import make_plot
from train import shaped_reward


def vertical_guidance_reward(state, next_state):
    """Dense feedback for reaching the pad at a controlled vertical speed."""
    height = max(0.0, next_state[1])
    target_descent = SAFE_VERTICAL_SPEED + (2.5 - SAFE_VERTICAL_SPEED) * height / (height + 3.0)
    speed_error = min(abs(next_state[3] + target_descent), 5.0)
    descent = min(max(state[1] - next_state[1], 0.0), 0.05)
    return 2.0 * descent - 0.03 * speed_error ** 2


def lateral_guidance_reward(state, next_state):
    """Guide X movement toward the pad, then slow down before touchdown."""
    dx = next_state[6]
    height = max(0.0, next_state[1])
    target_vx = max(-1.0, min(1.0, 0.5 * dx)) * min(1.0, height / 3.0)
    speed_error = min(abs(next_state[2] - target_vx), 3.0)
    progress = abs(state[6]) - abs(dx)
    return 8.0 * progress - 0.02 * speed_error ** 2


def attitude_guidance_reward(state, next_state):
    """Reward reducing tilt and spin before touchdown."""
    angle_progress = abs(state[4]) - abs(next_state[4])
    spin_progress = abs(state[5]) - abs(next_state[5])
    return (8.0 * angle_progress + 2.0 * spin_progress
            - 0.03 * next_state[4] ** 2 - 0.02 * next_state[5] ** 2)


def load_xy_angle_policy(path):
    """Load an existing XY-angle policy without carrying over its optimizer."""
    agent, mode = PPOAgent.from_model(path)
    if mode != "xy-angle" or agent.allowed_actions != XY_ANGLE_ALLOWED_ACTIONS:
        raise ValueError(f"{path} is not a PPO xy-angle policy")
    return agent


def ppo_update(agent, optimizer, episodes, args):
    """Updates the policy using each recent transition over a few epochs."""
    states, actions, old_log_probs, advantages, returns = [], [], [], [], []
    for trajectory in episodes:
        gae = 0.0
        next_value = 0.0  # Each trajectory ends by contact or timeout.
        episode_advantages = []
        for step in reversed(trajectory):
            delta = step["reward"] + args.gamma * next_value - step["value"]
            gae = delta + args.gamma * args.gae_lambda * gae
            episode_advantages.append(gae)
            next_value = step["value"]
        episode_advantages.reverse()
        for step, advantage in zip(trajectory, episode_advantages):
            states.append(step["state"])
            actions.append(step["action_index"])
            old_log_probs.append(step["log_prob"])
            advantages.append(advantage)
            returns.append(advantage + step["value"])

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    losses = []
    for _ in range(args.update_epochs):
        for indices in torch.randperm(len(states)).split(args.minibatch_size):
            logits, values = agent(states[indices])
            distribution = Categorical(logits=logits)
            ratio = (distribution.log_prob(actions[indices]) - old_log_probs[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (values - returns[indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + args.value_coefficient * value_loss - args.entropy_coefficient * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


class TrainingRenderer:
    """Renders the current flight; pygame is only loaded when --render is requested."""
    def __init__(self):
        import pygame
        from play import BG, WIDTH, HEIGHT, draw_rocket, screen_point
        pygame.init()
        self.pygame = pygame
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Rocket Lab — training PPO")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 17)
        self.small = pygame.font.SysFont("monospace", 14)
        self.bg, self.width, self.height = BG, WIDTH, HEIGHT
        self.draw_rocket, self.screen_point = draw_rocket, screen_point
        self.enabled = True

    def draw(self, env, episode, reward):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.enabled = False
                pygame.quit()
                return
        self.screen.fill(self.bg)
        ground_y = self.screen_point(0, env.pad_y)[1]
        pygame.draw.line(self.screen, (72, 91, 117), (0, ground_y), (self.width, ground_y), 3)
        pygame.draw.line(self.screen, (68, 211, 143),
                         self.screen_point(env.pad_x - PAD_HALF_WIDTH, env.pad_y),
                         self.screen_point(env.pad_x + PAD_HALF_WIDTH, env.pad_y), 9)
        self.draw_rocket(self.screen, env)
        state = env.state
        lines = [f"PPO episode {episode} | reward {reward:+.1f}",
                 f"x={state.x:+.2f}m  y={state.y:+.2f}m  vx={state.vx:+.2f}m/s  vy={state.vy:+.2f}m/s",
                 f"angle={state.angle:+.2f}rad  rotation={state.angular_velocity:+.2f}rad/s  action={env.last_action}",
                 f"result: {env.last_outcome or 'in flight'} | ESC/close: continue without render"]
        for index, line in enumerate(lines):
            font = self.font if index == 0 else self.small
            self.screen.blit(font.render(line, True, (234, 238, 247)), (18, 18 + index * 26))
        pygame.display.flip()
        self.clock.tick(60)


    def close(self):
        if self.enabled:
            self.pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--mode", choices=("y", "xy", "xy-angle", "full"), default="y")
    parser.add_argument("--planet", choices=("moon", "earth", "both"), default="moon",
                        help="Physics used in training; 'both' alternates moon and earth")
    parser.add_argument("--earth-ratio", type=float, default=0.5,
                        help="Fraction of earth episodes when --planet both (0.5 by default)")
    parser.add_argument("--render", action="store_true", help="Shows each flight at 60 steps per second")
    initial_policy = parser.add_mutually_exclusive_group()
    initial_policy.add_argument("--init-model", help="PPO-Y model for XY, or PPO-XY for XY-angle")
    initial_policy.add_argument("--init-policy", help="An already trained xy-angle policy for a new phase")
    parser.add_argument("--curriculum-episodes", type=int, default=2000,
                        help="Episodes to grow the initial X offset from 0.5 to 3.5 m")
    parser.add_argument("--angle-curriculum-episodes", type=int, default=2000,
                        help="Episodes to grow tilt to 0.45 rad and rotation to 0.3 rad/s")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="In xy-angle, use the full difficulty from the first episode")
    parser.add_argument("--recovery-log", type=Path, action="append", default=[],
                        help="Play log to start recoveries; can be repeated")
    parser.add_argument("--recovery-step", type=int,
                        help="Specific step of each log; by default uses the manual-to-AI handovers")
    parser.add_argument("--recovery-ratio", type=float, default=0.3,
                        help="Fraction of recovery episodes when using --recovery-log (0.3 by default)")
    parser.add_argument("--recovery-curriculum-episodes", type=int, default=3000,
                        help="Episodes to grow the recovery up to the log's severity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/ppo-y-8")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                        help="Minimum steps of complete episodes per update")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--shaping-scale", type=float, default=0.2)
    parser.add_argument("--no-reward-shaping", action="store_true")
    parser.add_argument("--no-vertical-guidance", action="store_true",
                        help="Disables the dense controlled-descent signal")
    parser.add_argument("--timeout-penalty", type=float, default=-100.0,
                        help="Extra training penalty when time runs out")
    parser.add_argument("--success-window", type=int, default=100)
    parser.add_argument("--snapshot-frequency", type=int, default=100)
    args = parser.parse_args()
    if (args.episodes < 1 or args.rollout_steps < 1 or args.minibatch_size < 1 or
            args.update_epochs < 1 or args.snapshot_frequency < 0 or args.success_window < 0 or
            args.curriculum_episodes < 1 or args.angle_curriculum_episodes < 1 or
            args.recovery_curriculum_episodes < 1 or
            not 0 < args.recovery_ratio <= 1 or
            (args.recovery_step is not None and args.recovery_step < 1) or
            not 0 <= args.gamma <= 1 or
            not 0 <= args.gae_lambda <= 1 or args.clip_range <= 0 or
            not 0 <= args.earth_ratio <= 1):
        parser.error("Invalid training parameters")
    if args.init_model and args.mode not in ("xy", "xy-angle"):
        parser.error("--init-model is only used when training with --mode xy or xy-angle")
    if args.init_policy and args.mode != "xy-angle":
        parser.error("--init-policy requires --mode xy-angle")
    if args.no_curriculum and args.mode != "xy-angle":
        parser.error("--no-curriculum requires --mode xy-angle")
    if args.recovery_log and args.mode != "xy-angle":
        parser.error("--recovery-log requires --mode xy-angle")
    if args.recovery_step is not None and not args.recovery_log:
        parser.error("--recovery-step requires --recovery-log")
    gravity = EARTH_GRAVITY if args.planet == "earth" else GRAVITY
    engine_accel = EARTH_ENGINE_ACCEL if args.planet == "earth" else ENGINE_ACCEL
    planet_physics = {"moon": (GRAVITY, ENGINE_ACCEL), "earth": (EARTH_GRAVITY, EARTH_ENGINE_ACCEL)}
    try:
        if args.planet == "both":
            recovery_starts = load_recovery_starts(args.recovery_log, args.recovery_step,
                                                   gravity=None)
        else:
            recovery_starts = load_recovery_starts(args.recovery_log, args.recovery_step,
                                                   gravity=gravity, engine_accel=engine_accel)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Could not load the recoveries: {error}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "metrics.csv"
    checkpoint_path = output / "checkpoint.pt"
    config = vars(args).copy()
    config["recovery_start_states"] = tuple(
        (str(start.path), start.step, start.state, start.pad_x, start.pad_y)
        for start in recovery_starts)
    allowed_actions = ((0, 1) if args.mode == "y" else
                       XY_ALLOWED_ACTIONS if args.mode == "xy" else
                       XY_ANGLE_ALLOWED_ACTIONS if args.mode == "xy-angle" else None)
    agent = PPOAgent(allowed_actions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
    env = RocketEnv(seed=args.seed,
                    max_horizontal_distance=8.0 if args.mode in ("xy", "xy-angle") else None,
                    max_height=14.0 if args.mode in ("xy", "xy-angle") else None,
                    gravity=gravity, engine_accel=engine_accel)

    def episode_physics():
        if args.planet == "both":
            planet = "earth" if random.random() < args.earth_ratio else "moon"
        else:
            planet = args.planet
        return planet_physics[planet]
    renderer = TrainingRenderer() if args.render else None
    completed = 0
    updates = 0
    previous_elapsed = 0.0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["policy"]["body.0.weight"].shape[1] != OBSERVATION_SIZE:
            parser.error(f"{checkpoint_path} uses old observations; use a different --output-dir")
        if saved.get("algorithm") != "ppo" or saved.get("config", {}).get("mode") != args.mode:
            parser.error(f"{checkpoint_path} is not PPO in mode {args.mode}")
        fixed_options = ("planet", "seed", "rollout_steps", "minibatch_size", "update_epochs",
                         "learning_rate", "gamma", "gae_lambda", "clip_range",
                         "value_coefficient", "entropy_coefficient", "shaping_scale",
                         "no_reward_shaping", "no_vertical_guidance", "timeout_penalty",
                         "init_model", "init_policy", "curriculum_episodes", "no_curriculum")
        if args.mode == "xy-angle":
            fixed_options += ("angle_curriculum_episodes",)
        if args.planet == "both":
            fixed_options += ("earth_ratio",)
        if args.recovery_log or saved["config"].get("recovery_log"):
            fixed_options += ("recovery_log", "recovery_step", "recovery_ratio",
                              "recovery_curriculum_episodes", "recovery_start_states")
        changed = [name for name in fixed_options
                   if saved["config"].get(name, "moon" if name == "planet" else
                                          False if name == "no_curriculum" else None) != config[name]]
        if changed:
            parser.error(f"The checkpoint uses different values for {', '.join(changed)}; use another folder")
        agent.load_state_dict(saved["policy"])
        optimizer.load_state_dict(saved["optimizer"])
        completed, updates = saved["episode"], saved["updates"]
        previous_elapsed = saved.get("training_elapsed_seconds", 0.0)
        env.random.setstate(saved["environment_random_state"])
        random.setstate(saved["python_random_state"])
        torch.set_rng_state(saved["torch_random_state"])
        if not csv_path.exists():
            parser.error(f"Missing {csv_path} to resume the checkpoint")
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows or int(rows[-1]["episode"]) != completed:
            parser.error("The metrics and the checkpoint do not match; use another folder")
        landing_streak = 0
        for row in reversed(rows):
            if row["outcome"] != "LANDED":
                break
            landing_streak += 1
        file_mode = "a"
        print(f"Resuming PPO from episode {completed}.")
    else:
        if csv_path.exists():
            parser.error(f"There is metrics without a checkpoint in {csv_path}; use another folder")
        if args.init_policy:
            try:
                agent = load_xy_angle_policy(args.init_policy)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                parser.error(f"Could not initialize from {args.init_policy}: {error}")
            optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
            print(f"Initializing PPO xy-angle from {args.init_policy}")
        elif args.init_model:
            try:
                agent = (PPOAgent.from_vertical_model(args.init_model) if args.mode == "xy"
                         else PPOAgent.from_xy_model(args.init_model))
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                parser.error(f"Could not initialize from {args.init_model}: {error}")
            optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)
            print(f"Initializing PPO {args.mode} from {args.init_model}")
        landing_streak = 0
        file_mode = "w"

    started_at = time.perf_counter()
    fields = ["episode", "total_reward", "duration_steps", "epsilon", "mean_loss", "outcome", "training_steps"]
    if recovery_starts:
        fields.append("scenario")

    def save(episode):
        torch.save({"algorithm": "ppo", "policy": agent.state_dict(), "optimizer": optimizer.state_dict(),
                    "input_size": OBSERVATION_SIZE,
                    "episode": episode, "updates": updates, "config": config,
                    "environment_random_state": env.random.getstate(),
                    "python_random_state": random.getstate(), "torch_random_state": torch.get_rng_state(),
                    "training_elapsed_seconds": previous_elapsed + time.perf_counter() - started_at}, checkpoint_path)
        agent.save_model(output / "model.pt", args.mode, planet=args.planet)

    next_snapshot = ((completed // args.snapshot_frequency) + 1) * args.snapshot_frequency if args.snapshot_frequency else None
    with csv_path.open(file_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file_mode == "w":
            writer.writeheader()
        target_episode = completed + args.episodes
        while completed < target_episode:
            trajectories = []
            episode_metrics = []
            rollout_steps = 0
            mastered = False
            while rollout_steps < args.rollout_steps and completed + len(trajectories) < target_episode:
                env.gravity, env.engine_accel = episode_physics()
                scenario = "normal"
                if args.mode == "y":
                    state = env.reset_y_only()
                elif args.mode == "xy":
                    # The vertical flight remains the known one; only the
                    # position error and X speed are gradually grown.
                    progress = min(1.0, (completed + len(trajectories)) / args.curriculum_episodes)
                    max_offset = 0.5 + 3.0 * progress
                    max_lateral_speed = 0.5 * progress
                    state = env.reset_lateral(max_offset, max_lateral_speed)
                elif args.mode == "xy-angle":
                    if recovery_starts and random.random() < args.recovery_ratio:
                        scenario = "recovery"
                        progress = (1.0 if args.no_curriculum else
                                    min(1.0, (completed + len(trajectories)) /
                                        args.recovery_curriculum_episodes))
                        start = env.random.choice(recovery_starts)
                        state = reset_recovery_scenario(env, start, progress)
                    else:
                        progress = (1.0 if args.no_curriculum or recovery_starts else
                                    min(1.0, (completed + len(trajectories)) /
                                        args.angle_curriculum_episodes))
                        env.pad_x = env.pad_y = 0.0
                        state = policy_observation(env.reset_lateral_attitude(
                            max_offset=2.0 + 1.5 * progress,
                            max_speed=0.25 + 0.25 * progress,
                            max_angle=0.05 + 0.40 * progress,
                            max_angular_speed=0.05 + 0.25 * progress))
                else:
                    state = env.reset(randomize=True)
                trajectory = []
                total_reward = 0.0
                while not env.finished:
                    action, action_index, log_prob, value = agent.select_action(state)
                    raw_next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = policy_observation(raw_next_state) if args.mode == "xy-angle" else raw_next_state
                    training_reward = shaped_reward(
                        state, next_state, reward, terminated, not args.no_reward_shaping,
                        gamma=args.gamma, truncated=truncated, scale=args.shaping_scale)
                    if args.mode in ("y", "xy", "xy-angle") and not args.no_vertical_guidance:
                        training_reward += vertical_guidance_reward(state, next_state)
                    if args.mode in ("xy", "xy-angle"):
                        training_reward += lateral_guidance_reward(state, next_state)
                    if args.mode == "xy-angle":
                        training_reward += attitude_guidance_reward(state, next_state)
                    if truncated:
                        training_reward += args.timeout_penalty
                    trajectory.append({"state": state, "action_index": action_index,
                                       "log_prob": log_prob, "value": value,
                                       "reward": training_reward / 100.0})
                    total_reward += reward
                    state = next_state
                    if renderer and renderer.enabled:
                        renderer.draw(env, completed + len(trajectories) + 1, total_reward)
                trajectories.append(trajectory)
                rollout_steps += len(trajectory)
                episode_metrics.append((total_reward, env.steps, info["outcome"], scenario))
                episode_number = completed + len(trajectories)
                if args.mode == "xy-angle" and not args.no_curriculum and episode_number < (
                        args.recovery_curriculum_episodes if recovery_starts else
                        args.angle_curriculum_episodes):
                    landing_streak = 0
                else:
                    landing_streak = landing_streak + 1 if info["outcome"] == "LANDED" else 0
                if args.success_window and landing_streak >= args.success_window:
                    mastered = True
                    break

            # The episodes were collected with the policy before the update.
            # If it has already mastered the challenge, that exact policy is saved.
            mean_loss = 0.0 if mastered else ppo_update(agent, optimizer, trajectories, args)
            if not mastered:
                updates += 1
            for total_reward, steps, outcome, scenario in episode_metrics:
                completed += 1
                row = {"episode": completed, "total_reward": total_reward,
                       "duration_steps": steps, "epsilon": "",
                       "mean_loss": mean_loss, "outcome": outcome,
                       "training_steps": updates}
                if recovery_starts:
                    row["scenario"] = scenario
                writer.writerow(row)
            file.flush()
            save(completed)
            if next_snapshot is not None and completed >= next_snapshot:
                snapshot = output / f"model_episode_{completed:06d}.pt"
                agent.save_model(snapshot, args.mode, planet=args.planet)
                next_snapshot = ((completed // args.snapshot_frequency) + 1) * args.snapshot_frequency
                print(f"Episode {completed}: model saved to {snapshot}")
            if mastered:
                print(f"PPO reached {args.success_window} consecutive landings; stopping.")
                break
            if completed % 50 < len(episode_metrics):
                print(f"PPO episode {completed}: {episode_metrics[-1][2]}, loss={mean_loss:.4f}")

    final_snapshot = output / f"model_episode_{completed:06d}.pt"
    agent.save_model(final_snapshot, args.mode, planet=args.planet)
    if renderer:
        renderer.close()
    print(f"PPO finished at episode {completed}. Plot: {make_plot(csv_path)}")
    print(f"Model: {output / 'model.pt'}; checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
