"""Trains a RocketBrain by repeating flights and learning from what happens.

Think of an episode as a complete flight, from takeoff until landing, crash or
timeout. At each step:

1. The environment returns eight numbers: position, speed, tilt, rotation and
   the horizontal and vertical distance to the pad.
2. The agent chooses one of the six actions: coast, fire the engine, rotate, or
   combine engine and rotation.
3. The environment applies that action and returns the new state and a reward.
4. The program saves state, action, reward and new state in memory.
5. When there are enough examples, the agent practices with a random group of
   them and adjusts its network so its predictions improve.

At the end of each flight its metrics are saved. Every certain number of
flights the model is also saved so training can be resumed or tested later.
This training does not open Pygame. To start: ``python3 train.py``.

Related files: ``physics.py`` simulates the movement; ``brain.py`` defines the
neural network; ``replay_buffer.py`` saves examples; ``dqn_agent.py`` chooses
actions and updates the network; ``evaluate.py`` tests an already trained
model.
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import torch

from rocket_lab.ml.dqn_agent import DQNAgent
from rocket_lab.ml.replay_buffer import ReplayBuffer
from rocket_lab.physics import RocketEnv, OBSERVATION_SIZE, PAD_HALF_WIDTH, SAFE_HORIZONTAL_SPEED, SAFE_VERTICAL_SPEED, SAFE_ANGLE
from rocket_lab.reporting import make_plot


def shaped_reward(state, next_state, reward, terminated, enabled=False, *, gamma=0.99,
                  truncated=False, scale=0.2, height_coef=1.0,
                  vertical_speed_coef=1.0, horizontal_position_coef=1.0,
                  horizontal_speed_coef=1.0, attitude_coef=1.0):
    """Returns the training reward; ``reward`` remains unchanged.

    The potential difference is the only additional reward. Absorbing states
    (landing, crash and timeout) have zero potential.
    """
    if not enabled:
        return reward

    def potential(s):
        x, y, vx, vy, angle, angular_velocity = s[:6]
        y = max(0.0, y)
        # Target descent: up to 2.5 m/s at altitude, easing down to
        # SAFE_VERTICAL_SPEED near the ground.
        target_descent = SAFE_VERTICAL_SPEED + (2.5 - SAFE_VERTICAL_SPEED) * y / (y + 3.0)
        descent_error = (vy + target_descent) / 2.5
        cost = (
            height_coef * (y / 10.0)
            + vertical_speed_coef * descent_error ** 2
            + horizontal_position_coef * (x / 3.0) ** 2
            + horizontal_speed_coef * (vx / max(SAFE_HORIZONTAL_SPEED, 0.1)) ** 2
            + attitude_coef * ((angle / SAFE_ANGLE) ** 2 + (angular_velocity / 0.35) ** 2)
        )
        # Bounded, positive potential: when staying in a state without progress,
        # gamma*Phi - Phi is negative, so hovering does not produce a reward.
        return math.exp(-max(0.0, cost))

    # Terminal potential = 0 prevents an artificial reward at the end and
    # respects the original terminal reward (+100 / -100). Timeout is also
    # absorbing: the flight cannot be prolonged to accumulate potential.
    next_potential = 0.0 if terminated or truncated else potential(next_state)
    return reward + scale * (gamma * next_potential - potential(state))


def main():
    # The following options can be changed when running the command.
    # For example: --episodes 200 --batch-size 32 --seed 7.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="How many complete flights to run")
    parser.add_argument("--batch-size", type=int, default=64, help="How many examples to review in each training step")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Size of the changes to the network weights")
    parser.add_argument("--gamma", type=float, default=0.99, help="How much the future counts when valuing an action (0 to 1)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Exploration at the start: 1 means always random")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum exploration: 0.05 means 5 percent random")
    parser.add_argument("--epsilon-decay-steps", type=int, default=30000, help="Steps to lower epsilon from start to minimum")
    parser.add_argument("--target-update-frequency", type=int, default=500, help="Every how many practice steps to copy policy to target")
    parser.add_argument("--seed", type=int, default=42, help="Seed to try to repeat the experiment")
    parser.add_argument("--buffer-capacity", type=int, default=50000, help="Maximum examples the memory stores")
    parser.add_argument("--warmup", type=int, default=1000, help="Examples collected before starting to train")
    parser.add_argument("--reward-shaping", action="store_true", help="Enables the extra progress hints")
    parser.add_argument("--shaping-scale", type=float, default=0.2, help="Lambda of the potential-based reward shaping")
    parser.add_argument("--shaping-height", type=float, default=1.0, help="Weight of altitude progress")
    parser.add_argument("--shaping-vertical-speed", type=float, default=1.0, help="Weight of vertical speed matching the altitude")
    parser.add_argument("--shaping-horizontal-position", type=float, default=1.0, help="Weight of moving closer to the center")
    parser.add_argument("--shaping-horizontal-speed", type=float, default=1.0, help="Weight of reducing lateral speed")
    parser.add_argument("--shaping-attitude", type=float, default=1.0, help="Weight of tilt and rotation")
    parser.add_argument("--output-dir", default="runs/dqn-y", help="Folder for metrics and saved models")
    parser.add_argument("--mode", choices=("y", "full"), default="y",
                        help="y: fixed vertical landing with engine; full: 2D control and random scenarios")
    parser.add_argument("--success-window", type=int, default=100,
                        help="Consecutive episodes required to declare mastery (0 disables the notice)")
    parser.add_argument("--snapshot-frequency", type=int, default=100,
                        help="Save a model copy every N episodes (0 disables periodic copies)")
    args = parser.parse_args()
    if args.episodes < 1 or args.batch_size < 1 or args.snapshot_frequency < 0 or args.success_window < 0:
        parser.error("episodes and batch-size must be positive; snapshot-frequency and success-window cannot be negative")

    # Fix the random generators so the run is reproducible.
    # Learning on different hardware can still vary a little.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()


    # agent decides actions and learns; buffer saves examples; env computes the physics.
    allowed_actions = (0, 1) if args.mode == "y" else None
    agent = DQNAgent(args.learning_rate, args.gamma, args.target_update_frequency,
                     seed=args.seed, allowed_actions=allowed_actions)
    buffer = ReplayBuffer(args.buffer_capacity, args.seed)
    env = RocketEnv(seed=args.seed)
    csv_path = output / "metrics.csv"
    fields = ["episode", "total_reward", "duration_steps", "epsilon", "mean_loss", "outcome", "training_steps"]

    checkpoint_path = output / "checkpoint.pt"
    previous_elapsed = 0.0
    if checkpoint_path.exists():
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if metadata.get("input_size", metadata["policy"]["network.0.weight"].shape[1]) != OBSERVATION_SIZE:
            parser.error(f"The checkpoint {checkpoint_path} uses old observations. Use a different --output-dir to start a new training run")
        saved_mode = metadata.get("config", {}).get("mode", "full")
        if saved_mode != args.mode:
            parser.error(f"The checkpoint {checkpoint_path} is from mode '{saved_mode}', not '{args.mode}'. Use a different --output-dir to start this mode")
        saved = agent.load_checkpoint(checkpoint_path)
        start_episode = int(saved.get("episode", 0))
        previous_elapsed = float(saved.get("training_elapsed_seconds", 0.0))
        if saved.get("replay_buffer") is not None:
            buffer.load_state_dict(saved["replay_buffer"])
        else:
            print("Warning: this old checkpoint did not save the experience memory; training will resume with an empty buffer.")
        saved_random = saved.get("random_states")
        if saved_random:
            random.setstate(saved_random["python"])
            agent.random.setstate(saved_random["agent"])
            env.random.setstate(saved_random["environment"])
            torch.set_rng_state(saved_random["torch"])
        print(f"Resuming from episode {start_episode} ({agent.training_steps} updates).")
        file_mode = "a"
    else:
        if csv_path.exists():
            parser.error(f"There is metrics in {csv_path} but {checkpoint_path} does not exist; use another folder to start a new training run")
        start_episode = 0
        file_mode = "w"

    run_started_at = time.perf_counter()

    def save_training_checkpoint(episode, epsilon):
        random_states = {"python": random.getstate(), "agent": agent.random.getstate(),
                         "environment": env.random.getstate(), "torch": torch.get_rng_state()}
        agent.save_checkpoint(checkpoint_path, config, episode, epsilon,
                              replay_buffer_state=buffer.state_dict(), random_states=random_states,
                              training_elapsed_seconds=previous_elapsed + time.perf_counter() - run_started_at)

    with csv_path.open(file_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file_mode == "w":
            writer.writeheader()
        for episode in range(start_episode + 1, start_episode + args.episodes + 1):
            # Mode y repeats the same vertical descent; full uses random scenarios.
            state = env.reset_y_only() if args.mode == "y" else env.reset(randomize=True)
            total_reward, losses, terminated, truncated = 0.0, [], False, False
            # The flight continues until landing/crash (terminated) or timeout
            # (truncated), according to the signals returned by RocketEnv.step().
            while not (terminated or truncated):
                # Epsilon is the chance of picking a random action.
                # It starts high to explore and decreases with the learning steps.
                epsilon = max(args.epsilon_end, args.epsilon_start -
                              (args.epsilon_start - args.epsilon_end) * agent.training_steps / max(1, args.epsilon_decay_steps))
                # The agent receives eight sensors and chooses an action from 0 to 5.
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                # reward is the original reward; train_reward can include
                # the optional hint that helps learn progress step by step.
                train_reward = shaped_reward(
                    state, next_state, reward, terminated, args.reward_shaping,
                    gamma=args.gamma, truncated=truncated, scale=args.shaping_scale,
                    height_coef=args.shaping_height,
                    vertical_speed_coef=args.shaping_vertical_speed,
                    horizontal_position_coef=args.shaping_horizontal_position,
                    horizontal_speed_coef=args.shaping_horizontal_speed,
                    attitude_coef=args.shaping_attitude)
                # Save what it saw, did and got. The two final indicators let us
                # distinguish crash/landing from a time limit.
                buffer.add(state, action, train_reward, next_state, terminated, truncated)
                # The metrics show the simulator's original reward,
                # so the shaping does not make the flight look artificially better.
                total_reward += reward
                # Warmup prevents training with an almost empty memory. Afterwards,
                # train_batch takes batch_size random examples and adjusts the policy.
                if len(buffer) >= max(args.batch_size, args.warmup):
                    losses.append(agent.train_batch(buffer, args.batch_size))
                # The state just returned by the physics becomes the next step's input.
                state = next_state
            mean_loss = sum(losses) / len(losses) if losses else 0.0
            # Save one row per flight: original reward, steps, exploration,
            # mean learning error, result and update count.
            writer.writerow({"episode": episode, "total_reward": total_reward, "duration_steps": env.steps,
                             "epsilon": epsilon, "mean_loss": mean_loss, "outcome": info["outcome"],
                             "training_steps": agent.training_steps})
            file.flush()
            mastered = False
            if args.success_window and episode >= args.success_window:
                recent_outcomes = []
                with csv_path.open(encoding="utf-8") as read_file:
                    recent_outcomes = [row["outcome"] for row in list(csv.DictReader(read_file))[-args.success_window:]]
                if all(outcome == "LANDED" for outcome in recent_outcomes):
                    print(f"Mastery reached: {args.success_window} consecutive landings. Training stops to preserve this model.")
                    mastered = True
            # Periodic summary to observe the trend without printing every flight.
            if episode % 50 == 0:
                recent = []
                with csv_path.open(encoding="utf-8") as read_file:
                    recent = list(csv.DictReader(read_file))[-50:]
                mean_reward = sum(float(row["total_reward"]) for row in recent) / len(recent)
                print(f"Episode {episode}: mean reward(50)={mean_reward:.2f}, epsilon={epsilon:.3f}, loss={mean_loss:.4f}")
            # The checkpoint also saves the optimizer state to resume learning;
            # model.pt saves only the network to use when piloting.
            if args.snapshot_frequency and episode % args.snapshot_frequency == 0:
                save_training_checkpoint(episode, epsilon)
                agent.save_inference_model(output / "model.pt")
                snapshot = output / f"model_episode_{episode:06d}.pt"
                agent.save_inference_model(snapshot)
                print(f"Model saved: {snapshot}")
            if mastered:
                break
    final_episode = episode
    save_training_checkpoint(final_episode, epsilon)
    agent.save_inference_model(output / "model.pt")
    final_snapshot = output / f"model_episode_{final_episode:06d}.pt"
    agent.save_inference_model(final_snapshot)
    print(f"Final model saved: {final_snapshot}")
    print(f"Metrics: {csv_path}; model: {output / 'model.pt'}")
    run_elapsed = time.perf_counter() - run_started_at
    total_elapsed = previous_elapsed + run_elapsed

    def format_duration(seconds):
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    print(f"This session's time: {format_duration(run_elapsed)}")
    print(f"Total training time (active sessions): {format_duration(total_elapsed)}")
    # Generates an SVG plot at the end, without requiring Matplotlib.
    print(f"Plot: {make_plot(csv_path)}")


if __name__ == "__main__":
    main()
