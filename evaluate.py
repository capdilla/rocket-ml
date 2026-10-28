"""Tests an already trained pilot and counts how well it does.

This file does not teach or change the network: it only feeds it states and
follows the highest-scoring action. It runs several flights and compares the
result with a pilot that picks at random. Add ``--render`` to watch the flight
in Pygame. Training is done with ``train.py``; here the ``model.pt`` file is
loaded.
"""
import argparse
from datetime import datetime
from pathlib import Path
import random

from rocket_lab.ml.agents import RandomAgent
from rocket_lab.ml.dqn_agent import DQNAgent
from rocket_lab.physics import RocketEnv


def run_episode(env, action_fn, render=False, mode="full", show_final_state=False,
                reset_fn=None):
    """Runs a complete flight with the given policy and returns its metrics."""
    if reset_fn is not None:
        state = reset_fn(env)
    elif mode == "y":
        state = env.reset_y_only()
    elif mode == "xy":
        state = env.reset_lateral(3.5, 0.5)
    elif mode == "xy-angle":
        state = env.reset_lateral_attitude(3.5, 0.5, 0.45, 0.3)
    else:
        state = env.reset(randomize=True)
    total_reward = 0.0
    terminated = truncated = False
    info = {"outcome": None}
    if render:
        # Pygame is only imported/opened when showing the flight was requested.
        import pygame
        from play import BG, WIDTH, HEIGHT, draw_rocket, PAD_HALF_WIDTH, screen_point
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
    while not (terminated or truncated):
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt
        # Evaluation does not explore: it always picks the highest-scoring action.
        action = action_fn(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render:
            screen.fill(BG)
            ground_y = screen_point(0, env.pad_y)[1]
            pygame.draw.line(screen, (72, 91, 117), (0, ground_y), (WIDTH, ground_y), 3)
            pygame.draw.line(screen, (68, 211, 143), screen_point(env.pad_x - PAD_HALF_WIDTH, env.pad_y),
                             screen_point(env.pad_x + PAD_HALF_WIDTH, env.pad_y), 9)
            draw_rocket(screen, env)
            pygame.display.flip()
            pygame.time.delay(16)
    if render and show_final_state:
        screen.fill(BG)
        ground_y = screen_point(0, env.pad_y)[1]
        pygame.draw.line(screen, (72, 91, 117), (0, ground_y), (WIDTH, ground_y), 3)
        pygame.draw.line(screen, (68, 211, 143), screen_point(env.pad_x - PAD_HALF_WIDTH, env.pad_y),
                         screen_point(env.pad_x + PAD_HALF_WIDTH, env.pad_y), 9)
        draw_rocket(screen, env)
        font = pygame.font.SysFont("monospace", 20, bold=True)
        state = env.state
        observation = env.observation()
        lines = [
            f"{info['outcome']} — final state ({env.steps} steps)",
            f"x={state.x:+.2f} m  pad={env.pad_x:+.2f} m  dx={observation[6]:+.2f} m",
            f"vx={state.vx:+.2f} m/s  vy={state.vy:+.2f} m/s",
            f"angle={state.angle:+.3f} rad  rotation={state.angular_velocity:+.3f} rad/s",
        ]
        panel = pygame.Surface((WIDTH - 80, 150), pygame.SRCALPHA)
        panel.fill((11, 18, 33, 220))
        screen.blit(panel, (40, 35))
        for index, line in enumerate(lines):
            label = font.render(line, True, (76, 221, 151) if index == 0 and info["outcome"] == "LANDED"
                                else (255, 110, 115) if index == 0 else (234, 238, 247))
            screen.blit(label, (58, 48 + index * 32))
        pygame.display.flip()
        # Keeps the outcome visible before closing this episode's window.
        deadline = pygame.time.get_ticks() + 2500
        while pygame.time.get_ticks() < deadline:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    deadline = 0
                    break
            pygame.time.delay(20)
    if render:
        pygame.quit()
    return info["outcome"], total_reward, env.steps


def summarize(results):
    """Summarizes landing rate, mean reward and mean duration."""
    return {"landing_percent": 100 * sum(r[0] == "LANDED" for r in results) / len(results),
            "mean_reward": sum(r[1] for r in results) / len(results),
            "mean_duration_steps": sum(r[2] for r in results) / len(results)}


def choose_model(root=Path("runs")):
    """Shows models and checkpoints by modification date, newest first."""
    candidates = []
    for path in root.rglob("model_episode_*.pt"):
        candidates.append(path)
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not find snapshots model_episode_*.pt in {root.resolve()}")

    print("Available models (newest first):")
    for index, path in enumerate(candidates, start=1):
        modified = datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M")
        print(f"  {index}. {path}  ({modified})")
    while True:
        choice = input("Choose a number, or q to exit: ").strip().lower()
        if choice == "q":
            raise SystemExit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print(f"Enter a number between 1 and {len(candidates)}, or q to exit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="model.pt/checkpoint.pt file; if omitted, a runs/ selector appears")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mode", choices=("auto", "y", "full"), default="auto",
                        help="Scenario to evaluate; auto reads the actions saved in the model")
    parser.add_argument("--render", action="store_true", help="Show the flight with Pygame")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    model_path = Path(args.model) if args.model else choose_model()
    agent = DQNAgent.from_inference_model(model_path)
    agent.policy_network.eval()
    mode = ("y" if agent.allowed_actions == (0, 1) else "full") if args.mode == "auto" else args.mode
    if mode == "y":
        agent.allowed_actions = (0, 1)
    env = RocketEnv(seed=args.seed)
    # Here we only ask the network for actions; no errors are computed and no weights change.
    trained = [run_episode(env, lambda state: agent.select_action(state), args.render, mode) for _ in range(args.episodes)]
    random_env = RocketEnv(seed=args.seed)
    random_agent = RandomAgent(seed=args.seed)
    vertical_random = random.Random(args.seed)
    # Same initial scenarios and number of episodes for a simple baseline.
    random_action = ((lambda state: vertical_random.choice((0, 1))) if mode == "y"
                     else (lambda state: random_agent.select_action(state)))
    baseline = [run_episode(random_env, random_action, mode=mode) for _ in range(args.episodes)]
    print("DQN greedy:", summarize(trained))
    print("Random agent:", summarize(baseline))
    print(f"Model tested: {model_path}")


if __name__ == "__main__":
    main()
