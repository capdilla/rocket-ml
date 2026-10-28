"""Pygame renderer and keyboard controller. RocketEnv works without pygame."""
import argparse
from dataclasses import replace
import math
from pathlib import Path
import random

import pygame

from rocket_lab.data.play_log import PlayLogger
from rocket_lab.physics import (ACTIONS, XY_ACTIONS, DT, PAD_HALF_WIDTH, RocketEnv,
                                GRAVITY, ENGINE_ACCEL, EARTH_GRAVITY, EARTH_ENGINE_ACCEL,
                                policy_observation, upright_angle)

WIDTH, HEIGHT = 960, 740
GROUND_Y = 675
PIXELS_PER_METER = 48
BG = (11, 18, 33)
FG = (234, 238, 247)
MUTED = (162, 177, 197)
DEFAULT_MODELS = {
    "y": "runs/ppo-y-8/model.pt",
    "xy": "runs/ppo-xy-from-y/model.pt",
    "xy-angle": "runs/ppo-xy-angle-from-xy/model.pt",
}


def screen_point(x: float, y: float) -> tuple[int, int]:
    return (int(WIDTH / 2 + x * PIXELS_PER_METER), int(GROUND_Y - y * PIXELS_PER_METER))


def local_point(x: float, y: float, bx: float, by: float, angle: float):
    return screen_point(
        bx + x * math.cos(angle) - y * math.sin(angle),
        by + x * math.sin(angle) + y * math.cos(angle),
    )


def choose_action(keys) -> int:
    engine = bool(keys[pygame.K_UP] or keys[pygame.K_w])
    left = bool(keys[pygame.K_LEFT] or keys[pygame.K_q])
    right = bool(keys[pygame.K_RIGHT] or keys[pygame.K_d])
    turn = int(left) - int(right)
    return next(k for k, v in ACTIONS.items() if v == (int(engine), turn))


def reset_scenario(env: RocketEnv, mode: str, randomize: bool = False):
    """Use the starting scenario that corresponds to the chosen PPO mode."""
    if mode == "y":
        return env.reset(randomize=True) if randomize else env.reset_y_only()
    if mode == "xy":
        return env.reset_lateral(3.5, 0.5)
    if mode == "xy-angle":
        return env.reset_lateral_attitude(3.5, 0.5, 0.45, 0.3)
    return env.reset(randomize=True)


def draw_rocket(screen, env: RocketEnv):
    s = env.state
    point = lambda x, y: local_point(x, y, s.x, s.y, s.angle)

    # Fins & legs, drawn in the same world coordinate frame as the hull.
    pygame.draw.line(screen, (160, 177, 204), point(-0.20, 0.24), point(-0.55, 0.02), 3)
    pygame.draw.line(screen, (160, 177, 204), point(0.20, 0.24), point(0.55, 0.02), 3)
    pygame.draw.line(screen, (160, 177, 204), point(-0.68, 0.02), point(-0.42, 0.02), 3)
    pygame.draw.line(screen, (160, 177, 204), point(0.42, 0.02), point(0.68, 0.02), 3)

    action_components = ACTIONS.get(env.last_action, XY_ACTIONS.get(env.last_action))
    if not env.finished and action_components[0]:
        flame = [point(-0.13, 0.07), point(0.13, 0.07), point(0, -0.60)]
        pygame.draw.polygon(screen, (255, 155, 59), flame)
    if not env.finished and env.last_action in XY_ACTIONS:
        side = XY_ACTIONS[env.last_action][1]
        x = -0.24 if side > 0 else 0.24
        flame = [point(x, 0.54), point(x, 0.68), point(x - side * 0.28, 0.61)]
        pygame.draw.polygon(screen, (71, 164, 245), flame)

    hull = [point(-0.24, 0.08), point(0.24, 0.08), point(0.22, 0.86),
            point(0, 1.30), point(-0.22, 0.86)]
    pygame.draw.polygon(screen, FG, hull)
    pygame.draw.polygon(screen, (91, 112, 146), hull, 2)
    pygame.draw.circle(screen, (71, 164, 245), point(0, 0.73), 7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Seed of the scenario generator")
    parser.add_argument("--mode", choices=("y", "xy", "xy-angle", "full"), default="y",
                        help="Initial scenario to play")
    parser.add_argument("--planet", choices=("moon", "earth"), default="moon",
                        help="Lunar or Earth physics; on Earth the engine is also more powerful")
    parser.add_argument("--model", help="PPO model for the A key; by default the one for --mode is used")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/play"),
                        help="Folder to save the flights; runs/play by default")
    parser.add_argument("--pad-range", type=float, default=0.0, metavar="METERS",
                        help="Random pad in X between -METERS and +METERS in each scenario")
    args = parser.parse_args()
    if not math.isfinite(args.pad_range) or args.pad_range < 0:
        parser.error("pad-range must be a finite, non-negative number")

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Rocket Lab — play ({args.planet}, mode={args.mode}, seed={args.seed})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 19)
    small = pygame.font.SysFont("monospace", 15)

    env = RocketEnv(seed=args.seed,
                    gravity=EARTH_GRAVITY if args.planet == "earth" else GRAVITY,
                    engine_accel=EARTH_ENGINE_ACCEL if args.planet == "earth" else ENGINE_ACCEL)
    pad_random = random.Random(args.seed)
    env.pad_x = pad_random.uniform(-args.pad_range, args.pad_range) if args.pad_range else 0.0
    reset_scenario(env, args.mode)
    initial_state = replace(env.state)
    initial_pad_x = env.pad_x
    model_path = args.model or DEFAULT_MODELS.get(args.mode)
    logger = PlayLogger(args.log_dir)
    logger.start(env, args.mode, args.seed, str(Path(model_path).resolve()) if model_path else None)
    ai_agent = None
    ai_mode = None
    ai_enabled = False
    ai_message = ""
    paused = False
    last_reward = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    logger.finish(env, "reset", ai_mode)
                    randomize = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    if randomize:
                        env.pad_x = (pad_random.uniform(-args.pad_range, args.pad_range)
                                     if args.pad_range else 0.0)
                        reset_scenario(env, args.mode, randomize=True)
                        initial_state = replace(env.state)
                        initial_pad_x = env.pad_x
                    else:
                        env.pad_x = initial_pad_x
                        env.state = replace(initial_state)
                        env.steps = 0
                        env.finished = False
                        env.last_action = 0
                        env.last_outcome = None
                    logger.start(env, args.mode, args.seed, str(Path(model_path).resolve()) if model_path else None)
                    last_reward = 0
                    paused = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_a:
                    if ai_enabled:
                        ai_enabled = False
                        ai_message = "Manual control"
                    elif model_path is None:
                        ai_message = "Provide a PPO model with --model"
                    else:
                        if ai_agent is None:
                            try:
                                from rocket_lab.ml.ppo_agent import PPOAgent
                                ai_agent, ai_mode = PPOAgent.from_model(model_path)
                            except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
                                ai_message = f"Could not load AI: {error}"
                        if ai_agent is not None:
                            ai_enabled = True
                            ai_message = f"AI activates ({ai_mode})"
                            if ai_agent.planet != args.planet:
                                ai_message += f"; model trained on {ai_agent.planet}"
                elif event.key == pygame.K_f:
                    logger.mark_for_review()
                    ai_message = "Flight marked for review"

        if running and not env.finished and not paused:
            state = env.observation()
            ai_action = (ai_agent.select_action(policy_observation(state), deterministic=True)[0]
                         if ai_agent is not None else None)
            action = ai_action if ai_enabled else choose_action(pygame.key.get_pressed())
            obs, last_reward, terminated, truncated, info = env.step(action)
            logger.record(state, action, last_reward, obs, terminated, truncated, info,
                          "ai" if ai_enabled else "manual", ai_action, ai_mode)
            if env.finished:
                logger.finish(env, "finished", ai_mode)

        screen.fill(BG)
        # Floor and landing pad.
        ground_y = screen_point(0, env.pad_y)[1]
        pygame.draw.line(screen, (72, 91, 117), (0, ground_y), (WIDTH, ground_y), 3)
        left_pad = screen_point(env.pad_x - PAD_HALF_WIDTH, env.pad_y)
        right_pad = screen_point(env.pad_x + PAD_HALF_WIDTH, env.pad_y)
        pygame.draw.line(screen, (68, 211, 143), left_pad, right_pad, 9)
        pygame.draw.line(screen, (55, 88, 105), screen_point(env.pad_x, env.pad_y), screen_point(env.pad_x, env.pad_y + 14), 1)
        draw_rocket(screen, env)

        s = env.state
        lines = [
            f"Rocket Lab | {args.planet.upper()} | mode {args.mode} | pilot: {'AI ' + ai_mode if ai_enabled else 'MANUAL'}",
            "A: activate/deactivate AI   UP/W: engine   LEFT/Q, RIGHT/D: rotate",
            "R: repeat   SHIFT+R: new scenario   F: review   P: pause   ESC: exit",
            f"x={s.x:+.2f}m  y={s.y:+.2f}m  vx={s.vx:+.2f}m/s  vy={s.vy:+.2f}m/s",
            f"angle={math.degrees(upright_angle(s.angle)):+.1f}deg  angular_v={s.angular_velocity:+.2f}rad/s",
            f"g={env.gravity:.2f}m/s²  engine={env.engine_accel:.2f}m/s²",
            f"t={env.steps * DT:.2f}s   action={env.last_action}   reward={last_reward:+.2f}",
        ]
        if ai_message:
            lines.append(ai_message[:100])
        for i, line in enumerate(lines):
            color = (76, 221, 151) if i == 0 and ai_enabled else FG if i == 0 else MUTED
            label = small.render(line, True, color)
            screen.blit(label, (20, 20 + i * 25))

        if env.finished or paused:
            title = env.last_outcome if env.finished else "PAUSED"
            color = (76, 221, 151) if title == "LANDED" else (255, 110, 115)
            if title == "PAUSED":
                color = FG
            headline = font.render(f"{title}  |  Press R to restart", True, color)
            screen.blit(headline, headline.get_rect(center=(WIDTH / 2, HEIGHT / 2)))

        pygame.display.flip()
        clock.tick(round(1 / DT))
    logger.finish(env, "exit", ai_mode)
    pygame.quit()


if __name__ == "__main__":
    main()
