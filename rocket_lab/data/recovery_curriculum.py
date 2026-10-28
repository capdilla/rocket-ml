"""Start new PPO rollouts near manual-to-AI handovers from play logs.

The recorded actions are never used as labels: PPO must try its own actions
from these states and learn from the resulting rewards.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .play_log import load_play_episode
from ..physics import (OBSERVATION_SIZE, RocketEnv, RocketState, GRAVITY, ENGINE_ACCEL,
                       policy_observation, upright_angle)


@dataclass(frozen=True)
class RecoveryStart:
    path: Path
    step: int
    state: tuple[float, ...]
    pad_x: float
    pad_y: float


def load_recovery_starts(paths: list[Path], step: int | None = None,
                         gravity: float | None = GRAVITY,
                         engine_accel: float = ENGINE_ACCEL) -> list[RecoveryStart]:
    """Choose AI handovers, or an explicit step, from each play log (any mode).

    Pass ``gravity=None`` to accept logs from any planet (mixed-planet training).
    """
    starts = []
    for path in paths:
        episode = load_play_episode(path)
        if episode["mode"] not in ("y", "xy", "xy-angle", "full"):
            raise ValueError(f"{path} has an unknown flight mode")
        if gravity is not None and (
                not math.isclose(episode["gravity"], gravity, rel_tol=1e-9)
                or not math.isclose(episode["engine_accel"], engine_accel, rel_tol=1e-9)):
            raise ValueError(f"{path} uses a different gravity or engine power")
        transitions = episode["transitions"]
        if step is None:
            selected = [transition for index, transition in enumerate(transitions)
                        if transition.get("controller") == "ai"
                        and (index == 0 or transitions[index - 1].get("controller") != "ai")]
        else:
            selected = [transition for transition in transitions if transition.get("step") == step]
        if not selected:
            detail = f"step {step}" if step is not None else "handover manual→AI"
            raise ValueError(f"{path} does not contain {detail}")
        for transition in selected:
            observation = transition["state"]
            if len(observation) != OBSERVATION_SIZE or not all(math.isfinite(v) for v in observation):
                raise ValueError(f"{path}, step {transition['step']}: invalid observation")
            pad_x, pad_y = episode["pad_x"], episode["pad_y"]
            if (not math.isfinite(pad_x) or not math.isfinite(pad_y)
                    or not math.isclose(observation[6], pad_x - observation[0], abs_tol=1e-6)
                    or not math.isclose(observation[7], pad_y - observation[1], abs_tol=1e-6)
                    or observation[1] <= pad_y):
                raise ValueError(f"{path}, step {transition['step']}: impossible initial state")
            starts.append(RecoveryStart(Path(path), transition["step"], tuple(observation[:6]),
                                        pad_x, pad_y))
    return starts


def reset_recovery_scenario(env: RocketEnv, start: RecoveryStart, progress: float,
                            augment: bool = True) -> tuple[float, ...]:
    """Restart near a logged handover, gradually restoring its full severity."""
    if not 0 <= progress <= 1:
        raise ValueError("progress must be between 0 and 1")
    env.pad_x, env.pad_y = start.pad_x, start.pad_y
    env.reset()
    x, y, vx, vy, angle, spin = start.state
    severity = 0.2 + 0.8 * progress
    mirror = env.random.choice((-1, 1)) if augment else 1
    if augment:
        x = start.pad_x + mirror * (x - start.pad_x) + env.random.uniform(-0.4, 0.4)
        y += env.random.uniform(-0.5, 0.5)
        vx = mirror * vx + env.random.uniform(-0.15, 0.15)
        vy += env.random.uniform(-0.2, 0.2)
        angle = mirror * upright_angle(angle) * severity + env.random.uniform(-0.08, 0.08)
        spin = mirror * spin * severity + env.random.uniform(-0.12, 0.12)
    else:
        angle = upright_angle(angle) * severity
        spin *= severity
    if augment:
        y = max(start.pad_y + 1.0, y)
        if env.max_height is not None:
            y = min(y, start.pad_y + env.max_height - 0.5)
        if env.max_horizontal_distance is not None:
            limit = env.max_horizontal_distance - 0.5
            x = max(start.pad_x - limit, min(start.pad_x + limit, x))
    env.state = RocketState(x, y, vx, vy, angle, spin)
    return policy_observation(env.observation())
