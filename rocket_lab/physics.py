"""2D rocket lander physics; no rendering or ML dependencies.

Coordinate system: x to the right, y upwards; y=0 is the ground.
Angle: 0 is upright. Positive angle tilts the thrust direction left.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

DT = 1 / 60                 # simulated seconds per step
GRAVITY = 1.62             # m/s^2, approximate lunar gravity
EARTH_GRAVITY = 9.81       # m/s^2, approximate Earth gravity
ENGINE_ACCEL = 4.5        # m/s^2 while the main engine is on
EARTH_ENGINE_ACCEL = ENGINE_ACCEL * EARTH_GRAVITY / GRAVITY
SIDE_ACCEL = 2.0          # m/s^2 from translation thrusters in the XY stage
ROTATION_ACCEL = 4.0      # rad/s^2 from side thrusters
ROTATION_DAMPING = 0.985  # per-step damping (simplified physics)
PAD_HALF_WIDTH = 1.5      # meters
SAFE_VERTICAL_SPEED = 1.0 # m/s
SAFE_HORIZONTAL_SPEED = 0.7
SAFE_ANGLE = math.radians(12)
MAX_STEPS = 60 * 30
OBSERVATION_NAMES = ("x", "y", "vx", "vy", "angle", "angular_velocity", "dx", "dy")
OBSERVATION_SIZE = len(OBSERVATION_NAMES)


def finite_number(value: float, name: str) -> float:
    """Convert a numeric configuration value and reject NaN or infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def upright_angle(angle: float) -> float:
    """Return the equivalent orientation closest to upright, in radians."""
    if -math.pi <= angle <= math.pi:
        return angle
    return (angle + math.pi) % (2 * math.pi) - math.pi


def policy_observation(state: tuple[float, ...]) -> tuple[float, ...]:
    """Use the same orientation for PPO after any number of full rotations."""
    return (*state[:4], upright_angle(state[4]), *state[5:])

# (main_engine, turn): turn=+1 tilts left, turn=-1 tilts right.
ACTIONS = {
    0: (0, 0),   # coast
    1: (1, 0),   # main engine
    2: (0, 1),   # rotate left
    3: (0, -1),  # rotate right
    4: (1, 1),   # main + left
    5: (1, -1),  # main + right
}

# Stage 2 controls translation in X without changing the rocket's angle.
# (main_engine, lateral_thrust): +1 moves right, -1 moves left.
XY_ACTIONS = {
    6: (0, 1),
    7: (0, -1),
    8: (1, 1),
    9: (1, -1),
}
XY_ALLOWED_ACTIONS = (0, 1, 6, 7, 8, 9)
# Preserve the XY actor's first six outputs when adding rotation controls.
XY_ANGLE_ALLOWED_ACTIONS = (*XY_ALLOWED_ACTIONS, 2, 3, 4, 5)


@dataclass
class RocketState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float

    def as_observation(self) -> tuple[float, ...]:
        """The six physical state values, before adding pad-relative sensors."""
        return (self.x, self.y, self.vx, self.vy, self.angle, self.angular_velocity)


class RocketEnv:
    def __init__(self, seed: int | None = None, pad_x: float = 0.0, pad_y: float = 0.0,
                 max_horizontal_distance: float | None = None, max_height: float | None = None,
                 gravity: float = GRAVITY, engine_accel: float = ENGINE_ACCEL):
        gravity = finite_number(gravity, "gravity")
        engine_accel = finite_number(engine_accel, "engine_accel")
        if gravity <= 0:
            raise ValueError("gravity must be positive")
        if engine_accel <= 0:
            raise ValueError("engine_accel must be positive")
        pad_x = finite_number(pad_x, "pad_x")
        pad_y = finite_number(pad_y, "pad_y")
        if max_horizontal_distance is not None:
            max_horizontal_distance = finite_number(max_horizontal_distance, "max_horizontal_distance")
            if max_horizontal_distance < 0:
                raise ValueError("max_horizontal_distance must be non-negative")
        if max_height is not None:
            max_height = finite_number(max_height, "max_height")
            if max_height < 0:
                raise ValueError("max_height must be non-negative")
        self.random = random.Random(seed)
        self.gravity = gravity
        self.engine_accel = engine_accel
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.max_horizontal_distance = max_horizontal_distance
        self.max_height = max_height
        self.state = RocketState(0, 0, 0, 0, 0, 0)
        self.steps = 0
        self.finished = True
        self.last_action = 0
        self.last_outcome: str | None = None
        self.reset()

    def observation(self) -> tuple[float, ...]:
        s = self.state
        return (*s.as_observation(), self.pad_x - s.x, self.pad_y - s.y)

    def reset(self, randomize: bool = False) -> tuple[float, ...]:
        """Reset to a reproducible challenge, or randomize training scenarios."""
        self.state = RocketState(
            x=self.random.uniform(-3, 3) if randomize else 2.0,
            y=self.random.uniform(7, 10) if randomize else 9.0,
            vx=self.random.uniform(-0.5, 0.5) if randomize else 0.0,
            vy=self.random.uniform(-0.8, 0.1) if randomize else -0.4,
            angle=self.random.uniform(-0.12, 0.12) if randomize else 0.0,
            angular_velocity=0.0,
        )
        self.steps = 0
        self.finished = False
        self.last_action = 0
        self.last_outcome = None
        return self.observation()

    def reset_y_only(self) -> tuple[float, ...]:
        """Start a fixed vertical landing challenge with lateral motion disabled."""
        self.state = RocketState(x=0.0, y=9.0, vx=0.0, vy=-0.4, angle=0.0, angular_velocity=0.0)
        self.steps = 0
        self.finished = False
        self.last_action = 0
        self.last_outcome = None
        return self.observation()

    def reset_lateral(self, max_offset: float, max_speed: float = 0.0) -> tuple[float, ...]:
        """Keep the vertical challenge fixed while introducing lateral error."""
        max_offset = finite_number(max_offset, "max_offset")
        max_speed = finite_number(max_speed, "max_speed")
        if max_offset < 0.25:
            raise ValueError("max_offset must be at least 0.25")
        if max_speed < 0:
            raise ValueError("max_speed must be non-negative")
        self.state = RocketState(
            x=self.random.choice((-1.0, 1.0)) * self.random.uniform(0.25, max_offset),
            y=9.0,
            vx=self.random.uniform(-max_speed, max_speed),
            vy=-0.4,
            angle=0.0,
            angular_velocity=0.0,
        )
        self.steps = 0
        self.finished = False
        self.last_action = 0
        self.last_outcome = None
        return self.observation()

    def reset_lateral_attitude(self, max_offset: float, max_speed: float,
                               max_angle: float, max_angular_speed: float) -> tuple[float, ...]:
        """Start an XY challenge with a random tilt and angular velocity."""
        max_angle = finite_number(max_angle, "max_angle")
        max_angular_speed = finite_number(max_angular_speed, "max_angular_speed")
        if max_angle < 0:
            raise ValueError("max_angle must be non-negative")
        if max_angular_speed < 0:
            raise ValueError("max_angular_speed must be non-negative")
        self.reset_lateral(max_offset, max_speed)
        self.state.angle = self.random.uniform(-max_angle, max_angle)
        self.state.angular_velocity = self.random.uniform(-max_angular_speed, max_angular_speed)
        return self.observation()

    def step(self, action: int) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        """Execute ONE physics step: observation, reward, terminated, truncated, info."""
        if self.finished:
            raise RuntimeError("Episode finished; call reset() before step().")
        if action in ACTIONS:
            main_engine, turn = ACTIONS[action]
            lateral_thrust = 0
        elif action in XY_ACTIONS:
            main_engine, lateral_thrust = XY_ACTIONS[action]
            turn = 0
        else:
            raise ValueError(f"Unknown action {action}; choose one of {list(ACTIONS) + list(XY_ACTIONS)}")
        s = self.state
        self.last_action = action

        # 1) Rotation; steering changes angular velocity, not angle directly.
        s.angular_velocity = (s.angular_velocity + turn * ROTATION_ACCEL * DT) * ROTATION_DAMPING
        s.angle += s.angular_velocity * DT

        # 2) Force -> acceleration. Main engine points along the rocket axis.
        ax = -math.sin(s.angle) * self.engine_accel * main_engine + lateral_thrust * SIDE_ACCEL
        ay = math.cos(s.angle) * self.engine_accel * main_engine - self.gravity

        # 3) Semi-implicit Euler integration.
        s.vx += ax * DT
        s.vy += ay * DT
        s.x += s.vx * DT
        s.y += s.vy * DT
        self.steps += 1

        # 4) Collision & reward (very sparse, on purpose for v0).
        terminated = False
        truncated = False
        reward = -0.01  # Small cost per simulation step.
        outcome: str | None = None

        if s.y <= self.pad_y:
            s.y = self.pad_y
            terminated = True
            safe = (
                abs(s.x - self.pad_x) <= PAD_HALF_WIDTH
                and abs(s.vx) <= SAFE_HORIZONTAL_SPEED
                and abs(s.vy) <= SAFE_VERTICAL_SPEED
                and abs(upright_angle(s.angle)) <= SAFE_ANGLE
                and abs(s.angular_velocity) <= 0.35
            )
            outcome = "LANDED" if safe else "CRASHED"
            reward = 100.0 if safe else -100.0
        elif ((self.max_horizontal_distance is not None and
               abs(s.x - self.pad_x) > self.max_horizontal_distance) or
              (self.max_height is not None and s.y - self.pad_y > self.max_height)):
            terminated = True
            outcome = "CRASHED"
            reward = -100.0
        elif self.steps >= MAX_STEPS:
            truncated = True
            outcome = "TIMEOUT"

        self.finished = terminated or truncated
        self.last_outcome = outcome
        return self.observation(), reward, terminated, truncated, {
            "outcome": outcome,
            "main_engine": bool(main_engine),
            "turn": turn,
            "lateral_thrust": lateral_thrust,
            "sim_time": self.steps * DT,
        }
