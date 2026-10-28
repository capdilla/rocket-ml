"""Defines the experience format and saves/loads them as JSON."""
from __future__ import annotations  # Allows forward type references.

from dataclasses import asdict, dataclass  # Data classes and conversion to dictionaries.
import json  # Writes and reads the standard JSON format.
from pathlib import Path  # Handles file paths across operating systems.
from typing import Iterable  # Indicates that a sequence of episodes is accepted.

from ..physics import RocketEnv, RocketState, OBSERVATION_SIZE  # Used to replay actions.


@dataclass
class Transition:
    """One experience step: (state, action, reward, next state, end)."""
    state: tuple[float, ...]  # The eight values observed before acting.
    action: int  # Identifier of the action the agent executed (0-5).
    reward: float  # Reward returned by env.step().
    next_state: tuple[float, ...]  # The eight values observed after acting.
    terminated: bool  # Natural end: landing or crash.
    truncated: bool  # End by step limit.

    def to_dict(self) -> dict:
        data = asdict(self)  # Converts this transition's fields into a dictionary.
        data["state"] = list(self.state)  # JSON represents sequences as lists.
        data["next_state"] = list(self.next_state)  # Converts the next state as well.
        return data  # Returns an object ready for json.dump().


@dataclass
class Episode:
    """A complete flight made up of its steps and the final result."""
    transitions: list[Transition]  # Experiences in the order they happened.
    outcome: str  # Result: LANDED, CRASHED or TIMEOUT.
    total_reward: float  # Sum of the rewards of all its steps.

    @property
    def duration_steps(self) -> int:
        return len(self.transitions)  # Duration measured in simulation steps.

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,  # Saves the human-readable outcome of the episode.
            "total_reward": self.total_reward,  # Saves the total rewards.
            "duration_steps": self.duration_steps,  # Saves how many steps it had.
            "transitions": [transition.to_dict() for transition in self.transitions],  # Saves each step.
        }


def save_episodes(episodes: Iterable[Episode], path: str | Path) -> None:
    """Saves selected episodes as readable JSON."""
    with Path(path).open("w", encoding="utf-8") as file:  # Opens/creates the file for writing.
        json.dump([episode.to_dict() for episode in episodes], file, indent=2)  # Serializes the list.
        file.write("\n")  # Adds a trailing newline so the file ends cleanly.


def load_episodes(path: str | Path) -> list[dict]:
    """Loads the JSON episodes as a list of dictionaries."""
    with Path(path).open(encoding="utf-8") as file:  # Opens the file in read mode.
        return json.load(file)  # Converts the JSON text to Python objects.


def replay_episode(env: RocketEnv, episode_data: dict) -> list[dict]:
    """Repeats the saved actions and returns the computed results."""
    transitions = episode_data["transitions"]  # Extracts the steps from the JSON dictionary.
    if not transitions:  # An empty episode has nothing to replay.
        return []
    initial_state = transitions[0]["state"]  # The first step contains the initial state.
    if len(initial_state) != OBSERVATION_SIZE:  # Checks the environment contract.
        raise ValueError("The saved state must contain eight values")
    if tuple(initial_state[6:]) != (env.pad_x - initial_state[0], env.pad_y - initial_state[1]):
        raise ValueError("The relative sensors do not match the environment's pad")
    env.gravity = episode_data.get("gravity", env.gravity)
    env.engine_accel = episode_data.get("engine_accel", env.engine_accel)
    env.state = RocketState(*initial_state[:6])  # Places the rocket exactly in that state.
    env.steps = 0  # Resets the simulated time counter.
    env.finished = False  # Allows step() to accept actions.
    env.last_action = 0  # Resets the data used by the interface to show the rocket.
    env.last_outcome = None  # Clears any previous result.
    replayed = []  # The simulator's produced output is stored here.
    for expected in transitions:  # Walks through the original actions in order.
        if env.finished:  # A finished episode cannot receive additional steps.
            raise ValueError("The saved episode has actions after its end")
        observation, reward, terminated, truncated, info = env.step(expected["action"])  # Repeats the action.
        replayed.append({
            "state": list(expected["state"]),  # Previous state recorded in the file.
            "action": expected["action"],  # Repeated action.
            "reward": reward,  # Reward produced when repeating it.
            "next_state": list(observation),  # State the simulator produced now.
            "terminated": terminated,  # Whether the episode ended by the physical result.
            "truncated": truncated,  # Whether it ended by the step limit.
            "outcome": info["outcome"],  # Result reported by RocketEnv.
        })
    return replayed  # Returns the results for inspection or comparison.
