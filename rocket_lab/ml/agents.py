"""Simple reference agents; ``RandomAgent`` picks among the six actions.

It is kept as a Phase 2 baseline and to compare it against the DQN in evaluate.py.
"""
from __future__ import annotations
import random  # Standard library random generator; requires no ML.

from ..physics import ACTIONS  # Valid actions defined by the simulator (0 to 5).


class RandomAgent:
    """Base agent that picks at random among all available actions."""

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)  # The seed makes the same sequence reproducible.

    def select_action(self, observation: tuple[float, ...]) -> int:
        # This agent ignores the state; future agents can base their decision on it.
        del observation  # Explicitly indicates this argument is unused for now.
        return self.random.choice(tuple(ACTIONS))  # Pick one of the identifiers 0-5.
