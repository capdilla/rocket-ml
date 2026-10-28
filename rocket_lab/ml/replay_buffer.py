"""Stores examples of what the rocket did so the network can learn.

Each example (also called an experience) records: what the rocket saw, what
action it took, what reward it received, and what it saw afterwards. Training
takes a few examples at random each time; that group is called a minibatch.
When the memory fills up, the oldest examples are dropped.

To explore what is stored, look at ``buffer.experiences``. ``sample`` returns
plain Python examples and ``sample_tensors`` prepares them for PyTorch.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch
from ..physics import OBSERVATION_SIZE


@dataclass(frozen=True)
class Experience:
    """A single-step example: state, action, reward and the step's result."""
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """Bounded memory of examples; lets training review them in random groups."""
    def __init__(self, capacity: int, seed: int | None = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.experiences: deque[Experience] = deque(maxlen=capacity)  # The list remembers only the latest examples.
        self.random = random.Random(seed)  # A seed makes the example order reproducible.

    def __len__(self) -> int:
        return len(self.experiences)

    def state_dict(self) -> dict:
        """Returns the experiences and the random state to resume sampling."""
        return {"capacity": self.capacity, "experiences": list(self.experiences),
                "random_state": self.random.getstate()}

    def load_state_dict(self, state: dict) -> None:
        """Restores the saved experiences, respecting the current capacity."""
        if state["capacity"] != self.capacity:
            raise ValueError(f"The saved capacity ({state['capacity']}) does not match the current one ({self.capacity})")
        if any(len(e.state) != OBSERVATION_SIZE or len(e.next_state) != OBSERVATION_SIZE
               for e in state["experiences"]):
            raise ValueError("The replay buffer uses outdated observations; start a new training run")
        self.experiences.clear()
        self.experiences.extend(state["experiences"])
        self.random.setstate(state["random_state"])

    def add(self, state, action, reward, next_state, terminated, truncated) -> None:
        # Validate the RocketBrain contract before saving the data.
        if len(state) != OBSERVATION_SIZE or len(next_state) != OBSERVATION_SIZE:
            raise ValueError(f"state and next_state must have {OBSERVATION_SIZE} values")
        # Convert to simple, immutable types so they cannot change from outside.
        self.experiences.append(Experience(tuple(map(float, state)), int(action), float(reward),
                                           tuple(map(float, next_state)), bool(terminated), bool(truncated)))

    def sample(self, batch_size: int) -> list[Experience]:
        # Pick examples at random so the network does not learn only from similar consecutive steps.
        if batch_size < 1 or batch_size > len(self):
            raise ValueError("batch_size must be between 1 and the buffer size")
        return self.random.sample(list(self.experiences), batch_size)

    def sample_tensors(self, batch_size: int, device="cpu") -> dict[str, torch.Tensor]:
        batch = self.sample(batch_size)
        # Each field becomes a separate tensor to vectorize the DQN computation.
        return {
            "states": torch.tensor([e.state for e in batch], dtype=torch.float32, device=device),
            "actions": torch.tensor([e.action for e in batch], dtype=torch.long, device=device),
            "rewards": torch.tensor([e.reward for e in batch], dtype=torch.float32, device=device),
            "next_states": torch.tensor([e.next_state for e in batch], dtype=torch.float32, device=device),
            "terminated": torch.tensor([e.terminated for e in batch], dtype=torch.bool, device=device),
            "truncated": torch.tensor([e.truncated for e in batch], dtype=torch.bool, device=device),
        }
