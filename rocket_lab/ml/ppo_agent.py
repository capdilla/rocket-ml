"""Actor and critic for PPO with the rocket's discrete actions."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from ..physics import ACTIONS, XY_ACTIONS, XY_ALLOWED_ACTIONS, XY_ANGLE_ALLOWED_ACTIONS, OBSERVATION_SIZE


class PPOAgent(nn.Module):
    def __init__(self, allowed_actions=None):
        super().__init__()
        self.allowed_actions = tuple(ACTIONS) if allowed_actions is None else tuple(allowed_actions)
        if not self.allowed_actions or any(action not in ACTIONS and action not in XY_ACTIONS
                                           for action in self.allowed_actions):
            raise ValueError("allowed_actions must contain valid actions")
        self.register_buffer("observation_scale", torch.tensor([3.0, 10.0, 2.0, 3.0, 0.5, 1.0, 3.0, 10.0]))
        self.body = nn.Sequential(nn.Linear(OBSERVATION_SIZE, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, len(self.allowed_actions))
        self.critic = nn.Linear(64, 1)

    def forward(self, observations):
        features = self.body(observations / self.observation_scale)
        return self.actor(features), self.critic(features).squeeze(-1)

    @torch.no_grad()
    def select_action(self, observation, deterministic=False):
        state = torch.as_tensor(observation, dtype=torch.float32)
        logits, value = self(state)
        distribution = Categorical(logits=logits)
        index = logits.argmax() if deterministic else distribution.sample()
        return (self.allowed_actions[int(index)], int(index),
                float(distribution.log_prob(index)), float(value))

    def save_model(self, path, mode, planet="moon"):
        torch.save({"algorithm": "ppo", "policy": self.state_dict(), "input_size": OBSERVATION_SIZE,
                    "allowed_actions": self.allowed_actions, "mode": mode, "planet": planet}, path)

    @classmethod
    def from_model(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("algorithm") != "ppo":
            raise ValueError(f"{path} is not a PPO model")
        size = saved["policy"]["body.0.weight"].shape[1]
        if size != OBSERVATION_SIZE or saved.get("input_size", size) != OBSERVATION_SIZE:
            raise ValueError(f"{path} has {size} PPO inputs; {OBSERVATION_SIZE} are required. "
                             "Start a new training run in another folder")
        agent = cls(saved["allowed_actions"])
        agent.load_state_dict(saved["policy"])
        agent.eval()
        agent.planet = saved.get("planet", "moon")
        return agent, saved["mode"]

    @classmethod
    def from_vertical_model(cls, path, mode="xy"):
        """Expands PPO-Y while keeping the vertical decisions already learned."""
        if mode != "xy":
            raise ValueError("Transfer from PPO-Y is only available in mode XY")
        source, source_mode = cls.from_model(path)
        if source_mode != "y" or source.allowed_actions != (0, 1):
            raise ValueError(f"{path} is not a PPO model trained in mode Y")
        target = cls(XY_ALLOWED_ACTIONS)
        target.body.load_state_dict(source.body.state_dict())
        target.critic.load_state_dict(source.critic.state_dict())
        # Each new action starts from the equivalent vertical decision, with a
        # lower initial probability so it can explore without losing the descent.
        with torch.no_grad():
            # In the Y phase, x/vx/dx were always zero. Their weights were left
            # untrained and would perturb the vertical pilot by moving the craft in X.
            target.body[0].weight[:, (0, 2, 6)] = 0
            target.actor.weight[:2].copy_(source.actor.weight)
            target.actor.bias[:2].copy_(source.actor.bias)
            for index in range(2, len(target.allowed_actions)):
                base = 0 if target.allowed_actions[index] in (6, 7) else 1
                target.actor.weight[index].copy_(source.actor.weight[base])
                target.actor.bias[index].copy_(source.actor.bias[base] - 4.5)
        return target

    @classmethod
    def from_xy_model(cls, path):
        """Add rotation controls to PPO-XY without discarding lateral control."""
        source, source_mode = cls.from_model(path)
        if source_mode != "xy" or source.allowed_actions != XY_ALLOWED_ACTIONS:
            raise ValueError(f"{path} is not a PPO model trained in mode XY")
        target = cls(XY_ANGLE_ALLOWED_ACTIONS)
        target.body.load_state_dict(source.body.state_dict())
        target.critic.load_state_dict(source.critic.state_dict())
        with torch.no_grad():
            # Angle inputs were always zero in XY, so their old weights are untrained.
            target.body[0].weight[:, (4, 5)] = 0
            target.actor.weight[:len(XY_ALLOWED_ACTIONS)].copy_(source.actor.weight)
            target.actor.bias[:len(XY_ALLOWED_ACTIONS)].copy_(source.actor.bias)
            for index, action in enumerate(target.allowed_actions[len(XY_ALLOWED_ACTIONS):],
                                           start=len(XY_ALLOWED_ACTIONS)):
                base = source.allowed_actions.index(1 if action in (4, 5) else 0)
                target.actor.weight[index].copy_(source.actor.weight[base])
                target.actor.bias[index].copy_(source.actor.bias[base] - 3.5)
        return target
