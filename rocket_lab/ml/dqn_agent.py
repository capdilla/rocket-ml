"""The pilot that learns to choose actions using the RocketBrain network.

The network outputs six scores, one per action; here we call them Q values. The
policy network is the pilot that is learning. The target network is a copy that
helps compute more stable targets. Every few steps we copy the policy's changes
over to the target.

``select_action`` sometimes tests a random action (explores) and other times
chooses the action with the highest score (exploits what it learned).
``train_batch`` teaches the network with saved examples: it predicts scores,
computes how wrong it was (loss), computes how to tune the weights (gradients),
and updates them.
"""
from __future__ import annotations

import random

import torch
from torch import nn

from .brain import RocketBrain
from .replay_buffer import ReplayBuffer
from ..physics import ACTIONS, OBSERVATION_SIZE


def check_observation_size(saved, path):
    """Reject old or mislabeled weights before loading any parameters."""
    size = saved["policy"]["network.0.weight"].shape[1]
    if size != OBSERVATION_SIZE or saved.get("input_size", size) != OBSERVATION_SIZE:
        raise ValueError(f"{path} has {size} inputs; {OBSERVATION_SIZE} are required. "
                         "Start a new training run in another folder")


class DQNAgent:
    """Decides actions and learns to estimate the Q value of each possible action."""
    def __init__(self, learning_rate=1e-3, gamma=0.99, target_update_frequency=500,
                 device="cpu", seed=None, gradient_clip=10.0, allowed_actions=None):
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        self.policy_network = RocketBrain().to(self.device)  # The main pilot: this network changes as it learns.
        self.target_network = RocketBrain().to(self.device)  # Reference copy for computing the targets.
        self.target_network.load_state_dict(self.policy_network.state_dict())  # They start identical.
        self.target_network.eval()  # The target network only computes predictions; it is not trained directly.
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = nn.SmoothL1Loss()  # Huber measures the error and keeps huge errors from dominating.
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.gradient_clip = gradient_clip
        self.random = random.Random(seed)
        self.training_steps = 0
        self.allowed_actions = tuple(ACTIONS) if allowed_actions is None else tuple(allowed_actions)
        if not self.allowed_actions or any(action not in ACTIONS for action in self.allowed_actions):
            raise ValueError("allowed_actions must contain valid actions")

    def select_action(self, state, epsilon=0.0) -> int:
        # epsilon is the chance of testing at random; otherwise we pick the best-scoring action.
        if self.random.random() < epsilon:
            return self.random.choice(self.allowed_actions)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            # forward() computes the six Q(s,a); argmax picks the action with the highest estimated value.
            q_values = self.policy_network(state_tensor)
            allowed = torch.tensor(self.allowed_actions, dtype=torch.long, device=self.device)
            return int(allowed[q_values[allowed].argmax()].item())

    def train_batch(self, replay_buffer: ReplayBuffer, batch_size: int) -> float:
        batch = replay_buffer.sample_tensors(batch_size, self.device)
        # The policy computes scores for all the examples; gather takes the score
        # of the action that was actually taken in each example.
        predicted_q = self.policy_network(batch["states"]).gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Target estimates how good the next situation can be.
            # If it crashed or landed (terminated), there is no future to add.
            # If it only ran out of time (truncated), we do use that future estimate.
            next_values = self.target_network(batch["next_states"])
            allowed = torch.tensor(self.allowed_actions, dtype=torch.long, device=self.device)
            next_q = next_values[:, allowed].max(dim=1).values
            # Learning target: current reward + gamma * best future score.
            # gamma decides how much the future matters. A physical end sets that part to zero.
            target_q = batch["rewards"] + self.gamma * next_q * (~batch["terminated"]).float()
        # We compare what the network predicted against the target: that difference is the error (loss).
        loss = self.loss_function(predicted_q, target_q)
        self.optimizer.zero_grad()  # Clears the adjustment calculations from the previous example.
        loss.backward()  # Computes how much each weight contributed to the error and which way to move it.
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.gradient_clip)  # Avoids excessive changes.
        self.optimizer.step()  # Adam nudges the weights to reduce the error next time.
        if self.training_steps % 100 == 0:
            print("Predicted Q:", predicted_q[:5].detach().tolist())
            print("Target Q:", target_q[:5].detach().tolist())
            print("Loss:", loss.item())

        self.training_steps += 1
        if self.target_update_frequency > 0 and self.training_steps % self.target_update_frequency == 0:
            self.sync_target_network()
        return float(loss.detach().item())

    def sync_target_network(self) -> None:
        """Copies the newly learned weights to the target network."""
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_checkpoint(self, path, config=None, episode=0, epsilon=0.0,
                        replay_buffer_state=None, random_states=None,
                        training_elapsed_seconds=0.0):
        """Saves networks, optimizer and counters to resume training."""
        torch.save({"policy": self.policy_network.state_dict(), "target": self.target_network.state_dict(),
                    "input_size": OBSERVATION_SIZE,
                    "optimizer": self.optimizer.state_dict(), "gamma": self.gamma,
                    "target_update_frequency": self.target_update_frequency,
                    "training_steps": self.training_steps, "episode": episode, "epsilon": epsilon,
                    "config": config or {}, "replay_buffer": replay_buffer_state,
                    "random_states": random_states,
                    "training_elapsed_seconds": training_elapsed_seconds}, path)

    def load_checkpoint(self, path):
        """Restores the saved training state and returns its metadata."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        check_observation_size(checkpoint, path)
        self.policy_network.load_state_dict(checkpoint["policy"])
        self.target_network.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gamma = checkpoint["gamma"]
        self.target_update_frequency = checkpoint["target_update_frequency"]
        self.training_steps = checkpoint["training_steps"]
        return checkpoint

    def save_inference_model(self, path):
        """Saves only the policy needed to choose actions when running the model."""
        torch.save({"policy": self.policy_network.state_dict(), "input_size": OBSERVATION_SIZE,
                    "output_size": len(ACTIONS), "allowed_actions": self.allowed_actions}, path)

    @classmethod
    def from_inference_model(cls, path, device="cpu"):
        """Builds an agent whose policy weights come from a trained model."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        check_observation_size(checkpoint, path)
        agent = cls(device=device, allowed_actions=checkpoint.get("allowed_actions"))
        agent.policy_network.load_state_dict(checkpoint["policy"])
        agent.policy_network.eval()
        return agent
