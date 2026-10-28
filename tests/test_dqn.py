"""Didactic tests for dimensions, replay, learning, checkpoint and DQN evaluation."""
import tempfile
import unittest
from pathlib import Path

import torch

from rocket_lab.ml.brain import RocketBrain
from rocket_lab.ml.dqn_agent import DQNAgent
from rocket_lab.ml.replay_buffer import ReplayBuffer


class DQNTests(unittest.TestCase):
    def test_network_dimensions(self):
        # The network accepts an isolated state and a batch; in both cases there are six Q-values.
        self.assertEqual(tuple(RocketBrain()(torch.zeros(8)).shape), (6,))
        self.assertEqual(tuple(RocketBrain()(torch.zeros(4, 8)).shape), (4, 6))

    def test_replay_round_trip_and_tensor_batch(self):
        # Checks stored fields and the minibatch tensor shapes.
        buffer = ReplayBuffer(2, seed=1)
        buffer.add((1, 2, 3, 4, 5, 6, -1, -2), 2, 1.5, (2, 3, 4, 5, 6, 7, -2, -3), False, True)
        exp = buffer.sample(1)[0]
        self.assertEqual(exp.action, 2)
        self.assertTrue(exp.truncated)
        tensors = buffer.sample_tensors(1)
        self.assertEqual(tuple(tensors["states"].shape), (1, 8))

    def test_update_changes_weights_and_returns_loss(self):
        # A valid update must produce a numerical loss and modify at least one weight.
        agent = DQNAgent(seed=4, target_update_frequency=100)
        buffer = ReplayBuffer(8, seed=2)
        for i in range(4):
            buffer.add((i, 1, 0, -1, 0, 0, -i, -1), i % 6, 1,
                       (i + 1, 1, 0, -1, 0, 0, -i - 1, -1), False, False)
        before = [p.detach().clone() for p in agent.policy_network.parameters()]
        loss = agent.train_batch(buffer, 4)
        self.assertGreaterEqual(loss, 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.policy_network.parameters())))

    def test_target_sync(self):
        # After syncing, every policy and target parameter must match.
        agent = DQNAgent(seed=5)
        with torch.no_grad():
            next(agent.policy_network.parameters()).add_(1)
        agent.sync_target_network()
        for a, b in zip(agent.policy_network.parameters(), agent.target_network.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_checkpoint_and_inference_preserve_predictions(self):
        # Both saved formats must reconstruct exactly the same prediction.
        agent = DQNAgent(seed=6)
        state = torch.zeros(8)
        with torch.no_grad():
            expected = agent.policy_network(state).clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, model = Path(directory) / "checkpoint.pt", Path(directory) / "model.pt"
            agent.save_checkpoint(checkpoint)
            agent.save_inference_model(model)
            loaded = DQNAgent.from_inference_model(model)
            full = DQNAgent(seed=7)
            full.load_checkpoint(checkpoint)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, loaded.policy_network(state)))
            self.assertTrue(torch.equal(expected, full.policy_network(state)))

    def test_evaluation_action_selection_does_not_change_weights(self):
        # Choosing greedy actions reads the model, but does not run an update.
        agent = DQNAgent(seed=8)
        before = [p.detach().clone() for p in agent.policy_network.parameters()]
        for _ in range(10):
            agent.select_action((0, 1, 0, -1, 0, 0, 0, -1), epsilon=0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, agent.policy_network.parameters())))

    def test_old_model_rejected(self):
        old = RocketBrain().state_dict()
        old["network.0.weight"] = old["network.0.weight"][:, :6].clone()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "old.pt"
            torch.save({"policy": old, "input_size": 6, "allowed_actions": (0, 1)}, source)
            with self.assertRaisesRegex(ValueError, "6 inputs"):
                DQNAgent.from_inference_model(source)


if __name__ == "__main__":
    unittest.main()
