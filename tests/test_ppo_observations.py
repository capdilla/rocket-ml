"""PPO consumes the same eight sensor values during training and inference."""
import tempfile
import unittest
from pathlib import Path

import torch

from rocket_lab.ml.ppo_agent import PPOAgent
from rocket_lab.physics import RocketEnv, XY_ALLOWED_ACTIONS, XY_ANGLE_ALLOWED_ACTIONS


class PPOObservationTests(unittest.TestCase):
    def test_model_round_trip_and_old_model_rejection(self):
        state = RocketEnv().reset()
        agent = PPOAgent((0, 1))
        with torch.no_grad():
            expected = agent(torch.tensor(state, dtype=torch.float32))
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / "current.pt"
            old = Path(directory) / "old.pt"
            agent.save_model(current, "y")
            loaded, mode = PPOAgent.from_model(current)
            self.assertEqual(mode, "y")
            with torch.no_grad():
                current = loaded(torch.tensor(state, dtype=torch.float32))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, current)))
            old_weights = agent.state_dict()
            old_weights["body.0.weight"] = old_weights["body.0.weight"][:, :6].clone()
            old_weights["observation_scale"] = old_weights["observation_scale"][:6].clone()
            torch.save({"algorithm": "ppo", "policy": old_weights, "allowed_actions": (0, 1), "mode": "y"}, old)
            with self.assertRaisesRegex(ValueError, "6 PPO inputs"):
                PPOAgent.from_model(old)

    def test_xy_transfer_preserves_existing_action_scores(self):
        source = PPOAgent(XY_ALLOWED_ACTIONS)
        state = torch.tensor(RocketEnv(seed=3).reset_lateral(2.0, 0.25), dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "xy.pt"
            source.save_model(path, "xy")
            target = PPOAgent.from_xy_model(path)
            with torch.no_grad():
                old_logits, old_value = source(state)
                new_logits, new_value = target(state)
            self.assertEqual(target.allowed_actions, XY_ANGLE_ALLOWED_ACTIONS)
            self.assertTrue(torch.allclose(old_logits, new_logits[:len(XY_ALLOWED_ACTIONS)]))
            self.assertTrue(torch.allclose(old_value, new_value))
            self.assertTrue(torch.equal(target.body[0].weight[:, (4, 5)],
                                        torch.zeros_like(target.body[0].weight[:, (4, 5)])))

    def test_xy_transfer_rejects_other_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "y.pt"
            PPOAgent((0, 1)).save_model(path, "y")
            with self.assertRaisesRegex(ValueError, "mode XY"):
                PPOAgent.from_xy_model(path)

    def test_model_records_training_planet(self):
        with tempfile.TemporaryDirectory() as directory:
            earth = Path(directory) / "earth.pt"
            moon = Path(directory) / "moon.pt"
            PPOAgent(XY_ANGLE_ALLOWED_ACTIONS).save_model(earth, "xy-angle", planet="earth")
            PPOAgent(XY_ANGLE_ALLOWED_ACTIONS).save_model(moon, "xy-angle")
            self.assertEqual(PPOAgent.from_model(earth)[0].planet, "earth")
            self.assertEqual(PPOAgent.from_model(moon)[0].planet, "moon")


if __name__ == "__main__":
    unittest.main()
