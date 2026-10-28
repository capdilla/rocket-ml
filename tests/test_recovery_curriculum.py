"""Recovery starts come from logged states, with fresh PPO rollouts."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from evaluate import run_episode
from rocket_lab.data.recovery_curriculum import load_recovery_starts, reset_recovery_scenario
from rocket_lab.ml.ppo_agent import PPOAgent
from rocket_lab.physics import (RocketEnv, XY_ALLOWED_ACTIONS, XY_ANGLE_ALLOWED_ACTIONS,
                                EARTH_GRAVITY, EARTH_ENGINE_ACCEL)
from train_ppo import load_xy_angle_policy


class RecoveryCurriculumTests(unittest.TestCase):
    def make_log(self, directory, earth=False):
        path = Path(directory) / "episode.jsonl"
        state = [-0.63, 8.73, -0.28, 0.23, -1.62, -1.67, 0.63, -8.73]
        events = [
            {"type": "start", "schema_version": 1, "mode": "xy-angle", "seed": 42,
             "model_path": "model.pt", "pad_x": 0.0, "pad_y": 0.0,
             "gravity": EARTH_GRAVITY if earth else 1.62,
             "engine_accel": EARTH_ENGINE_ACCEL if earth else 4.5,
             "initial_observation": state},
            {"type": "step", "step": 1, "controller": "manual", "state": state, "reward": 0},
            {"type": "step", "step": 2, "controller": "manual", "state": state, "reward": 0},
            {"type": "step", "step": 3, "controller": "ai", "state": state, "reward": 0},
            {"type": "step", "step": 4, "controller": "ai", "state": state, "reward": 0},
        ]
        path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        return path

    def test_handover_and_explicit_step_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_log(directory)
            starts = load_recovery_starts([path])
            self.assertEqual([start.step for start in starts], [3])
            self.assertEqual(load_recovery_starts([path], step=2)[0].step, 2)
            with self.assertRaisesRegex(ValueError, "step 99"):
                load_recovery_starts([path], step=99)

    def test_earth_logs_require_earth_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_log(directory, earth=True)
            with self.assertRaisesRegex(ValueError, "different gravity"):
                load_recovery_starts([path])
            starts = load_recovery_starts([path], gravity=EARTH_GRAVITY,
                                          engine_accel=EARTH_ENGINE_ACCEL)
            self.assertEqual(starts[0].step, 3)

    def test_recovery_curriculum_and_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            start = load_recovery_starts([self.make_log(directory)])[0]
            env = RocketEnv(seed=3, max_horizontal_distance=8, max_height=14)
            exact = reset_recovery_scenario(env, start, 1.0, augment=False)
            self.assertEqual(exact[:6], start.state)
            self.assertEqual(env.steps, 0)
            self.assertFalse(env.finished)
            early = reset_recovery_scenario(env, start, 0.0, augment=False)
            self.assertAlmostEqual(early[4], start.state[4] * 0.2)
            self.assertAlmostEqual(early[5], start.state[5] * 0.2)
            angles = [reset_recovery_scenario(env, start, 1.0)[4] for _ in range(30)]
            self.assertTrue(any(angle < 0 for angle in angles))
            self.assertTrue(any(angle > 0 for angle in angles))
            self.assertTrue(all(abs(angle) < math.pi for angle in angles))

    def test_starting_policy_must_already_be_xy_angle(self):
        with tempfile.TemporaryDirectory() as directory:
            correct = Path(directory) / "xy-angle.pt"
            wrong = Path(directory) / "xy.pt"
            PPOAgent(XY_ANGLE_ALLOWED_ACTIONS).save_model(correct, "xy-angle")
            PPOAgent(XY_ALLOWED_ACTIONS).save_model(wrong, "xy")
            self.assertEqual(load_xy_angle_policy(correct).allowed_actions, XY_ANGLE_ALLOWED_ACTIONS)
            with self.assertRaisesRegex(ValueError, "is not a PPO xy-angle policy"):
                load_xy_angle_policy(wrong)

    def test_evaluation_can_start_from_a_logged_state(self):
        with tempfile.TemporaryDirectory() as directory:
            start = load_recovery_starts([self.make_log(directory)])[0]
            env = RocketEnv(seed=3)
            observed = []

            def reset_fn(current_env):
                reset_recovery_scenario(current_env, start, 1.0, augment=False)
                current_env.state.y = 0.01
                current_env.state.vy = -3.0
                return current_env.observation()

            outcome, _, steps = run_episode(
                env, lambda state: observed.append(state) or 0,
                mode="xy-angle", reset_fn=reset_fn)
            self.assertEqual(steps, 1)
            self.assertEqual(outcome, "CRASHED")
            self.assertAlmostEqual(observed[0][4], start.state[4])


if __name__ == "__main__":
    unittest.main()
