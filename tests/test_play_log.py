"""Interactive flight logs retain replayable transitions and review marks."""
import tempfile
import unittest

from rocket_lab.data.experience_store import replay_episode
from rocket_lab.data.play_log import PlayLogger, load_play_episode
from rocket_lab.physics import RocketEnv, EARTH_GRAVITY, EARTH_ENGINE_ACCEL


class PlayLogTests(unittest.TestCase):
    def test_complete_flight_is_replayable_and_keeps_control_source(self):
        env = RocketEnv(seed=3, pad_x=2.0)
        env.reset_lateral(1.0)
        with tempfile.TemporaryDirectory() as directory:
            logger = PlayLogger(directory)
            self.assertRegex(logger.directory.name, r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
            logger.start(env, "xy", 3, "model.pt")
            while not env.finished:
                state = env.observation()
                action = 0
                next_state, reward, terminated, truncated, info = env.step(action)
                logger.record(state, action, reward, next_state, terminated, truncated, info,
                              "manual", 1, "xy")
            logger.finish(env, "finished", "xy")
            logger.mark_for_review()  # The player can mark a failure after seeing the result.
            saved = load_play_episode(logger.path)
            replayed = replay_episode(RocketEnv(seed=0, pad_x=saved["pad_x"]), saved)
            self.assertEqual(saved["duration_steps"], env.steps)
            self.assertEqual(saved["outcome"], env.last_outcome)
            self.assertTrue(saved["needs_review"])
            self.assertEqual(saved["transitions"][0]["controller"], "manual")
            self.assertEqual(saved["transitions"][0]["ai_action"], 1)
            self.assertEqual(replayed[-1]["next_state"], saved["transitions"][-1]["next_state"])

    def test_interrupted_flight_keeps_steps(self):
        env = RocketEnv(seed=4)
        with tempfile.TemporaryDirectory() as directory:
            logger = PlayLogger(directory)
            logger.start(env, "y", 4, None)
            state = env.observation()
            next_state, reward, terminated, truncated, info = env.step(0)
            logger.record(state, 0, reward, next_state, terminated, truncated, info,
                          "manual", None, None)
            logger.finish(env, "exit")
            saved = load_play_episode(logger.path)
            self.assertEqual(saved["outcome"], "INTERRUPTED")
            self.assertEqual(saved["end_reason"], "exit")
            self.assertEqual(len(saved["transitions"]), 1)

    def test_earth_flight_replays_with_logged_physics(self):
        env = RocketEnv(seed=4, gravity=EARTH_GRAVITY, engine_accel=EARTH_ENGINE_ACCEL)
        env.reset_y_only()
        with tempfile.TemporaryDirectory() as directory:
            logger = PlayLogger(directory)
            logger.start(env, "y", 4, None)
            state = env.observation()
            next_state, reward, terminated, truncated, info = env.step(1)
            logger.record(state, 1, reward, next_state, terminated, truncated, info,
                          "manual", None, None)
            logger.finish(env, "exit")
            saved = load_play_episode(logger.path)
            self.assertEqual(saved["gravity"], EARTH_GRAVITY)
            self.assertEqual(saved["engine_accel"], EARTH_ENGINE_ACCEL)
            replayed = replay_episode(RocketEnv(seed=0), saved)
            self.assertEqual(replayed[0]["next_state"], list(next_state))


if __name__ == "__main__":
    unittest.main()
