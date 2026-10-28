import unittest

from train import shaped_reward


class PotentialRewardShapingTests(unittest.TestCase):
    def shaped(self, before, after):
        return shaped_reward(before, after, 0.0, False, True, gamma=0.99, scale=1.0)

    def test_controlled_descent_is_rewarded(self):
        before = (0, 8, 0, -1.5, 0, 0)
        after = (0, 7.8, 0, -1.5, 0, 0)
        self.assertGreater(self.shaped(before, after), 0)

    def test_uncontrolled_descent_does_not_get_height_progress_reward(self):
        before = (0, 8, 0, -1.5, 0, 0)
        after = (0, 7.8, 0, -5.0, 0, 0)
        self.assertLess(self.shaped(before, after), 0)

    def test_horizontal_position_and_speed_progress(self):
        centered = self.shaped((2, 5, 0, -1, 0, 0), (1, 5, 0, -1, 0, 0))
        slowed = self.shaped((0, 5, 1.5, -1, 0, 0), (0, 5, 0.5, -1, 0, 0))
        self.assertGreater(centered, 0)
        self.assertGreater(slowed, 0)

    def test_attitude_progress(self):
        stable = self.shaped((0, 5, 0, -1, 0.2, 0.1), (0, 5, 0, -1, 0.05, 0.02))
        self.assertGreater(stable, 0)

    def test_disabled_returns_original_reward(self):
        self.assertEqual(shaped_reward((0, 5, 0, -1, 0, 0), (0, 4, 0, -1, 0, 0), 2.5, False), 2.5)

    def test_terminal_and_timeout_do_not_use_next_state_potential(self):
        state = (0, 5, 0, -1, 0, 0)
        terminal_a = shaped_reward(state, (0, 0, 0, 0, 0, 0), 100, True, True, scale=1)
        terminal_b = shaped_reward(state, (100, 0, 50, 0, 2, 2), 100, True, True, scale=1)
        timeout_a = shaped_reward(state, (0, 5, 0, -1, 0, 0), -15, False, True, truncated=True, scale=1)
        timeout_b = shaped_reward(state, (100, 0, 50, 0, 2, 2), -15, False, True, truncated=True, scale=1)
        self.assertEqual(terminal_a, terminal_b)
        self.assertEqual(timeout_a, timeout_b)


if __name__ == "__main__":
    unittest.main()
