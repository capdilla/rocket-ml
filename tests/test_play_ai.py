"""Angles sent to the interactive pilot describe physical orientation."""
import math
import unittest
from collections import defaultdict

import pygame

from play import choose_action, policy_observation


class PlayAITests(unittest.TestCase):
    def test_a_is_reserved_for_ai_and_q_turns_left(self):
        keys = defaultdict(bool)
        keys[pygame.K_a] = True
        self.assertEqual(choose_action(keys), 0)
        keys[pygame.K_q] = True
        self.assertEqual(choose_action(keys), 2)
        keys[pygame.K_q] = False
        keys[pygame.K_LEFT] = True
        self.assertEqual(choose_action(keys), 2)

    def test_full_turns_do_not_change_the_pilots_observation(self):
        state = (-1.0, 9.7, -0.78, -1.6, math.radians(-403.4), -1.39, 1.0, -9.7)
        observation = policy_observation(state)
        self.assertAlmostEqual(math.degrees(observation[4]), -43.4)
        self.assertEqual(observation[:4] + observation[5:], state[:4] + state[5:])
        self.assertEqual(state[4], math.radians(-403.4))


if __name__ == "__main__":
    unittest.main()
