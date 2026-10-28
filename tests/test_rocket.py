import unittest
import math

from rocket_lab.physics import (DT, RocketEnv, MAX_STEPS, GRAVITY, ENGINE_ACCEL,
                                EARTH_GRAVITY, EARTH_ENGINE_ACCEL, upright_angle)


class RocketPhysicsTests(unittest.TestCase):
    def test_reset_state_format(self):
        env = RocketEnv(seed=0)
        self.assertEqual(len(env.reset()), 8)

    def test_invalid_configuration_fails_before_simulation_starts(self):
        for name, value in (
            ("gravity", float("nan")),
            ("engine_accel", float("inf")),
            ("pad_x", float("nan")),
            ("pad_y", float("inf")),
            ("max_horizontal_distance", -1),
            ("max_height", -1),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    RocketEnv(**{name: value})

    def test_invalid_lateral_scenarios_fail_before_they_change_the_environment(self):
        env = RocketEnv(seed=0)
        initial = env.observation()
        for arguments in ((0.24, 0), (1, -0.1), (float("nan"), 0)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    env.reset_lateral(*arguments)
                self.assertEqual(env.observation(), initial)
        for arguments in ((1, 0, -0.1, 0), (1, 0, 0, float("inf"))):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    env.reset_lateral_attitude(*arguments)
                self.assertEqual(env.observation(), initial)

    def test_relative_pad_sensors_use_current_positions(self):
        env = RocketEnv(seed=1, pad_x=4.5, pad_y=1.25)
        initial = env.reset()
        self.assertEqual(initial[6:], (env.pad_x - initial[0], env.pad_y - initial[1]))
        after, *_ = env.step(0)
        self.assertEqual(after[6:], (env.pad_x - after[0], env.pad_y - after[1]))
        vertical = env.reset_y_only()
        self.assertEqual(vertical[6:], (env.pad_x - vertical[0], env.pad_y - vertical[1]))

    def test_without_engine_rocket_accelerates_downwards(self):
        env = RocketEnv(seed=0)
        before = env.state.vy
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertLess(observation[3], before)
        self.assertFalse(terminated)

    def test_main_engine_accelerates_upwards(self):
        env = RocketEnv(seed=0)
        before = env.state.vy
        observation, *_ = env.step(1)
        self.assertGreater(observation[3], before)

    def test_earth_preset_gravity_and_engine_are_flyable(self):
        moon = RocketEnv(gravity=GRAVITY, engine_accel=ENGINE_ACCEL)
        earth = RocketEnv(gravity=EARTH_GRAVITY, engine_accel=EARTH_ENGINE_ACCEL)
        moon.reset_y_only()
        earth.reset_y_only()
        moon_fall, *_ = moon.step(0)
        earth_fall, *_ = earth.step(0)
        self.assertLess(earth_fall[3], moon_fall[3])
        earth.reset_y_only()
        earth_burn, *_ = earth.step(1)
        self.assertGreater(earth_burn[3], -0.4)
        self.assertGreater(EARTH_ENGINE_ACCEL, EARTH_GRAVITY)

    def test_turning_changes_angular_velocity(self):
        env = RocketEnv(seed=0)
        observation, *_ = env.step(2)
        self.assertGreater(observation[5], 0)

    def test_lateral_attitude_reset_sets_bounded_tilt_and_spin(self):
        env = RocketEnv(seed=0)
        for _ in range(20):
            observation = env.reset_lateral_attitude(3.5, 0.5, 0.45, 0.3)
            self.assertLessEqual(abs(observation[4]), 0.45)
            self.assertLessEqual(abs(observation[5]), 0.3)
            self.assertLessEqual(abs(observation[0]), 3.5)
            self.assertFalse(env.finished)

    def test_crash_ends_episode(self):
        env = RocketEnv(seed=0)
        env.state.y = 0.001
        env.state.vy = -3
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["outcome"], "CRASHED")
        self.assertEqual(reward, -100)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_safe_landing(self):
        env = RocketEnv(seed=0)
        env.state.x = 0
        env.state.y = 0.001
        env.state.vx = 0
        env.state.vy = -0.5
        obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["outcome"], "LANDED")
        self.assertEqual(reward, 100)

    def test_full_rotation_is_upright_for_landing(self):
        env = RocketEnv(seed=0)
        env.state.x = 0
        env.state.y = 0.001
        env.state.vx = 0
        env.state.vy = -0.5
        env.state.angle = -2 * math.pi
        _, reward, terminated, _, info = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["outcome"], "LANDED")
        self.assertEqual(reward, 100)
        self.assertAlmostEqual(upright_angle(-2 * math.pi - 0.4), -0.4)


if __name__ == "__main__":
    unittest.main()
