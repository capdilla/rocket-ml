"""Didactic tests that verify the agent and data collection."""
import tempfile  # Creates a temporary folder so no test JSON is left in the project.
import unittest  # Standard tools for defining and running tests.
from pathlib import Path  # Builds file paths portably.

from rocket_lab.data.episode_runner import EpisodeRunner  # The episode runner under test.
from rocket_lab.data.experience_store import load_episodes, replay_episode, save_episodes  # Save, load and replay.
from rocket_lab.ml.agents import RandomAgent  # The agent being verified.
from rocket_lab.physics import RocketEnv  # The simulator used by the tests.


class ExperienceCollectionTests(unittest.TestCase):
    def test_random_agent_selects_only_available_actions(self):
        agent = RandomAgent(seed=4)  # Creates a reproducible agent.
        actions = [agent.select_action((0, 0, 0, 0, 0, 0)) for _ in range(100)]  # Asks for 100 actions.
        self.assertTrue(all(0 <= action <= 5 for action in actions))  # All must honor the contract.

    def test_episode_transitions_chain_states_and_finish(self):
        runner = EpisodeRunner(RocketEnv(seed=7), RandomAgent(seed=7))  # Prepares environment and agent.
        episode = runner.run_episode()  # Runs until landing, crash or timeout.
        self.assertGreater(len(episode.transitions), 0)  # A flight must have at least one step.
        for previous, current in zip(episode.transitions, episode.transitions[1:]):  # Compares neighboring steps.
            self.assertEqual(previous.next_state, current.state)  # The next state becomes the current one.
        final = episode.transitions[-1]  # Gets the last saved step.
        self.assertTrue(final.terminated or final.truncated)  # The episode must have an ending condition.
        self.assertEqual(len(final.state), 8)
        self.assertEqual(len(final.next_state), 8)
        self.assertEqual(episode.duration_steps, len(episode.transitions))  # Duration matches the steps.

    def test_seed_reproduces_runs_and_statistics(self):
        def collect():  # Repeats a batch from the same initial seeds.
            return EpisodeRunner(RocketEnv(seed=21), RandomAgent(seed=21)).run(5)
        first, first_stats = collect()  # First run.
        second, second_stats = collect()  # Second, identical run.
        self.assertEqual([e.to_dict() for e in first], [e.to_dict() for e in second])  # Same steps.
        self.assertEqual(first_stats, second_stats)  # Same statistics.

    def test_json_round_trip_for_selected_episode(self):
        episode = EpisodeRunner(RocketEnv(seed=2), RandomAgent(seed=2)).run_episode()  # Produces data.
        with tempfile.TemporaryDirectory() as directory:  # Folder removed when the block exits.
            path = Path(directory) / "episodes.json"  # Path for the test file.
            save_episodes([episode], path)  # Write the episode.
            loaded = load_episodes(path)  # Read the JSON back.
        self.assertEqual(len(loaded), 1)  # There must be one episode.
        self.assertEqual(loaded[0]["outcome"], episode.outcome)  # It must preserve the outcome.
        self.assertEqual(loaded[0]["transitions"][0]["state"], list(episode.transitions[0].state))  # And the state.

    def test_saved_actions_replay_to_same_result(self):
        env = RocketEnv(seed=19)  # Original environment of the episode.
        episode = EpisodeRunner(env, RandomAgent(seed=19)).run_episode()  # Saves a sequence of actions.
        replay_env = RocketEnv(seed=0)  # Another environment; the replay loads the saved initial state.
        replayed = replay_episode(replay_env, episode.to_dict())  # Repeats the recorded actions.
        self.assertEqual(len(replayed), episode.duration_steps)  # Must repeat the same number of steps.
        self.assertEqual(replayed[-1]["outcome"], episode.outcome)  # Must end with the same result.
        self.assertEqual(replayed[-1]["next_state"], list(episode.transitions[-1].next_state))  # Same final state.


if __name__ == "__main__":  # Allows running this file directly.
    unittest.main()  # Runs every method whose name starts with test_.
