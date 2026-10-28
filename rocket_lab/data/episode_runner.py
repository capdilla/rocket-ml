"""Runs complete episodes and turns each step into an experience."""
from __future__ import annotations  # Allows type annotations that are defined later.

from dataclasses import dataclass  # Creates a simple class for grouping statistics.

from ..ml.agents import RandomAgent  # The agent that decides which action to take.
from .experience_store import Episode, Transition  # Formats for saving the data.
from ..physics import RocketEnv  # The physics simulator that receives the actions.


@dataclass
class RunStatistics:
    """Numerical summary of a batch of episodes."""
    episodes: int  # How many episodes ran.
    successful_landings: int  # Episodes whose result was LANDED.
    accidents: int  # Episodes whose result was CRASHED.
    timeouts: int  # Episodes that hit the step limit.
    mean_reward: float  # Average of the sum of rewards per episode.
    mean_duration_steps: float  # Average of physical steps per episode.


class EpisodeRunner:
    def __init__(self, env: RocketEnv, agent: RandomAgent, randomize: bool = True):
        self.env = env  # Stores the environment that will advance the physics.
        self.agent = agent  # Stores who selects the actions.
        self.randomize = randomize  # Decides whether reset() uses varied initial positions.

    def run_episode(self) -> Episode:
        state = tuple(self.env.reset(randomize=self.randomize))  # Initial state (8 numbers).
        transitions: list[Transition] = []  # Each step of the flight is stored here.
        total_reward = 0.0  # Accumulates rewards for the episode summary.
        terminated = truncated = False  # The loop continues until one of these becomes True.
        info = {"outcome": None}  # step() will report landing, crash or timeout here.
        while not (terminated or truncated):
            action = self.agent.select_action(state)  # The agent decides using the current state.
            next_state, reward, terminated, truncated, info = self.env.step(action)  # Simulates one step.
            next_state = tuple(next_state)  # Keeps the observation as an eight-value tuple.
            transitions.append(Transition(
                state=state,  # Observation before applying the action.
                action=action,  # Action chosen by the agent.
                reward=reward,  # Reward or cost returned by the simulator.
                next_state=next_state,  # Observation after the physics step.
                terminated=terminated,  # True if the episode ended by landing/crash.
                truncated=truncated,  # True if it ended by the time/step limit.
            ))
            total_reward += reward  # Adds this step to the episode total.
            state = next_state  # The next step starts where the previous one ended.
        return Episode(transitions, str(info["outcome"]), total_reward)  # Returns the complete flight.

    def run(self, episode_count: int) -> tuple[list[Episode], RunStatistics]:
        if episode_count < 1:
            raise ValueError("episode_count must be at least 1")  # Avoids averages without data.
        episodes = [self.run_episode() for _ in range(episode_count)]  # Runs N complete flights.
        count = len(episodes)  # Actual number of results to summarize.
        stats = RunStatistics(
            episodes=count,
            successful_landings=sum(ep.outcome == "LANDED" for ep in episodes),  # Counts safe landings.
            accidents=sum(ep.outcome == "CRASHED" for ep in episodes),  # Counts crashes.
            timeouts=sum(ep.outcome == "TIMEOUT" for ep in episodes),  # Counts truncated episodes.
            mean_reward=sum(ep.total_reward for ep in episodes) / count,  # Mean total reward.
            mean_duration_steps=sum(ep.duration_steps for ep in episodes) / count,  # Mean steps.
        )
        return episodes, stats  # Returns both the detailed data and the summary.
