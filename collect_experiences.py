"""Entry point: collects random episodes from the terminal."""
import argparse  # Reads options such as --episodes and --seed.
from dataclasses import asdict  # Converts the statistics to a dictionary.
import json  # Prints the statistics in a readable format.

from rocket_lab.data.episode_runner import EpisodeRunner  # Loop that runs and records episodes.
from rocket_lab.data.experience_store import save_episodes  # Function that writes episodes to JSON.
from rocket_lab.ml.agents import RandomAgent  # Policy that picks random actions.
from rocket_lab.physics import RocketEnv  # Physics simulator; does not depend on Pygame.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)  # Prepares the terminal option reader.
    parser.add_argument("--episodes", type=int, default=100)  # Number of episodes; default value 100.
    parser.add_argument("--seed", type=int, default=42)  # Seed to reproduce scenarios and actions.
    parser.add_argument("--save", help="Optional JSON path to save episodes")  # Output file.
    parser.add_argument("--save-first", type=int, default=0, help="Save the first N episodes")
    args = parser.parse_args()  # Interprets the arguments written by the user.
    if args.save_first < 0:  # A negative number makes no sense for selecting episodes.
        parser.error("--save-first cannot be negative")  # Shows an error and ends the program.

    runner = EpisodeRunner(RocketEnv(seed=args.seed), RandomAgent(seed=args.seed))  # Connects environment and agent.
    episodes, stats = runner.run(args.episodes)  # Runs the requested count and computes statistics.
    print(json.dumps(asdict(stats), indent=2))  # Shows the formatted JSON summary on screen.
    if args.save:  # Only creates the file if --save was given.
        selected = episodes[:args.save_first] if args.save_first else episodes  # Pick all or the first N.
        save_episodes(selected, args.save)  # Write the selected episodes to JSON.
        print(f"Saved {len(selected)} episodes to {args.save}")  # Confirms count and path.


if __name__ == "__main__":  # Only runs when this file is run directly.
    main()  # Starts the collection program.
