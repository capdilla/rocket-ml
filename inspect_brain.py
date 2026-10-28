"""Inspects RocketBrain's Q values for synthetic states.

Example:
    python inspect_brain.py model.pt
    python inspect_brain.py runs/dqn/model.pt --height 8.5

It only loads the model and computes predictions; it does not change its
weights or train it. The test states may not correspond to usual training
situations. In particular, vy=-2 and vy=-3 m/s exceed the initial training
range (-0.8 to 0.1 m/s), although they are possible during the simulation.
"""
import argparse

import torch

from rocket_lab.ml.brain import ACTION_NAMES, RocketBrain
from rocket_lab.ml.dqn_agent import check_observation_size
from rocket_lab.physics import RocketEnv, RocketState


VERTICAL_SPEEDS = (-0.1, -0.5, -1.0, -2.0, -3.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", metavar="MODEL",
                        help="One or more inference models (for example model.pt)")
    parser.add_argument("--height", type=float, default=9.0,
                        help="Common altitude in meters (initial training range: 7-10; default: 9)")
    args = parser.parse_args()
    if not 7.0 <= args.height <= 10.0:
        parser.error("--height must be between 7 and 10 m to match the initial training range")

    # x, vx, angle and angular speed stay fixed and within the ranges
    # of the training scenario generator.
    env = RocketEnv()
    states = []
    for vy in VERTICAL_SPEEDS:
        env.state = RocketState(0.0, args.height, 0.0, vy, 0.0, 0.0)
        states.append(torch.tensor(env.observation(), dtype=torch.float32))

    for checkpoint_path in args.checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
            parser.error(f"{checkpoint_path}: expected an inference model with the 'policy' key")
        try:
            check_observation_size(checkpoint, checkpoint_path)
        except ValueError as error:
            parser.error(str(error))

        brain = RocketBrain()
        brain.load_state_dict(checkpoint["policy"])
        brain.eval()

        print(f"\nCheckpoint: {checkpoint_path}")
        print("Observation: (x, y, vx, vy, angle, angular_velocity, dx, dy)")
        print(f"Altitude: {args.height:.1f} m; other values fixed: x=vx=angle=angular_velocity=0")
        with torch.inference_mode():
            for state in states:
                q_values = brain(state)
                action_id = int(torch.argmax(q_values).item())
                values = ", ".join(f"{value:.6f}" for value in q_values.tolist())
                print(f"vy={state[3].item():>4.1f} m/s | Q[0..5]=[{values}] | argmax={action_id} ({ACTION_NAMES[action_id]})")

    print("\nNote: these synthetic states may not correspond to usual training situations.")


if __name__ == "__main__":
    main()
