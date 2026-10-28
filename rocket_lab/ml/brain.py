"""Definition of the RocketBrain network already used by the project.

It receives eight observations of the rocket and returns six Q values, one per
action. DQNAgent creates two instances of this same architecture: policy and
target. Running ``python3 brain.py`` shows an example prediction.
"""
import torch  # Tool for working with the numbers used by the network.
from torch import nn  # Tools for building the network layers.
from ..physics import ACTIONS, OBSERVATION_SIZE  # Actions and sensors of the simulator.

# Translates each action number to an easy-to-recognize description.
ACTION_NAMES = {
    0: "no thrust",
    1: "main engine",
    2: "rotate left",
    3: "rotate right",
    4: "main engine and rotate left",
    5: "main engine and rotate right",
}

class RocketBrain(nn.Module):  # Creates a type of neural network called RocketBrain.
    def __init__(self):  # Prepares the network when we create it.
        super().__init__()  # Activates the functions PyTorch needs for the network.

        self.network = nn.Sequential(  # Joins the layers: data passes from one to the next.
            nn.Linear(OBSERVATION_SIZE, 64),  # Receives 8 sensors and computes 64 new numbers.
            nn.ReLU(),  # Turns negative numbers into zero.
            nn.Linear(64, 64),  # Uses those 64 numbers to compute another 64.
            nn.ReLU(),  # Turns negative numbers into zero again.
            nn.Linear(64, 6)  # Computes 6 numbers, one for each rocket action.
        )

    def forward(self, state):  # Says what to do when we feed a state to the network.
        return self.network(state)  # Passes the state through the layers and returns the 6 results.


if __name__ == "__main__":
    brain = RocketBrain()
    state = torch.tensor([0.5, 4.2, -0.1, -0.8, 0.03, 0.01, -0.5, -4.2], dtype=torch.float32)
    with torch.no_grad():
        q_values = brain(state)
        action_id = torch.argmax(q_values).item()
    print("Values:", q_values)
    print("Chosen action:", ACTION_NAMES[action_id])
    print("Action number for the simulator:", action_id)
