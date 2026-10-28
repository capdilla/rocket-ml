"""See the eight raw observations and rewards, without any UI or ML libraries."""
from rocket_lab.physics import OBSERVATION_NAMES, RocketEnv

env = RocketEnv(seed=42)
observation = env.reset()
print("initial observation:", observation)
print("fields:", OBSERVATION_NAMES, "\n")

for step in range(120):
    action = 1 if step >= 40 else 0  # Coast, then turn on main engine.
    observation, reward, terminated, truncated, info = env.step(action)
    if step % 20 == 0 or terminated or truncated:
        print(f"step={step:03d} action={action} state={tuple(round(x, 3) for x in observation)} reward={reward:+.2f}")
    if terminated or truncated:
        print("outcome:", info["outcome"])
        break
