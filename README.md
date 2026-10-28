# Rocket Lab — Phase 1: craft and physics

A 2D landing simulator, without Gymnasium, Stable-Baselines3 or PyTorch. The
physics engine is implemented in pure Python, and `pygame` only renders the
craft and reads the keyboard.

## Project structure

To set up the environment and see the training, evaluation and play commands,
see [docs/comandos.md](docs/comandos.md).

```text
rocket_lab/
  physics.py       # Environment, observations and actions
  ml/              # DQN, PPO, neural network and replay buffer
  data/            # Episodes, play logs and recovery curriculum
  reporting.py     # Metric plots
data/              # Example JSON experiences
tests/             # Physics, agent and persistence tests
web/               # Browser simulator
```

The Python files at the root are executable entry points. Reusable code is
imported from `rocket_lab`, for example `from rocket_lab.physics import
RocketEnv`.

## Getting started

From this folder, preferably using Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python play.py
```

**Controls:** `↑` / `W` fires the engine; `←` / `Q` and `→` / `D` rotate; `A`
activates or disables the PPO pilot; `R` restarts the same situation, `Shift+R`
generates a new situation; `P` pauses; `Esc` closes. With the AI active, the
pilot makes the decisions and the keyboard regains control of the craft when
you press `A` again.

To play the tilted scenario and activate the PPO pilot with `A`:

```bash
python3 play.py --mode xy-angle --model runs/ppo-xy-angle-from-xy/model.pt
```

To try the same game with Earth gravity:

```bash
python3 play.py --planet earth --mode xy-angle \
  --model runs/ppo-xy-angle-recovery/model.pt
```

The Earth preset uses 9.81 m/s² of gravity and raises the engine acceleration
from 4.5 to 27.25 m/s² so it can still brake the fall. The default lunar value
is 1.62 m/s². The AI was trained with lunar physics: on Earth you can pilot it
manually, but activating the AI does not measure its ability in the
environment it was trained for. Further down there is a command to create an
Earth-specific policy from the logs of these flights.

`--model` is optional if the mode's default model is in `runs/`. There are also
`--mode y` and `--mode xy` scenarios; the initial manual mode remains `y`. The
pilot loads the weights the first time you press `A`. When handing over control
after one or more turns, `play` gives the pilot the equivalent angle between
−180° and +180°; the logs keep the original accumulated angle to reconstruct
the flight.

Every `play.py` flight is recorded automatically in `runs/play/`, in a `.jsonl`
file per episode. Each step saves the observation, the executed action, the
reward, the next state, whether the AI or the keyboard acted, and the action
the AI would have chosen when the model is loaded. On landing, crash, restart
or exit the result is recorded. Press `F` during the flight or after the
outcome to mark it as a case to review. You can change the location with
`--log-dir`. `--pad-range 3` activates a random pad between −3 and +3 m in each
scenario, just like in PPO evaluation; `R` repeats the current scenario and
`Shift+R` generates a new one.

`play_log.load_play_episode(path)` reconstructs a dictionary compatible with
`experience_store.replay_episode`. The records are ready to analyze or to
select manual-correction examples in a later phase; saving an AI failure does
not by itself indicate which action was correct.

To inspect the data **without any dependencies**, run:

```bash
python inspect_env.py
python -m unittest discover -s tests -v
```

## Environment API

```python
from rocket_lab.physics import RocketEnv

env = RocketEnv(seed=42)
observation = env.reset(randomize=True)
observation, reward, terminated, truncated, info = env.step(1)
```

- `observation`: `(x, y, vx, vy, angle, angular_velocity, dx, dy)`, where `dx = x_pad - x_craft` and `dy = y_pad - y_craft`. Positions and distances are in meters; velocities in m/s and angles in radians. DQN uses these eight values without normalizing; PPO scales the last two like positions (`3` and `10`), equally in training and inference.
- `action`: integer from 0 to 5. `0` coast; `1` main engine; `2` rotate left; `3` right; `4` engine+left; `5` engine+right.
- `reward`: `-0.01` per step, `+100` soft and stable landing, `-100` crash.
- `terminated`: landing or crash.
- `truncated`: maximum time reached.
- `info`: debugging details, **do not pass them to the network**.

The RL agent receives exclusively the 8-number observation, always in that
order. The reward is used to update its parameters, not to make the immediate
decision.

## Physics (deliberately simplified)

Axes: `x` to the right, `y` upwards. `angle = 0` is vertical; positive tilts
to the left.

```text
omega = (omega + rotation_input * ROTATION_ACCEL * dt) * ROTATION_DAMPING
angle = angle + omega * dt
ax = -sin(angle) * ENGINE_ACCEL * engine_on
ay =  cos(angle) * ENGINE_ACCEL * engine_on - GRAVITY
vx = vx + ax * dt
vy = vy + ay * dt
x = x + vx * dt
y = y + vy * dt
```

There is no air resistance, variable mass, fuel consumption, or complex
collisions. This is an educational environment, not a physically accurate
model of a real rocket. The simulation time is fixed (`dt=1/60 s`), and the
renderer also runs at 60 fps when the machine can sustain it.

## Next phase

1. Test controls, free fall and thrust; verify units and signs.
2. Separate per-episode statistics (duration, distance to target and landings).
3. Generate experiences `(state, action, reward, next_state, done)` in a loop without rendering.
4. Implement the network in PyTorch and our own DQN training.
5. Export weights for browser inference; the web simulator should replicate these equations and the observation space.

**Note:** the reward is deliberately sparse. It is useful to validate the
simulator, but you will probably need to design intermediate rewards and tune
hyperparameters to learn effectively.

# Rocket Lab

## Staged training

### Compare DQN and PPO

PPO training is separate from DQN. Both use the same eight observations, the
same physics and the same metrics format. PPO uses reward shaping and, in Y and
XY modes, an additional controlled-descent signal and a timeout penalty. The
`total_reward` column still shows the simulator's original reward.

```bash
python3 train_ppo.py --mode y --episodes 5000 --output-dir runs/ppo-y-8
python3 evaluate_ppo.py runs/ppo-y-8/model.pt --episodes 1 --render
```

Add `--render` to `train_ppo.py` to watch the flights while it learns. Rendering
advances at 60 steps per second and slows training; close the window or press
Esc to keep training without showing flights.

If you omit the model path, `evaluate_ppo.py --episodes 5` shows a selector of
PPO snapshots; `--episodes` indicates how many flights to evaluate.

The new PPO run's curve is in `runs/ppo-y-8/reward_moving_average.svg`. Compare
landings, crashes and the simulator's reward; `mean_loss` and `training_steps`
measure different operations in each algorithm.

PPO models and checkpoints are saved in their own folder and cannot be loaded
as DQN.

### Phase 2: learn lateral correction from PPO-Y

Once PPO-Y masters the vertical landing, start XY mode from that model. The
network body, the critic and the engine outputs are kept; the new outputs drive
lateral thrusters that do not rotate the craft. During the first 2,000
episodes, the craft starts at a gradually increasing lateral distance (up to
3.5 m) and with progressive horizontal speed (up to 0.5 m/s), while keeping the
same vertical challenge as phase 1. The policy initially ignores the X weights
that received no training in Y; it receives lateral-progress and descent
signals, and the flight ends if it drifts more than 8 m in X or climbs more
than 14 m:

```bash
python3 train_ppo.py --mode xy --init-model runs/ppo-y-8/model.pt \
  --output-dir runs/ppo-xy-from-y --episodes 5000
python3 evaluate_ppo.py runs/ppo-xy-from-y/model.pt --episodes 20
```

The PPO evaluator in `xy` mode tests that same lateral distribution. Use a new
folder for this phase: `runs/ppo-full-from-y` contains a policy that learned to
rotate and does not correspond to this stage. The `full` mode uses rotation
actions but does not preserve the translation thrusters learned in XY.

### Phase 3: stabilize a tilted craft from PPO-XY

The `xy-angle` mode preserves XY's six actions and adds four rotation actions.
It copies the XY policy, adds outputs for rotation and starts with small tilts.
Over 2,000 episodes it grows the initial angle up to ±0.45 rad and the angular
speed up to ±0.3 rad/s. The 100-consecutive-landings criterion is only enabled
once that curriculum completes.

```bash
python3 train_ppo.py --mode xy-angle \
  --init-model runs/ppo-xy-from-y/model.pt \
  --output-dir runs/ppo-xy-angle-from-xy --episodes 5000
python3 evaluate_ppo.py runs/ppo-xy-angle-from-xy/model.pt --episodes 100
```

To test whether it finds a pad placed at a random horizontal position in each
flight, use for example `--pad-range 3` (X between −3 and +3 m):

```bash
python3 evaluate_ppo.py runs/ppo-xy-angle-from-xy/model.pt --episodes 100 --pad-range 3
```

`--seed` replays the same platform sequence. The evaluation shows the pad's X
and the final distance `dx` in each flight. The craft still starts near the
origin; this test can also include initial distances larger than those used in
training.

### Experiment: learn XY and rotation at the same time from scratch

This experiment creates a new PPO policy, with the ten available actions and
without loading previous weights. `--no-curriculum` presents, from the first
flight, the complete distribution of positions, lateral speeds, tilts and spins
of `xy-angle`. It keeps the same reward signals as the normal mode. The output
folder is independent of the staged runs.

```bash
python3 train_ppo.py --mode xy-angle --no-curriculum --episodes 10000 \
  --success-window 0 --snapshot-frequency 0 \
  --output-dir runs/ppo-xy-angle-from-scratch
python3 evaluate_ppo.py runs/ppo-xy-angle-from-scratch/model.pt --episodes 100
python3 evaluate_ppo.py runs/ppo-xy-angle-from-xy/model.pt --episodes 100
```

Both evaluations use the same seed and the same test scenarios; compare their
landings and crashes. Repeat the training command to add episodes to the new
model from its own checkpoint.

### Phase 4: train recoveries from `play` flights

To learn to recover a tilted, spinning craft like the one in the step-225
flight, start a **new run** from the existing `xy-angle` weights:

```bash
python3 train_ppo.py --mode xy-angle \
  --init-policy runs/ppo-xy-angle-from-xy/model.pt \
  --recovery-log runs/play/20261018T230055Z-15c39cf7/episode_0001.jsonl \
  --recovery-step 225 --recovery-ratio 0.3 \
  --recovery-curriculum-episodes 3000 \
  --learning-rate 0.0001 --success-window 0 \
  --output-dir runs/ppo-xy-angle-recovery --episodes 10000
```

`--init-policy` loads the already trained policy and creates a new optimizer; it
does not resume the previous curriculum. 70% of the episodes keep the normal
`xy-angle` challenge. The remaining 30% start near the log's state, with small
variations in position, speed, tilt and rotation, and with the case mirrored
horizontally. Over 3,000 episodes the recovery tilt and rotation grow up to the
recorded severity. The **observations** are used, never the log's failed
actions as correct answers. You can repeat `--recovery-log` to mix several
flights. If you omit `--recovery-step`, every step where control passes from the
keyboard to the AI is selected.

The output folder must be different from the original model's. To resume this
phase, run the same command: `--episodes` indicates **additional** episodes.
Keep the other parameters and the same logs. The `scenario` column of
`metrics.csv` distinguishes recoveries from normal flights.

Evaluate both the recovery and the usual landings:

```bash
python3 evaluate_ppo.py runs/ppo-xy-angle-recovery/model.pt \
  --recovery-log runs/play/20261018T230055Z-15c39cf7/episode_0001.jsonl \
  --recovery-step 225 --episodes 30
python3 evaluate_ppo.py runs/ppo-xy-angle-recovery/model.pt --episodes 100
```

In the recovery evaluation, the first flight repeats the exact handover state;
the following ones test variants. The angles PPO receives are normalized the
same way in training, evaluation and `play`. To use the new model when playing:

```bash
python3 play.py --mode xy-angle --model runs/ppo-xy-angle-recovery/model.pt
```

### Phase 5: adapt the policy to Earth physics

Some terrestrial flights of the lunar model land, but a mid-descent handover
can fail because the velocities and braking time have changed. To train an
Earth policy from the already learned weights, use a new folder and an Earth
log. This example starts from the step-178 handover, at 6.7 m and −3.87 m/s:

```bash
python3 train_ppo.py --mode xy-angle --planet earth \
  --init-policy runs/ppo-xy-angle-recovery/model.pt \
  --recovery-log runs/play/20261018T232036Z-a6870812/episode_0002.jsonl \
  --recovery-step 178 --recovery-ratio 0.3 \
  --recovery-curriculum-episodes 3000 \
  --learning-rate 0.0001 --success-window 0 \
  --output-dir runs/ppo-xy-angle-earth --episodes 10000
```

Normal terrestrial flights and recoveries near the log are mixed. New models
save their planet with their weights; an old model is considered lunar. Check
both flight types and then try it in `play`:

```bash
python3 evaluate_ppo.py runs/ppo-xy-angle-earth/model.pt \
  --recovery-log runs/play/20261018T232036Z-a6870812/episode_0002.jsonl \
  --recovery-step 178 --episodes 30
python3 evaluate_ppo.py runs/ppo-xy-angle-earth/model.pt --episodes 100
python3 play.py --planet earth --mode xy-angle \
  --model runs/ppo-xy-angle-earth/model.pt
```

To continue afterwards in the same folder, repeat the training command:
`--episodes` indicates how many additional episodes to run. The program loads
its own `checkpoint.pt`. Do not use the XY checkpoint as `--init-model`: adding
actions changes the network output.

Older checkpoints with six inputs cannot be resumed with the new observations.
To train PPO from scratch, use a new folder:

```bash
python3 train_ppo.py --mode y --output-dir runs/ppo-y-8 --episodes 5000
python3 evaluate_ppo.py runs/ppo-y-8/model.pt --episodes 20
```

To start DQN from scratch in full mode:

```bash
python3 train.py --mode full --output-dir runs/dqn-full-8 --episodes 5000
python3 evaluate.py runs/dqn-full-8/model.pt --episodes 20
```

Training starts by default in vertical mode: the rocket falls from 9 m directly
above the pad and the agent can only leave the engine off or turn it on. Run:

```bash
python3 train.py --episodes 5000 --output-dir runs/dqn-y-8
```

The results are in `runs/dqn-y-8`. The program stops training and saves the
model when it achieves 100 consecutive landings; then the full challenge, with
lateral movement and random scenarios, can be started in another folder:

```bash
python3 train.py --mode full --output-dir runs/dqn-full --episodes 5000
```

Full mode continues using the six actions. Do not reuse the Y-mode checkpoint
for that training: the modes are validated on resume to avoid mixing
incompatible experiences.

A 2D lunar-landing physics simulator for learning the fundamentals of
Reinforcement Learning. The environment and the agent work without Pygame; there
is still no training or PyTorch dependency.

## Collect random experiences

From this folder, run 100 episodes (the default value) with a reproducible
seed:

```bash
.venv/bin/python collect_experiences.py --episodes 100 --seed 42
```

To run 1,000 episodes and save the first 5 as JSON:

```bash
.venv/bin/python collect_experiences.py --episodes 1000 --seed 42 --save data/experiences.json --save-first 5
```

If `--save-first` is omitted, `--save` saves all the episodes. The file includes
the transitions and the actions for inspection and reproduction.
`experience_store.load_episodes` loads the JSON, and `replay_episode` returns
to run the recorded actions starting from the saved initial state.

The printed statistics include the number of landings, crashes, timeouts, mean
reward and mean duration in simulated steps.

## Verify

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Manual interface

### Web version (no AI)

From the project root, with Bun installed:

```bash
cd web
bun run dev
```

Open <http://127.0.0.1:8765>. The web version uses Three.js included in
`web/vendor/`, so you do not need to install packages, PyTorch, Pygame or
download models. Bun only serves the files: the flight state is computed in the
browser. It keeps the physics of `rocket_lab/physics.py` with fixed 1/60 s
steps. The frontend always uses the Full Challenge, with lunar or Earth gravity
and a random pad.

Controls: **↑/W** engine, **←/Q** and **→/D** rotation, **J/L** lateral
thrusters, **R** repeat, **Shift+R** new scenario, **P** pause. The game fills
the window like a HUD: the readings appear around the view, the controls stay
pressed with several fingers, and "Settings" opens the gravity and pad options.
The flight waits for you to press "Start" or a flight control. The engine and
air sounds play while their controls are held and stop when released. The
lateral thrusters and rotation can be combined in the same step on the web. The
lateral effect is a 0.80 to 5.20 s cut of the contributed file
`25709_111413-lq.mp3`: ["20061111AirCompressorRelease.wav" by
daveincamas](https://freesound.org/people/daveincamas/sounds/25709/), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

To check the ported physics: `cd web && bun test`.

The existing manual controls remain available with:

```bash
.venv/bin/python play.py
```
