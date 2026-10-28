# Rocket Lab commands

Run the commands from the project root folder.

## Set up the environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Recommended training: staged PPO

PPO learns the vertical landing and then adds lateral and rotation control. Each stage writes its results to a different folder.

```bash
# 1. Learn to brake and land vertically
python train_ppo.py --mode y --episodes 5000 --output-dir runs/ppo-y

# 2. Learn to reach the pad from a lateral position
python train_ppo.py --mode xy \
  --init-model runs/ppo-y/model.pt \
  --output-dir runs/ppo-xy --episodes 5000

# 3. Learn to straighten the craft during flight
python train_ppo.py --mode xy-angle \
  --init-model runs/ppo-xy/model.pt \
  --output-dir runs/ppo-xy-angle --episodes 5000
```

Test each model before moving to the next stage:

```bash
python evaluate_ppo.py runs/ppo-y/model.pt --episodes 20
python evaluate_ppo.py runs/ppo-xy/model.pt --episodes 20
python evaluate_ppo.py runs/ppo-xy-angle/model.pt --episodes 100
```

To show each flight while training, add `--render`. Rendering limits the training pace.

Training saves `model.pt` for playing or evaluating, `checkpoint.pt` for resuming, `metrics.csv` and `reward_moving_average.svg`. Running the same command again with the same folder and configuration continues from the checkpoint. To start from scratch, use a new output folder. `--episodes` indicates the additional episodes for that run.

## Train with DQN

DQN has a vertical mode (`y`) and one with lateral movement (`full`). Train each mode in its own folder:

```bash
python train.py --mode y --episodes 5000 --output-dir runs/dqn-y
python evaluate.py runs/dqn-y/model.pt --episodes 20

python train.py --mode full --episodes 5000 --output-dir runs/dqn-full
python evaluate.py runs/dqn-full/model.pt --episodes 20
```

DQN resumes from `checkpoint.pt` if the folder already contains a compatible run. It can stop before the requested number when it reaches 100 consecutive landings. Use `--success-window 0` to disable that cutoff.

To enable the progress rewards during training:

```bash
python train.py --mode full --reward-shaping \
  --episodes 5000 --output-dir runs/dqn-full-shaped
```

## Play and test the AI

Manual play:

```bash
python play.py
```

Playing with a PPO model:

```bash
python play.py --mode y --model runs/ppo-y/model.pt
python play.py --mode xy --model runs/ppo-xy/model.pt
python play.py --mode xy-angle --model runs/ppo-xy-angle/model.pt
```

Press `A` to hand over or take back control of the craft. Flights are saved under `runs/play/`. To use Earth gravity or move the pad to a random position:

```bash
python play.py --planet earth --mode xy-angle \
  --model runs/ppo-xy-angle-earth/model.pt
python play.py --mode xy-angle --pad-range 3
```

PPO evaluation with random pad positions:

```bash
python evaluate_ppo.py runs/ppo-xy-angle/model.pt \
  --episodes 100 --pad-range 3
```

DQN evaluation with visualization:

```bash
python evaluate.py runs/dqn-full/model.pt --episodes 5 --render
```

If the model path is omitted in `evaluate.py` or `evaluate_ppo.py`, a selector of available models appears.

## Train recoveries from a flight

Find a flight saved by `play.py`:

```bash
find runs/play -name 'episode_*.jsonl'
```

Use the path of an `xy-angle` log to train recoveries from the moments when control passed from the keyboard to the AI:

```bash
python train_ppo.py --mode xy-angle \
  --init-policy runs/ppo-xy-angle/model.pt \
  --recovery-log runs/play/RUN_ID/episode_0001.jsonl \
  --recovery-ratio 0.3 \
  --output-dir runs/ppo-recovery --episodes 10000
```

Replace `RUN_ID` with the real folder name. You can add `--recovery-step 225` to choose a specific step. Recoveries require an `xy-angle` model; to combine several flights, repeat `--recovery-log`.

## Other tools

```bash
# Show a sequence of physics observations
python inspect_env.py

# Inspect the Q values of one or more DQN models
python inspect_brain.py runs/dqn-full/model.pt

# Regenerate a plot from the metrics
python -m rocket_lab.reporting runs/ppo-xy-angle/metrics.csv

# Collect episodes with a random policy
python collect_experiences.py --episodes 100 --seed 42
python collect_experiences.py --episodes 1000 --seed 42 \
  --save data/experiences.json --save-first 5

# Run the Python suite
python -m unittest discover -s tests -v

# See the full options of each program
python train.py --help
python train_ppo.py --help
python evaluate.py --help
python evaluate_ppo.py --help
python play.py --help
python collect_experiences.py --help
python inspect_brain.py --help
```

## Web simulator

With Bun installed:

```bash
cd web
bun run dev
```

Open `http://127.0.0.1:8765`. To check the web simulator physics, run `bun test` from `web/`.
