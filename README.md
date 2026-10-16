# WWM-NAV

Repository: git@github.com:vicent-hang/WWM-NAV.git

WWM-NAV is a compact open-source implementation of the research-one wireless world-model navigation path. It studies risk-aware UAV navigation on radio maps with a physics-conditioned RSSM world model and a Dreamer-style discrete Actor-Critic.

The package intentionally contains only the navigation layer:

- four grid actions: north, east, south, west;
- radio-map outage feedback at visited UAV positions;
- RSSM heads for observation, reward, outage, high-outage, and done prediction;
- Actor-only execution with RSSM posterior feedback;
- closed-loop RSSM imagination for Actor-Critic training;
- optional SNARM-style distance-to-goal navigation prior.

It does not include the communication/uplink controller layer, COMUDO power queues, or continual-learning COLD-RSSM protocols.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development checks:

```bash
pip install -e .[dev]
```

## Reproduce WWM-NAV

Generate the registered physical radio environment, then verify its fields and
formal experiment configuration:

```bash
python scripts/generate_radio_env.py \
  --output data/wwm_nav/radioenvir_wwm_nav.npz \
  --seed 7 \
  --fast-fading-samples 1000 \
  --los-samples 100
python scripts/verify_wwm_nav_setup.py --config configs/wwm_nav_default.yaml
```

Run the registered 1,500-episode, 300-step experiment:

```bash
python scripts/train_wwm_nav.py --config configs/wwm_nav_default.yaml
```

The YAML file is the standalone form of the original WWM-NAV command contract.
Strict validation rejects accidental changes to effective research parameters.
Training writes a resumable state every 20 episodes and saves the final model
checkpoint after episode 1,500. It does not run evaluation during training.

Resume an interrupted run without changing its output directory or config:

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --resume outputs/wwm_nav_seed1/training_state.pt
```

Evaluate the final checkpoint after training:

```bash
python scripts/evaluate_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/wwm_nav_seed1/checkpoint.pt
```

This writes `outputs/wwm_nav_seed1/eval_rollout.npz` and
`eval_summary.json`.

Evaluate the final checkpoint on the registered 100 shared random-start trials:

```bash
python scripts/evaluate_wwm_nav_trials.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/wwm_nav_seed1/checkpoint.pt \
  --save-dir outputs/wwm_nav_seed1/shared_trials \
  --num-trials 100 \
  --seed 2026 \
  --min-initial-distance 800 \
  --plot-all
```

This writes the frozen `trial_pairs.csv`, per-trial `trial_metrics.csv` and
`trial_metrics.npz`, one rollout archive per trial, and `summary.json`/`summary.md`.

Plot the evaluation trajectory:

```bash
python scripts/plot_trajectory.py \
  --radiomap data/wwm_nav/radioenvir_wwm_nav.npz \
  --rollout outputs/wwm_nav_seed1/eval_rollout.npz \
  --output outputs/wwm_nav_seed1/eval_trajectory.png
```

See [run_commands.md](run_commands.md) for the complete setup, map generation,
training, resume, evaluation, and plotting commands.

Console scripts are also registered after installation: `wwm-nav-train`,
`wwm-nav-eval`, `wwm-nav-plot`, `wwm-nav-generate-radio-env`,
and `wwm-nav-verify`.

## Repository Layout

```text
wwm_nav/
  agent.py                 # WWM-NAV training/evaluation loop
  env.py                   # Grid UAV navigation environment
  radiomap.py              # Radio-map loader and physical context helpers
  radio_env/               # WWM-NAV SNARM-style physical radio environment
  models/
    rssm.py                # RSSM world model
    dreamer_actor_critic.py
    losses.py
  tools/plotting.py
scripts/                    # CLI entrypoints
configs/                    # Reference YAML config
data/wwm_nav/               # Locally generated radio map
```

## Data Format

`arr_0` stores outage probabilities, while `arr_1` and `arr_2` store the map
axes. Physics-conditioned training also requires base-station geometry,
sector angles, transmit power, building heights, and best-link map fields.
Values near 1 mean higher outage risk.

## Method Boundary

The Actor objective uses the RSSM reward and done heads inside imagined
rollouts. Outage prediction is supervised and logged, while risk primarily
enters through the environment reward learned by the reward head.

## Main Outputs

Training writes:

- `config.json`: resolved run configuration;
- `history.csv`: per-episode metrics;
- `training_tasks.csv`: exact sampled start/goal sequence;
- `episode_trajectories.npz`: every training trajectory and episode-level risk summary;
- `training_state.pt`: full resumable state, including replay and RNG states;
- `provenance.json`: map, task, source-tree, Python, NumPy, Torch, CUDA, and cuDNN identity;
- `checkpoint.pt`: final RSSM, Actor, and Critic weights used for evaluation.

Evaluation writes:

- `eval_rollout.npz`: states, actions, rewards, outages, success flag, and step count.
- `shared_trials/trial_pairs.csv`: frozen shared start/goal tasks;
- `shared_trials/trial_metrics.csv` and `trial_metrics.npz`: per-trial metrics;
- `shared_trials/summary.json` and `summary.md`: aggregate statistics.

## Evidence Boundary

This repository packages the existing research-one navigation behavior as a standalone WWM-NAV codebase. The formal contract fixes source behavior, map contents, parameters, task RNG, and saved artifacts. Exact bitwise equality across different GPU architectures or Torch/CUDA/cuDNN versions is not guaranteed; `provenance.json` records that runtime boundary for each run.
