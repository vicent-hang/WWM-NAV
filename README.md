# WWM-NAV

WWM-NAV implements multi-step UAV navigation with a physics-conditioned RSSM
and a DreamerV3 discrete Actor-Critic. Training uses a generated SNARM-style
radio environment and runs on CUDA by default.

Run the following commands from the repository root.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Ruff checks only:

```bash
python -m pip install ruff
ruff check .
```

## Generate the radio environment

```bash
python scripts/generate_radio_env.py \
  --output data/wwm_nav/radioenvir_wwm_nav.npz \
  --seed 7 \
  --fast-fading-samples 1000 \
  --los-samples 100

python scripts/verify_wwm_nav_setup.py \
  --config configs/wwm_nav_default.yaml
```

## Train

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml
```

The default experiment runs for 1,500 episodes with at most 300 steps per
episode. Formal training prints one progress line every 20 episodes and updates
`training_state.pt` at the same interval. Evaluation is not run during training;
the final checkpoint and training figures are written after training finishes.

Resume an interrupted run:

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --resume outputs/wwm_nav_seed1/training_state.pt
```

Only resume a state created with the current experiment contract. Start a new
run when using training states produced by an older implementation.

## Evaluate the final checkpoint

```bash
python scripts/evaluate_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/wwm_nav_seed1/checkpoint.pt
```

## Run the 100-trial evaluation

```bash
python scripts/evaluate_wwm_nav_trials.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/wwm_nav_seed1/checkpoint.pt \
  --save-dir outputs/wwm_nav_seed1/shared_trials \
  --num-trials 100 \
  --seed 123 \
  --min-initial-distance 800 \
  --plot-all
```

The evaluator saves the sampled trial pairs, each rollout, per-trial metrics,
and aggregate statistics.

## Plot an evaluation trajectory

```bash
python scripts/plot_trajectory.py \
  --radiomap data/wwm_nav/radioenvir_wwm_nav.npz \
  --rollout outputs/wwm_nav_seed1/eval_rollout.npz \
  --output outputs/wwm_nav_seed1/eval_trajectory.png
```

## Main training outputs

These files are saved under `outputs/wwm_nav_seed1/`:

- `checkpoint.pt`: final RSSM, Actor, and Critic parameters.
- `training_state.pt`: resumable training state.
- `history.csv`: per-episode metrics.
- `training_tasks.csv`: sampled start and goal positions.
- `episode_trajectories.npz`: training trajectories.
- `wwm_nav_training.png`: return, success, outage, steps, and measured cells.
- `wwm_nav_world_model_training.png`: RSSM diagnostics.
- `wwm_nav_actor_critic_training.png`: Actor-Critic diagnostics.
