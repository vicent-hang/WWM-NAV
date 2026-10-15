# WWM-NAV run commands

Run every command from the repository root. The formal experiment requires a
CUDA-capable machine.

## 1. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The editable install is required when invoking files under `scripts/`; it makes
the `wwm_nav` package importable and registers the console entry points.

## 2. Generate and verify the WWM-NAV map

```bash
python scripts/generate_radio_env.py \
  --output data/wwm_nav/radioenvir_wwm_nav.npz \
  --seed 7 \
  --fast-fading-samples 1000 \
  --los-samples 100
python scripts/verify_wwm_nav_setup.py --config configs/wwm_nav_default.yaml
```

The generator runs the migrated WWM-NAV city, LoS/NLoS, antenna, path-loss,
fast-fading, and empirical-outage implementation directly. It writes the outage
grid, held-out radio samples, building map, base-station geometry, best-link
labels, generation parameters, and map contract required by training. The map
is generated locally and ignored by Git; no pre-generated output file is needed.

## 3. Formal WWM-NAV training

```bash
python scripts/train_wwm_nav.py --config configs/wwm_nav_default.yaml
```

The formal config uses seed 1, 1,500 episodes, 300 maximum steps, random starts
at least 800 m from the fixed `(1400, 1600)` goal, categorical RSSM, DreamerV3
Actor-Critic, and the fixed-weight SNARM-style navigation prior. Strict contract
validation fails before training if any registered effective parameter, map
schema, generation setting, or physical geometry differs.

## 4. Resume after interruption

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --resume outputs/wwm_nav_seed1/training_state.pt
```

Keep the same config, generated map, and output directory. The rolling state
contains models, optimizers, normalizers, sequence replay, evaluation leaders,
sampled tasks, and Python/NumPy/Torch/CUDA RNG states.

## 5. Evaluate and plot the final checkpoint

```bash
python scripts/evaluate_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/wwm_nav_seed1/checkpoint.pt

python scripts/plot_trajectory.py \
  --radiomap data/wwm_nav/radioenvir_wwm_nav.npz \
  --rollout outputs/wwm_nav_seed1/eval_rollout.npz \
  --output outputs/wwm_nav_seed1/eval_trajectory.png
```

Training itself evaluates the fixed start/goal every 20 episodes and retains
both the shortest-success and risk-aware best rollouts.

## Reproducibility boundary

The repository locks the algorithm path, effective WWM-NAV parameters, radio
environment generation contract, physical context, task sampler, seed, periodic
evaluation, and resumable RNG state. GPU kernels and dependency implementations
can vary across Torch/CUDA/cuDNN and hardware versions, so exact cross-machine
bitwise identity is not promised. Each formal run records those versions and
the source-tree state in `provenance.json`.
