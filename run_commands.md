# WWM-NAV run commands

Run every command from the repository root. The formal experiment requires a
CUDA-capable machine; the smoke check can run on CPU.

## 1. Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The editable install is required when invoking files under `scripts/`; it makes
the `wwm_nav` package importable and registers the console entry points.

## 2. Prepare and verify the C28.5 map

```bash
python scripts/prepare_c28_5_map.py
python scripts/verify_c28_5_setup.py --config configs/wwm_nav_default.yaml
```

The first command verifies the bundled legacy archive against its registered
SHA-256, preserves all five historical arrays exactly, and adds deterministic
building/base-station metadata. The generated map and manifest are ignored by
Git. Re-running preparation may change ZIP container bytes, but the printed
semantic SHA-256 remains content-stable.

The migrated C28.5 physical radio-environment implementation can also generate
a fresh SNARM-style map:

```bash
python scripts/generate_radio_env.py \
  --output data/radioenvir_c28_5_generated.npz \
  --seed 7 \
  --fast-fading-samples 1000 \
  --los-samples 100
```

This command exposes the original C28.5 city, LoS/NLoS, antenna, path-loss,
fast-fading, and empirical-outage code path. The generated map is not a
replacement for the hash-registered legacy map used by the formal config.

## 3. Formal C28.5 training

```bash
python scripts/train_wwm_nav.py --config configs/wwm_nav_default.yaml
```

The formal config uses seed 1, 1,500 episodes, 300 maximum steps, random starts
at least 800 m from the fixed `(1400, 1600)` goal, categorical RSSM, DreamerV3
Actor-Critic, and the fixed-weight SNARM-style navigation prior. Strict contract
validation fails before training if any registered effective parameter or map
content differs.

## 4. Resume after interruption

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --resume outputs/c28_5_wwm_nav_seed1/training_state.pt
```

Keep the same config, prepared map, and output directory. The rolling state
contains models, optimizers, normalizers, sequence replay, evaluation leaders,
sampled tasks, and Python/NumPy/Torch/CUDA RNG states.

## 5. Evaluate and plot the final checkpoint

```bash
python scripts/evaluate_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/c28_5_wwm_nav_seed1/checkpoint.pt

python scripts/plot_trajectory.py \
  --radiomap data/c28_5/radioenvir_c28_5.npz \
  --rollout outputs/c28_5_wwm_nav_seed1/eval_rollout.npz \
  --output outputs/c28_5_wwm_nav_seed1/eval_trajectory.png
```

Training itself evaluates the fixed start/goal every 20 episodes and retains
both the shortest-success and risk-aware best rollouts.

## 6. CPU smoke check

This checks installation and interfaces only. It is not a research run.

```bash
python scripts/generate_sample_map.py --output data/sample_radiomap.npz
python scripts/train_wwm_nav.py --config configs/smoke.yaml
python scripts/evaluate_wwm_nav.py \
  --config configs/smoke.yaml \
  --checkpoint outputs/smoke/checkpoint.pt
```

## 7. Test suite

```bash
python -m pytest
```

## Reproducibility boundary

The repository locks the algorithm path, effective C28.5 parameters, legacy
map core, reconstructed physical context, task sampler, seed, periodic
evaluation, and resumable RNG state. GPU kernels and dependency implementations
can vary across Torch/CUDA/cuDNN and hardware versions, so exact cross-machine
bitwise identity is not promised. Each formal run records those versions and
the source-tree state in `provenance.json`.
