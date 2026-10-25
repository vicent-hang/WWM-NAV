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

## Reproducible CUDA environment

The reference WWM-NAV environment uses Python 3.10, PyTorch 2.5.1, and the
CUDA 12.1 PyTorch runtime. For reproducible experiments on Linux with an NVIDIA
GPU, create an isolated environment:

```bash
conda create -n wwm-nav-cu121 python=3.10 -y
conda activate wwm-nav-cu121
which python
python --version
```

Install the official
[PyTorch 2.5.1 CUDA 12.1 wheels](https://docs.pytorch.org/get-started/previous-versions/):

```bash
python -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install the reference core dependencies, then install this repository without
allowing its broad dependency ranges to replace the pinned versions:

```bash
python -m pip install \
  numpy==2.2.5 \
  matplotlib==3.10.9 \
  PyYAML==6.0.3

python -m pip install -e . --no-deps
```

Verify the active runtime before training:

```bash
python -c "import torch, numpy, matplotlib, yaml; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('cuDNN:', torch.backends.cudnn.version()); print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__); print('PyYAML:', yaml.__version__); print('GPU:', torch.cuda.get_device_name(0)); print('CUDA available:', torch.cuda.is_available())"
```

The key expected values are `PyTorch: 2.5.1+cu121`, `CUDA: 12.1`,
`NumPy: 2.2.5`, `Matplotlib: 3.10.9`, and `PyYAML: 6.0.3`. The CUDA version
reported by `nvidia-smi` is the maximum version supported by the driver; use
`torch.version.cuda` to verify the runtime used by PyTorch.

Before comparing training results across systems, confirm that every run uses
the same source revision, configuration, and radio map:

```bash
git rev-parse HEAD
git status --short
sha256sum configs/wwm_nav_default.yaml
sha256sum data/wwm_nav/radioenvir_wwm_nav.npz
```

Training states are environment- and configuration-specific. When changing the
software environment or experiment configuration, start from a new output
directory instead of resuming an existing `training_state.pt`:

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --save-dir outputs/wwm_nav_cu121_seed1
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

## Plot the standalone radio map

```bash
python scripts/plot_radio_map.py \
  --radiomap data/wwm_nav/radioenvir_wwm_nav.npz \
  --output outputs/radioenvir_wwm_nav.png
```

The figure shows outage probability on a fixed 0-to-1 color scale. Dark regions
have lower outage probability, while bright regions have higher outage
probability. It does not include training or evaluation trajectories. Generated
figures are kept under `outputs/`; `data/` contains experiment inputs only.

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
`trial_metrics.png`, `all_trajectories.png`, and aggregate statistics.

## Plot an evaluation trajectory

```bash
python scripts/plot_trajectory.py \
  --radiomap data/wwm_nav/radioenvir_wwm_nav.npz \
  --rollout outputs/wwm_nav_seed1/eval_rollout.npz \
  --output outputs/wwm_nav_seed1/eval_trajectory.png
```

The final-checkpoint evaluator also writes `eval_trajectory.png` automatically.

## Reproduce the magazine figures

Publication-ready reference figures and their provenance-bound 100-trial
comparison data are stored in `reference/magazine/`. The plotting scripts use
repository-relative inputs and write regenerated figures to `outputs/` by
default:

```bash
python -m scripts.plot_navigation_magazine_case_study
python -m scripts.plot_navigation_radio_map_scene
python -m scripts.plot_navigation_25d_city_radio_map
```

Add `--overwrite` when replacing files already generated in the output
directory. See `reference/magazine/README.md` for the evidence boundary and
the contents of each panel.

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
