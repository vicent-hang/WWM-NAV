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

## Reproduce C28.5

The repository includes the provenance-bound legacy C28.5 outage map. First
verify it and reconstruct the deterministic physical metadata required by the
physics-conditioned RSSM:

```bash
python scripts/prepare_c28_5_map.py
python scripts/verify_c28_5_setup.py --config configs/wwm_nav_default.yaml
```

Run the registered 1,500-episode, 300-step experiment:

```bash
python scripts/train_wwm_nav.py --config configs/wwm_nav_default.yaml
```

The YAML file is the standalone form of the original C28.5 command contract.
Strict validation rejects accidental changes to effective research parameters.
Training writes a resumable state every 20 episodes, fixed-start evaluation
rollouts, resolved configuration, sampled training tasks, and map/code/runtime
provenance.

Resume an interrupted run without changing its output directory or config:

```bash
python scripts/train_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --resume outputs/c28_5_wwm_nav_seed1/training_state.pt
```

Evaluate a checkpoint:

```bash
python scripts/evaluate_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --checkpoint outputs/c28_5_wwm_nav_seed1/checkpoint.pt
```

This writes `outputs/c28_5_wwm_nav_seed1/eval_rollout.npz` and
`eval_summary.json`.

Plot the evaluation trajectory:

```bash
python scripts/plot_trajectory.py \
  --radiomap data/c28_5/radioenvir_c28_5.npz \
  --rollout outputs/c28_5_wwm_nav_seed1/eval_rollout.npz \
  --output outputs/c28_5_wwm_nav_seed1/eval_trajectory.png
```

See [run_commands.md](run_commands.md) for the complete setup, training,
resume, evaluation, plotting, smoke, and test commands.

## Smoke Check

The synthetic map and smoke config verify interfaces only; they must not be
used for research results:

```bash
python scripts/generate_sample_map.py --output data/sample_radiomap.npz
python scripts/train_wwm_nav.py --config configs/smoke.yaml
```

Console scripts are also registered after installation: `wwm-nav-train`,
`wwm-nav-eval`, `wwm-nav-plot`, `wwm-nav-generate-map`,
`wwm-nav-generate-radio-env`, `wwm-nav-prepare-c28-map`, and
`wwm-nav-verify-c28`.

## Repository Layout

```text
wwm_nav/
  agent.py                 # WWM-NAV training/evaluation loop
  env.py                   # Grid UAV navigation environment
  radiomap.py              # Radio-map loader and physical context helpers
  radio_env/               # C28.5 SNARM-style physical radio environment
  models/
    rssm.py                # RSSM world model
    dreamer_actor_critic.py
    losses.py
  tools/plotting.py
scripts/                    # CLI entrypoints
configs/                    # Reference YAML config
data/c28_5/                 # Frozen legacy C28.5 map and preparation notes
docs/                       # Method and data-format notes
tests/                      # Lightweight tests
```

## Data Format

See [docs/radiomap_format.md](docs/radiomap_format.md). In short, `arr_0` stores outage probabilities, `arr_1` and `arr_2` store map axes, and physics-conditioned training also expects base-station and height-map fields. Values near 1 mean higher outage risk.

## Method Boundary

See [docs/method_overview.md](docs/method_overview.md). The default Actor objective uses the RSSM reward head and done head inside imagined rollouts. Outage prediction is supervised and logged, while risk primarily enters through the environment reward learned by the reward head.

## Main Outputs

Training writes:

- `config.json`: resolved run configuration;
- `history.csv`: per-episode metrics;
- `training_tasks.csv`: exact sampled start/goal sequence;
- `episode_trajectories.npz`: every training trajectory and episode-level risk summary;
- `evaluation_history.json`: fixed-start evaluation history;
- `eval_rollout_best_shortest.npz` and `eval_rollout_best_risk_aware.npz`;
- `training_state.pt`: full resumable state, including replay and RNG states;
- `provenance.json`: map, task, source-tree, Python, NumPy, Torch, CUDA, and cuDNN identity;
- `checkpoint.pt`: final RSSM, Actor, and Critic weights.

Evaluation writes:

- `eval_rollout.npz`: states, actions, rewards, outages, success flag, and step count.

## Evidence Boundary

This repository packages the existing research-one navigation behavior as a standalone WWM-NAV codebase. The formal contract fixes source behavior, map contents, parameters, task RNG, and saved artifacts. Exact bitwise equality across different GPU architectures or Torch/CUDA/cuDNN versions is not guaranteed; `provenance.json` records that runtime boundary for each run.
