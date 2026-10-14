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

It does not include the communication/uplink controller layer, COMUDO power queues, continual-learning COLD-RSSM protocols, private experiment manifests, or historical checkpoint contracts.

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

## Quick Start

Create a synthetic sample map:

```bash
python scripts/generate_sample_map.py --output data/sample_radiomap.npz
```

Train with command-line flags:

```bash
python scripts/train_wwm_nav.py \
  --radiomap data/sample_radiomap.npz \
  --save-dir outputs/sample \
  --episodes 500 \
  --device cuda
```

Or use the documented YAML contract:

```bash
python scripts/train_wwm_nav.py --config configs/wwm_nav_default.yaml --save-dir outputs/sample
```

For a CPU smoke run, set `--device cpu` and use fewer episodes.

Evaluate a checkpoint:

```bash
python scripts/evaluate_wwm_nav.py \
  --config configs/wwm_nav_default.yaml \
  --save-dir outputs/sample \
  --device cuda
```

This writes `outputs/sample/eval_rollout.npz`.

Plot the evaluation trajectory:

```bash
python scripts/plot_trajectory.py \
  --radiomap data/sample_radiomap.npz \
  --rollout outputs/sample/eval_rollout.npz \
  --output outputs/sample/eval_trajectory.png
```

Console scripts are also registered after installation: `wwm-nav-train`, `wwm-nav-eval`, `wwm-nav-plot`, and `wwm-nav-generate-map`.

## Repository Layout

```text
wwm_nav/
  agent.py                 # WWM-NAV training/evaluation loop
  env.py                   # Grid UAV navigation environment
  radiomap.py              # Radio-map loader and physical context helpers
  models/
    rssm.py                # RSSM world model
    dreamer_actor_critic.py
    losses.py
  tools/plotting.py
scripts/                    # CLI entrypoints
configs/                    # Reference YAML config
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
- `checkpoint.pt`: RSSM, Actor, and Critic weights.

Evaluation writes:

- `eval_rollout.npz`: states, actions, rewards, outages, success flag, and step count.

## Evidence Boundary

This repository packages the existing research-one navigation behavior as a standalone WWM-NAV codebase. It is meant for inspection, reproduction, and extension. Training quality depends on the radio map, seed, compute budget, and configured evaluation protocol.
