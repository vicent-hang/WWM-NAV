# Contributing

Contributions are welcome once the public repository is opened. Please keep the scope focused on WWM-NAV: risk-aware UAV navigation with RSSM/Dreamer-style world models. Communication-control, COMUDO, and continual-learning experiments are intentionally out of scope for this repository.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Before Opening a Pull Request

Run lightweight checks locally when possible:

```bash
python -m compileall wwm_nav scripts
ruff check .
```

Training runs are not required for small code-quality pull requests. If a change affects learning behavior, document the exact command, seed, map, and output directory used for validation.

## Scope Rules

- Keep public method names as `WWM-NAV`.
- Do not add private experiment paths or unpublished artifact assumptions.
- Keep radio-map outage semantics explicit: values near 1 mean higher outage risk.
- Separate implementation behavior from unverified research claims.
