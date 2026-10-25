# Changelog

## Unreleased

- Set the default C28.5 outage penalty to `10.0`.
- Migrate the provenance-bound 100-trial magazine figures, compact comparison
  artifacts, and self-contained plotting commands into this repository.
- Write all generated experiment and magazine figures under `outputs/` by default.

## 0.1.0 - 2026-10-14

- Initial standalone WWM-NAV extraction.
- Includes the WWM-NAV radio-map navigation environment, categorical RSSM,
  DreamerV3-style Actor-Critic, and SNARM-style navigation prior.
- Adds direct WWM-NAV radio-environment generation, strict registered config,
  train-then-final-evaluation separation, full resumable state, runtime provenance,
  and WWM-NAV-only shared 100-trial evaluation with frozen task pairs and summaries.
