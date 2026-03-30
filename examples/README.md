# Examples

This directory is split into three roles.

## 1. Quickstart

- [`quickstart.py`](./quickstart.py)

This is the smallest end-to-end package example. It is for basic install and import smoke, not for boundary testing.

## 2. Project Descendants

These are under [`projects/`](./projects). They are the main way the repo tests whether the kernel boundary is drawn
honestly.

- `carving_machine_like`
  ancestor-style hierarchical descendant
- `conker_early_like`
  early exact-context causal descendant
- `conker_1_like`, `conker_2_like`, `conker_3_like`
  three causal composition variants
- `oracle_analysis_like`
  analysis-only descendant that reuses sampled readout, routing, and train-mode checkpoints
- `brelt_like`
  byte-patch latent descendant shaped after the real `brelt` repo

Read the project README before reading the code. Each one explains which parts are kernel reuse and which parts are
still project policy.

## 3. Development Tools

These are under [`tools/`](./tools).

Right now the main package is:

- [`tools/diagnostics/`](./tools/diagnostics)

That tooling exists to replace ad hoc analysis scripts with reusable example-facing helpers while keeping them outside
`src/`.

## Recommended Reading Order

If you want to understand the repo through examples:

1. `quickstart.py`
2. `projects/carving_machine_like`
3. `projects/conker_early_like`
4. `projects/conker_1_like` through `conker_3_like`
5. `projects/oracle_analysis_like`
6. `projects/brelt_like`
