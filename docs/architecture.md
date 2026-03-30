# Architecture

This document is the shortest route to understanding the repo as code rather than as a historical story.

## Reading Order

If you are new to the repo, read things in this order:

1. [`README.md`](../README.md)
2. [`kernel_matrix.md`](./kernel_matrix.md)
3. this file
4. [`examples/README.md`](../examples/README.md)
5. the specific example project README you care about

Use [`related_work.md`](./related_work.md) and [`landscape.md`](./landscape.md) for research and ecosystem context, not
for code orientation.

## Layer Model

The repo is intentionally split into three layers.

### 1. Kernel

The `src/open_predictive_coder/` package contains reusable mechanisms only.

It owns:

- substrate dynamics
- controller-side summaries, gates, routing, and modulation
- memory and latent primitives
- feature views and sampled readout
- readouts, experts, and scoring utilities
- runtime surfaces like traces, eval, train modes, and artifact accounting

It does not own:

- project-specific legality rules
- benchmark claims
- one descendant's routing policy
- one descendant's latent composition policy
- one descendant's oracle privileges

### 2. Project-Layer Descendants

The `examples/projects/` tree is where the kernel gets pressure-tested by actual descendant shapes.

These are not toy demos. They are boundary tests.

- `carving_machine_like/`
  ancestor-style hierarchical substrate plus predictor/gating policy
- `conker_early_like/`
  early exact-context causal memory shape
- `conker_1_like/`, `conker_2_like/`, `conker_3_like/`
  three different causal composition policies built from kernel primitives
- `oracle_analysis_like/`
  analysis-only descendant that reuses sampled readout, routing, and train-mode checkpoints
- `brelt_like/`
  byte-patch latent descendant shaped after the real `brelt` repo

If a mechanism is repeated across multiple descendants, it is a candidate for promotion into `src/`.

### 3. Development Tooling

The `examples/tools/` tree is for development and analysis support, not kernel code.

Right now that mainly means:

- `examples/tools/diagnostics/`

This keeps `look.py` / `look2.py` / `silence_test.py` style workflows available without polluting the public package.

## Package Map

The kernel is easiest to understand by category rather than by filename order.

### Foundation

- [`codecs.py`](../src/open_predictive_coder/codecs.py)
- [`config.py`](../src/open_predictive_coder/config.py)
- [`metrics.py`](../src/open_predictive_coder/metrics.py)

### Substrates

- [`reservoir.py`](../src/open_predictive_coder/reservoir.py)
- [`delay.py`](../src/open_predictive_coder/delay.py)
- [`linear_memory.py`](../src/open_predictive_coder/linear_memory.py)
- [`mixed_memory.py`](../src/open_predictive_coder/mixed_memory.py)
- [`hierarchical.py`](../src/open_predictive_coder/hierarchical.py)
- [`substrates.py`](../src/open_predictive_coder/substrates.py)
- [`factories.py`](../src/open_predictive_coder/factories.py)

### Control And Side Channels

- [`control.py`](../src/open_predictive_coder/control.py)
- [`gating.py`](../src/open_predictive_coder/gating.py)
- [`routing.py`](../src/open_predictive_coder/routing.py)
- [`modulation.py`](../src/open_predictive_coder/modulation.py)
- [`predictive_surprise.py`](../src/open_predictive_coder/predictive_surprise.py)

### Memory, Latents, And Views

- [`exact_context.py`](../src/open_predictive_coder/exact_context.py)
- [`latents.py`](../src/open_predictive_coder/latents.py)
- [`segmenters.py`](../src/open_predictive_coder/segmenters.py)
- [`views.py`](../src/open_predictive_coder/views.py)
- [`linear_views.py`](../src/open_predictive_coder/linear_views.py)
- [`hierarchical_views.py`](../src/open_predictive_coder/hierarchical_views.py)
- [`sampled_readout.py`](../src/open_predictive_coder/sampled_readout.py)

### Readouts, Experts, And Runtime

- [`readout.py`](../src/open_predictive_coder/readout.py)
- [`readouts.py`](../src/open_predictive_coder/readouts.py)
- [`experts.py`](../src/open_predictive_coder/experts.py)
- [`runtime.py`](../src/open_predictive_coder/runtime.py)
- [`eval.py`](../src/open_predictive_coder/eval.py)
- [`train_eval.py`](../src/open_predictive_coder/train_eval.py)
- [`train_modes.py`](../src/open_predictive_coder/train_modes.py)
- [`artifacts.py`](../src/open_predictive_coder/artifacts.py)

### Adapters And Presets

- [`adapters.py`](../src/open_predictive_coder/adapters.py)
- [`model.py`](../src/open_predictive_coder/model.py)
- [`presets.py`](../src/open_predictive_coder/presets.py)
- [`cli.py`](../src/open_predictive_coder/cli.py)

## Promotion Rule

Code moves from a project into `src/` only when all of these are true:

1. it is a mechanism rather than a project policy
2. at least two descendants want the same thing
3. the generalized API is simpler than keeping the duplication in project code

That rule is the main defense against turning the kernel into a renamed collection of branches.

## Current Boundary

Stable kernel examples of the right kind of promotion:

- `LinearMemorySubstrate`
- `FrozenReadoutExpert`
- `PredictiveSurpriseController`
- `HormoneModulator`
- `SampledMultiscaleReadout`
- `TrainModeConfig`
- `ArtifactMetadata` / `ReplaySpan` / `ArtifactAccounting`

Still project-local on purpose:

- Conker mixer and residual-repair policy
- oracle-vs-causal comparison policy
- `brelt` patch-boundary tuning and local/global bridge composition
- ancestor-specific predictor head choices

## Immediate Architectural Direction

The next real jump is not another example. It is the first true `src/`-level causal adapter that consumes:

- exact-context memory
- experts
- train/eval surfaces
- artifact accounting

That is the point where the repo graduates from "kernel plus descendants" to "kernel plus at least one real downstream contract."
