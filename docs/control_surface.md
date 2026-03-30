# Control Surface

This note anchors the next controller extraction step in the actual upstream controller code rather than in repo lore.
The upstream source coordinates below are attribution notes from the broader workspace, not vendored files in this
repository. See [`lineage.md`](./lineage.md).

## Pathway Gates

The first gate surface lives inside `HierarchicalCarverModel` in `carving_machine/models.py#L224`.

Relevant regions:

- config fields and gate heads:
  `carving_machine/models.py#L232`
- gate application helpers:
  `carving_machine/models.py#L306`
- slow-state and surprise-driven gate use in the forward pass:
  `carving_machine/models.py#L350`

The reusable idea is smaller than the model:

- a shared controller summary vector
- two scalar gates, `fast_to_mid` and `mid_to_slow`
- a policy for when those gates are refreshed

Minimal kernel scaffold:

- `PathwayGateConfig`
- `PathwayGateController`
- `PathwayGateValues`

That scaffold should stay independent of any one substrate implementation. It only needs a summary vector or scalar
surprise input and should emit gate values in `[0, 1]`.

## Routing

The next routing surface lives in `RoutedHierarchicalModel` in `carving_machine/models.py#L1129`.

Relevant regions:

- router mode setup:
  `carving_machine/models.py#L1208`
- summary construction for each branch:
  `carving_machine/models.py#L1252`
- route weight computation:
  `carving_machine/models.py#L1279`
- route trace collection:
  `carving_machine/models.py#L1290`

The reusable idea is:

- multiple candidate branch summaries
- a router mode: `equal`, `static`, or projection-from-summary
- route weights plus an optional route trace for analysis

Minimal kernel scaffold:

- `RoutingMode`
- `RoutingDecision`
- `SummaryRouter`

The kernel should not hardcode the exact four `RoutedHierarchicalModel` branches. The branch count and summary vectors
should be external inputs.

## Hormonal Modulation

The modulation surface lives in `HormonalHierarchicalCarverModel` in `carving_machine/models.py#L1354`.

Relevant regions:

- hormone and gate projection setup:
  `carving_machine/models.py#L1404`
- hormone-conditioned fast/mid gating:
  `carving_machine/models.py#L1448`
- optional hormone predictor/readout inclusion:
  `carving_machine/models.py#L1464`

The reusable idea is:

- project a slow summary into a low-dimensional hormone state
- use hormone-driven gates to modulate faster banks
- optionally expose hormones to predictor and readout paths

Minimal kernel scaffold:

- `HormoneModulationConfig`
- `HormoneState`
- `HormoneModulator`

The important boundary is that the kernel should expose modulation as a side-channel primitive, not as a full model
family. Predictor and readout inclusion should remain adapter-level choices.

## Current Extraction State

The shared controller boundary now exists in `src/`:

- `ControllerSummary`
- `ControllerSummaryBuilder`
- `PathwayGateController`
- `SummaryRouter`
- `PredictiveSurpriseController`
- `HormoneModulator`

That means the honest remaining order is now:

1. wire the new predictive-surprise and modulation primitives into more than one descendant
2. keep learned predictor-head policy in example/adapters until repetition forces a wider abstraction
3. only then consider richer learned controller stacks beyond deterministic kernel primitives

That keeps the kernel idea-based instead of baking upstream class structure directly into the public API.
