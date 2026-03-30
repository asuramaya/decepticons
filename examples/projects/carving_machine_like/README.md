# carving_machine_like

This folder is a runnable example project for the kernel layer in `open-predictive-coder`.

It is intentionally small and is meant to act as a development and smoke-test target for the
`carving_machine`-style idea:

- hierarchical substrate dynamics
- multiscale feature views
- gate-aware rollout and byte-level readout on top of the hierarchical state

The example stays on the public API and uses the current kernel primitives:

- `HierarchicalSubstrate`
- `HierarchicalFeatureView`
- `ControllerSummary`
- `PathwayGateController`
- `SummaryRouter`
- `HormoneModulator`
- `TrainModeConfig`
- `SampledMultiscaleReadout`
- `RidgeReadout`
- `hierarchical_small()`

The local project layer adds the parts that are still specific to the `carving_machine` family:

- a learned slow/mid-to-fast predictor
- slow-state, surprise-state, and routed pathway gating
- `prediction`, `zeros`, and `random` auxiliary readout modes
- the choice to apply sampled readout over `surprise`, `mid`, and `slow` bands and splice in auxiliary features
- the choice to use hormone outputs as gate-scale modulation
- the choice to treat `TrainModeConfig` as a runtime regime knob for detached vs through-state and rollout checkpoints

## What It Shows

The project has two entry points:

- `smoke.py`: steps a hierarchical substrate through a short sequence, then fits a
  `CarvingMachineKernelAdapter` that composes the substrate, predictor, summary, gate, and sampled-readout pieces.
- `probe.py`: prints the state slices, feature dimensions, and gate-facing adapter surface, including the sampled-readout layout.

## Run

From the repository root:

```bash
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/carving_machine_like/smoke.py
```

```bash
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/carving_machine_like/probe.py
```

## Scope

This is not a new kernel primitive. It is a reference composition layer that tests whether the current
kernel can express a `carving_machine`-style gated hierarchical model, because `carving_machine` is the source lineage
the kernel is being extracted from. If this example needs task policy rather than a shared mechanism, that policy
should stay here instead of being promoted into `src/`.

The reusable sampling mechanism now lives in the kernel as `SampledMultiscaleReadout`. This example still owns
the project-specific choices around which bands to sample (`surprise`, `mid`, `slow`), how to fuse auxiliary
prediction features into the final readout vector, and how to combine routing, hormone modulation, and runtime-mode
selection into the ancestor-specific control path.
