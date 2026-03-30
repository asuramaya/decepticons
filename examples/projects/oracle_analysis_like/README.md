# oracle_analysis_like

This folder is a small analysis-only descendant probe.

It does not implement a causal runtime codec. Instead, it compares a forward
causal scan against a reverse oracle scan so we can test which kernel surfaces
generalize beyond the ancestor example.

Kernel primitives used here:

- `HierarchicalSubstrate`
- `HierarchicalFeatureView`
- `SampledMultiscaleReadout`
- `SummaryRouter`
- `TrainModeConfig`

Project policy that stays local:

- which checkpoints to inspect
- how to compare forward and reverse states
- how to score oracle preference
- how much routing bias to give the oracle side

## Run

From the repository root:

```bash
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/oracle_analysis_like/probe.py
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/oracle_analysis_like/smoke.py
```

## What It Shows

- sampled multiscale readout can be used as an analysis feature surface, not only as a causal model input
- train-mode checkpointing can drive analysis cadence
- diagnostics can compare forward and reverse feature traces without turning the example into a codec
