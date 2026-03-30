# Early Conker-Like Example

This folder is a small downstream example for the library kernel.

It models the early `conker` direction as:

- a base byte model
- an exact-context count memory with `exact1`, `exact2`, and `exact3` style backoff levels
- a support-weighted mixer that prefers exact memory when the context is well supported

The example uses the kernel API directly:

- `ExactContextConfig`
- `ExactContextMemory`
- `ExactContextPrediction`
- `SupportMixConfig`
- `SupportWeightedMixer`

## Run

From the repository root:

```bash
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/conker_early_like/smoke.py
PYTHONPATH=open-predictive-coder/src python3 open-predictive-coder/examples/projects/conker_early_like/run.py --mode demo
```

## What It Shows

- exact-context memory can capture small repeated suffixes
- support should steer the blend, not dominate it blindly
- the base byte model still matters when exact history is sparse
