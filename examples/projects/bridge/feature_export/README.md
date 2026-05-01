# Feature Export

This example is a thin bridge-style descendant built on the shared probability-to-feature transforms.

It stays project-local on purpose:

- it synthesizes two probability streams locally
- it turns them into export-style summary features
- it does not widen `src/` with descendant-specific policy

Kernel pieces reused here:

- [`bridge_feature_arrays`](../../../../src/decepticons/bridge_features.py)

Entry points:

- [`probe.py`](./probe.py)
- [`smoke.py`](./smoke.py)

Run from the repo root:

```bash
PYTHONPATH=src python3 examples/projects/bridge/feature_export/probe.py
PYTHONPATH=src python3 examples/projects/bridge/feature_export/smoke.py
```
