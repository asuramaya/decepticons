# Packed Memory Controller

This example is a causal memory-first descendant built from current kernel primitives.

It stays in project space on purpose:

- packed statistical memory comes from [`NgramMemory`](../../../../src/decepticons/ngram_memory.py)
- exact repair comes from [`ExactContextMemory`](../../../../src/decepticons/exact_context.py)
- trust features come from [`bridge_feature_arrays`](../../../../src/decepticons/bridge_features.py)
- the tiny trust controller is example-local policy, not a kernel abstraction

This is a rebuild of the current memory-first frontier shape:

- packed prior
- exact repair/cache
- confidence-aware controller that decides how much to trust memory

Run from the repo root:

```bash
PYTHONPATH=src python3 examples/projects/causal/packed_memory_controller/probe.py
PYTHONPATH=src python3 examples/projects/causal/packed_memory_controller/smoke.py
```
