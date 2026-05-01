# Program Controller

This example rebuilds a stronger causal descendant from current kernel primitives.

It stays project-local on purpose:

- packed local memory comes from [`NgramMemory`](../../../../src/decepticons/ngram_memory.py)
- exact repair comes from [`ExactContextMemory`](../../../../src/decepticons/exact_context.py)
- controller features come from [`bridge_feature_arrays`](../../../../src/decepticons/bridge_features.py)
- span grouping comes from [`select_scored_spans`](../../../../src/decepticons/span_selection.py)
- route logic stays local to this example

The shape is controller-heavy and programmatic:

- route among prior, exact, and repair programs
- learn a separate repair-strength controller
- group repair dominance into spans

Run from the repo root:

```bash
PYTHONPATH=src python3 examples/projects/causal/program_controller/probe.py
PYTHONPATH=src python3 examples/projects/causal/program_controller/smoke.py
```
