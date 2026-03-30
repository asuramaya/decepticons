# Tests

The test tree is organized by purpose, not by historical order.

## Kernel Primitive Tests

These validate reusable mechanisms in `src/`.

Examples:

- `test_reservoir.py`
- `test_delay.py`
- `test_hierarchical.py`
- `test_linear_memory.py`
- `test_control.py`
- `test_gating.py`
- `test_routing.py`
- `test_predictive_surprise.py`
- `test_modulation.py`
- `test_sampled_readout.py`
- `test_artifacts.py`

## Runtime And Evaluation Tests

These validate scoring, reporting, and runtime surfaces.

Examples:

- `test_eval.py`
- `test_train_eval.py`
- `test_train_modes.py`
- `test_runtime_knobs.py`

## Boundary Tests

These check the split between kernel and project policy.

Examples:

- `test_kernel_project_split.py`
- `test_example_projects.py`

## Project Descendant Tests

These validate the example descendants directly.

Examples:

- `test_carving_machine_like_fidelity.py`
- `test_oracle_analysis_like.py`
- `test_brelt_like.py`
- `test_diagnostics_examples.py`

## Practical Rule

When adding a new feature:

- if it lands in `src/`, add a kernel primitive test
- if it changes runtime behavior, add a runtime test
- if it changes the kernel/project boundary, add or update a boundary test
- if it is project-local, test it in the corresponding descendant suite
