# Examples

This directory is split into three roles.

## 1. Quickstart

- [`quickstart.py`](./quickstart.py)

This is the smallest end-to-end package example. It is for basic install and import smoke, not for boundary testing.

## 2. Project Descendants

These are under [`projects/`](./projects). They are the main way the repo tests whether the kernel boundary is drawn
honestly.

They are self-contained descendants that live in this repo. They are not imports from sibling repositories.

- `hierarchical_predictive`
  ancestor-style hierarchical predictive example in `projects/ancestor/`
- `exact_context_repair`
  early exact-context causal example in `projects/causal/`
- `memory_stability`, `linear_correction`, `residual_repair`
  three causal composition variants in `projects/causal/`
- `statistical_memory`
  causal statistical-memory composition example in `projects/causal/`
- `packed_memory_controller`
  causal memory-first confidence/controller example over the shared statistical-backoff layer in `projects/causal/`
- `program_controller`
  causal higher-order routing/controller descendant kept project-local above the shared statistical-backoff layer in `projects/causal/`
- `proxy_features`
  bridge-style descendant in `projects/bridge/`
- `feature_export`
  bridge export/report descendant in `projects/bridge/`
- `agreement_export`
  bridge agreement-focused descendant in `projects/bridge/`
- `teacher_export`
  bridge export descendant with local teacher-label and attack-report policy in `projects/bridge/`
- `bidirectional_analysis`
  analysis-only descendant in `projects/oracle/`
- `field_reconstruction`
  noncausal reconstruction descendant in `projects/noncausal/`
- `payload_choice`
  noncausal payload-layout descendant kept project-local in `projects/noncausal/`
- `patch_latent`
  learned byte-patch latent descendant in `projects/byte_latent/`

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
2. `projects/ancestor/hierarchical_predictive`
3. `projects/causal/exact_context_repair`
4. `projects/causal/memory_stability` through `causal/program_controller`
5. `projects/bridge/proxy_features`
6. `projects/bridge/feature_export`
7. `projects/bridge/agreement_export` and `bridge/teacher_export`
8. `projects/noncausal/field_reconstruction` and `noncausal/payload_choice`
9. `projects/oracle/bidirectional_analysis`
10. `projects/byte_latent/patch_latent`
