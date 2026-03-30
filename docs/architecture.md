# Architecture

This document is the shortest route to understanding the repo as code rather than as a historical story.

## Reading Order

If you are new to the repo, read things in this order:

1. [`README.md`](../README.md)
2. [`kernel_matrix.md`](./kernel_matrix.md)
3. [`frontier_pass.md`](./frontier_pass.md)
4. this file
5. [`examples/README.md`](../examples/README.md)
6. [`lineage.md`](./lineage.md)
7. the specific example project README you care about

Use [`related_work.md`](./related_work.md) and [`landscape.md`](./landscape.md) for research and ecosystem context, not
for code orientation.

## Layer Model

The repo is intentionally split into three layers.

### 1. Kernel

The `src/open_predictive_coder/` package contains reusable mechanisms only.

It owns:

- substrate dynamics
- controller-side summaries, gates, routing, and modulation
- memory, latent, and learned patch-latent primitives
- family-neutral probability diagnostics and bridge-side feature transforms
- feature views and sampled readout
- readouts, experts, and scoring utilities
- runtime surfaces like traces, eval, train modes, and artifact accounting
- the shared contracts above those primitives when they stay mechanism-level: causal, oracle, bridge-export, noncausal reconstruction, paired teacher/export, and artifact-boundary audit helpers

It does not own:

- project-specific legality rules
- benchmark claims
- one descendant's routing policy
- one descendant's latent composition policy
- one descendant's oracle privileges
- teacher-export policy
- payload-wire policy
- higher-order causal program/controller policy

### 2. Project-Layer Descendants

The `examples/projects/` tree is where the kernel gets pressure-tested by actual descendant shapes.

These are not toy demos. They are boundary tests.

- `ancestor/hierarchical_predictive/`
  ancestor-style hierarchical substrate plus predictor/gating policy
- `causal/exact_context_repair/`
  early exact-context causal memory shape
- `causal/memory_stability/`, `causal/linear_correction/`, `causal/residual_repair/`
  three different causal composition policies built from kernel primitives
- `causal/statistical_memory/`
  causal example that composes the shared statistical-backoff memory layer and exact-context repair
- `causal/packed_memory_controller/`
  causal memory-first descendant that adds an example-local trust controller over shared backoff priors and exact repair
- `bridge/proxy_features/`
  bridge-style descendant that turns probability streams into causal proxy features
- `bridge/feature_export/`
  bridge-style descendant that packages paired probability streams into a small export/report flow
- `bridge/agreement_export/`
  bridge-style descendant that focuses on agreement and disagreement over paired probability streams
- `noncausal/field_reconstruction/`
  noncausal field reconstruction descendant built from bidirectional context, exact-context memory, and replay accounting
- `oracle/bidirectional_analysis/`
  analysis-only descendant that reuses sampled readout, routing, and train-mode checkpoints
- `byte_latent/patch_latent/`
  byte-patch latent descendant shaped as a general patch-latent example

If a mechanism is repeated across multiple descendants, it is a candidate for promotion into `src/`.
The shared causal, oracle, and bridge adapters are the current examples of that rule: they live in `src/` only where
they read as shared contracts rather than descendant-specific models.

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
- [`oscillatory_memory.py`](../src/open_predictive_coder/oscillatory_memory.py)
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

- [`bridge_features.py`](../src/open_predictive_coder/bridge_features.py)
- [`bidirectional_context.py`](../src/open_predictive_coder/bidirectional_context.py)
- [`exact_context.py`](../src/open_predictive_coder/exact_context.py)
- [`latents.py`](../src/open_predictive_coder/latents.py)
- [`learned_segmentation.py`](../src/open_predictive_coder/learned_segmentation.py)
- [`ngram_memory.py`](../src/open_predictive_coder/ngram_memory.py)
- [`statistical_backoff.py`](../src/open_predictive_coder/statistical_backoff.py)
- [`patch_latent_blocks.py`](../src/open_predictive_coder/patch_latent_blocks.py)
- [`probability_diagnostics.py`](../src/open_predictive_coder/probability_diagnostics.py)
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
- [`span_selection.py`](../src/open_predictive_coder/span_selection.py)
- [`train_eval.py`](../src/open_predictive_coder/train_eval.py)
- [`train_modes.py`](../src/open_predictive_coder/train_modes.py)
- [`artifacts.py`](../src/open_predictive_coder/artifacts.py)

### Adapters And Presets

- [`adapters.py`](../src/open_predictive_coder/adapters.py)
- [`causal_predictive.py`](../src/open_predictive_coder/causal_predictive.py)
- [`bridge_export.py`](../src/open_predictive_coder/bridge_export.py)
- [`oracle_analysis.py`](../src/open_predictive_coder/oracle_analysis.py)
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
- `select_scored_spans` / `replay_spans_from_scores`
- `ProbabilityDiagnostics` / `probability_diagnostics`

Still project-local on purpose:

- descendant mixer and residual-repair policy
- interpretation and reporting around oracle comparisons
- rate-distortion weighting, second compression stage, and quantization/export policy in the patch-latent example
- project-specific bridge/export policy above the shared probability-to-feature transforms
- ancestor-specific predictor head choices

Recent shared promotion:

- `CausalPredictiveAdapter`
- `OracleAnalysisAdapter`
- `BridgeExportAdapter`
- `NoncausalReconstructiveAdapter`
- `TeacherExportAdapter`
- `ArtifactAuditRecord` / `ArtifactAuditSummary`

## Immediate Architectural Direction

The next real jump is not another single descendant. It is pressure-testing the shared causal, oracle, bridge,
noncausal, and teacher/export adapters across more than one consumer so the current contract line either holds or
shrinks.

That next shared layer now includes a noncausal reconstruction adapter, a paired teacher/export contract, and generic
artifact-boundary audit helpers. Keep their policy use local even though the contracts now live in `src/`.

That means:

- thinning the causal descendants around the causal adapter
- keeping the bidirectional-analysis example thin around the oracle adapter
- hardening the shared runtime/accounting contract
- keeping bridge export generic while more than one bridge-shaped consumer pushes on it
- using the noncausal descendants to decide whether the shared noncausal contract should stay narrow or absorb one more seam

That is how the repo graduates from "first shared contracts exist" to "the contracts are actually stable enough to keep."
