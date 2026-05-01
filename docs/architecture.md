# Architecture

The shortest route to understanding the repo as code rather than as a historical
story. For the public website version of this material, see
[the project site](https://asuramaya.github.io/decepticons/). For the kernel
boundary against the runtime descendant, see
[`chronohorn_boundary.md`](./chronohorn_boundary.md).

## Three Layers

```
src/decepticons/        →  reusable kernel (public package)
examples/projects/      →  descendant boundary tests
examples/tools/         →  development and analysis tooling
```

### 1. Kernel — `src/decepticons/`

The kernel owns reusable mechanisms only:

- substrate dynamics
- controller-side summaries, gates, routing, and modulation
- memory, latent, and learned patch-latent primitives
- family-neutral probability diagnostics and bridge feature transforms
- feature views and sampled readout
- readouts, experts, and scoring utilities
- runtime surfaces (traces, eval, train modes, artifact accounting)
- shared mechanism-level contracts for causal, oracle, bridge-export, noncausal
  reconstruction, paired teacher/export, and artifact audit

It does **not** own:

- project-specific legality rules
- benchmark claims
- one descendant's routing or composition policy
- one descendant's oracle privileges
- teacher-export policy
- payload-wire policy
- higher-order causal program/controller policy

### 2. Project Descendants — `examples/projects/`

Pressure-tests the kernel boundary with concrete descendant shapes. These are
not toy demos — they exist so the kernel boundary is drawn under realistic
load. Five families:

- `ancestor/` — hierarchical predictive baseline
- `causal/` — causal composition policies (memory, repair, packed, cache, …)
- `bridge/` — bridge-style boundary descendants
- `noncausal/` — field reconstruction and replay
- `oracle/` — analysis-only bidirectional descendants
- `byte_latent/` — byte-patch latent descendants

If a mechanism is repeated across multiple descendants, it is a candidate for
promotion into `src/`.

### 3. Tooling — `examples/tools/`

Development and analysis support, not kernel code. Currently:

- `examples/tools/diagnostics/` — reusable analysis helpers used by the example
  smokes and project READMEs.

## Promotion Rule

Code moves from a project into `src/` only when **all three** hold:

1. It is a mechanism, not a project policy.
2. At least two descendants want the same thing.
3. The generalized API is simpler than keeping the duplication in project code.

This rule is the main defense against turning the kernel into a renamed
collection of branches.

Recent promotions of the right kind:

- `LinearMemorySubstrate`
- `FrozenReadoutExpert`
- `PredictiveSurpriseController`
- `HormoneModulator`
- `SampledMultiscaleReadout`
- `TrainModeConfig`
- `ArtifactMetadata` / `ReplaySpan` / `ArtifactAccounting`
- `select_scored_spans` / `replay_spans_from_scores`
- `ProbabilityDiagnostics` / `probability_diagnostics`
- `ExactContextCache` / `StatisticalBackoffCache`
- `CausalPredictiveAdapter`
- `OracleAnalysisAdapter`
- `BridgeExportAdapter`
- `NoncausalReconstructiveAdapter`
- `TeacherExportAdapter`
- `ArtifactAuditRecord` / `ArtifactAuditSummary`

Still project-local on purpose:

- descendant mixer and residual-repair policy
- interpretation and reporting around oracle comparisons
- rate-distortion weighting, second compression stage, and quantization/export
  policy in the patch-latent example
- project-specific bridge/export policy above the shared probability-to-feature
  transforms
- ancestor-specific predictor head choices

## Package Map

The kernel is easiest to understand by category rather than by filename order.
For the full capability matrix and module pointers, see
[`kernel_matrix.md`](./kernel_matrix.md).

### Foundation

- [`codecs.py`](../src/decepticons/codecs.py)
- [`config.py`](../src/decepticons/config.py)
- [`metrics.py`](../src/decepticons/metrics.py)

### Substrates

- [`reservoir.py`](../src/decepticons/reservoir.py)
- [`delay.py`](../src/decepticons/delay.py)
- [`linear_memory.py`](../src/decepticons/linear_memory.py)
- [`oscillatory_memory.py`](../src/decepticons/oscillatory_memory.py)
- [`mixed_memory.py`](../src/decepticons/mixed_memory.py)
- [`hierarchical.py`](../src/decepticons/hierarchical.py)
- [`substrates.py`](../src/decepticons/substrates.py)
- [`factories.py`](../src/decepticons/factories.py)

### Control and side channels

- [`control.py`](../src/decepticons/control.py)
- [`gating.py`](../src/decepticons/gating.py)
- [`routing.py`](../src/decepticons/routing.py)
- [`modulation.py`](../src/decepticons/modulation.py)
- [`predictive_surprise.py`](../src/decepticons/predictive_surprise.py)

### Memory, latents, and views

- [`bridge_features.py`](../src/decepticons/bridge_features.py)
- [`bidirectional_context.py`](../src/decepticons/bidirectional_context.py)
- [`exact_context.py`](../src/decepticons/exact_context.py)
- [`latents.py`](../src/decepticons/latents.py)
- [`learned_segmentation.py`](../src/decepticons/learned_segmentation.py)
- [`memory_cache.py`](../src/decepticons/memory_cache.py)
- [`ngram_memory.py`](../src/decepticons/ngram_memory.py)
- [`statistical_backoff.py`](../src/decepticons/statistical_backoff.py)
- [`patch_latent_blocks.py`](../src/decepticons/patch_latent_blocks.py)
- [`probability_diagnostics.py`](../src/decepticons/probability_diagnostics.py)
- [`segmenters.py`](../src/decepticons/segmenters.py)
- [`views.py`](../src/decepticons/views.py)
- [`linear_views.py`](../src/decepticons/linear_views.py)
- [`hierarchical_views.py`](../src/decepticons/hierarchical_views.py)
- [`sampled_readout.py`](../src/decepticons/sampled_readout.py)

### Readouts, experts, and runtime

- [`readout.py`](../src/decepticons/readout.py)
- [`readouts.py`](../src/decepticons/readouts.py)
- [`experts.py`](../src/decepticons/experts.py)
- [`runtime.py`](../src/decepticons/runtime.py)
- [`eval.py`](../src/decepticons/eval.py)
- [`span_selection.py`](../src/decepticons/span_selection.py)
- [`train_eval.py`](../src/decepticons/train_eval.py)
- [`train_modes.py`](../src/decepticons/train_modes.py)
- [`artifacts.py`](../src/decepticons/artifacts.py)

### Adapters and presets

- [`adapters.py`](../src/decepticons/adapters.py)
- [`causal_predictive.py`](../src/decepticons/causal_predictive.py)
- [`bridge_export.py`](../src/decepticons/bridge_export.py)
- [`oracle_analysis.py`](../src/decepticons/oracle_analysis.py)
- [`noncausal_reconstructive.py`](../src/decepticons/noncausal_reconstructive.py)
- [`teacher_export.py`](../src/decepticons/teacher_export.py)
- [`model.py`](../src/decepticons/model.py)
- [`presets.py`](../src/decepticons/presets.py)
- [`cli.py`](../src/decepticons/cli.py)

## Causal-Bank Family

The causal-bank family (`causal_bank.py`) is the most actively explored
descendant family. The full configuration surface — input projection schemes,
oscillatory schedules, substrate modes, memory attachment, stacked blocks,
selective scan, readout geometry, byte-to-patch encoding, fast/slow splitting,
polynomial expansion, stability controls — is documented in the
`CausalBankConfig` docstring at
[`src/decepticons/causal_bank.py`](../src/decepticons/causal_bank.py).

Highlights of what it can be configured to do:

- frozen, learnable-decays, learnable-mixing, full learned recurrence, or
  gated-retention substrate modes
- attached n-gram, exact-context, or statistical-backoff memory
- stacked multi-timescale substrate blocks
- selective scan augment with per-head retention or scan
- timescale-banded readouts
- byte-to-patch encoding with autoregressive, MLP-factored, or hybrid decoders
- fast/slow hemisphere split with separate learning rates
- adaptive regularization and training-noise stability controls
