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

The `src/decepticons/` package contains reusable mechanisms only.

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
- `causal/cache_repair/`
  causal descendant that uses the shared cache-view layer directly and keeps only the repair gate local
- `bridge/proxy_features/`
  bridge-style descendant that turns probability streams into causal proxy features
- `bridge/feature_export/`
  bridge-style descendant that packages paired probability streams into a small export/report flow
- `bridge/agreement_export/`
  bridge-style descendant that focuses on agreement and disagreement over paired probability streams
- `bridge/support_export/`
  bridge-style descendant that exports shared cache-view support/order summaries through the shared teacher/export contract
- `noncausal/field_reconstruction/`
  noncausal field reconstruction descendant built from bidirectional context, exact-context memory, and replay accounting
- `noncausal/replay_fields/`
  noncausal descendant that overlays field-shaped spans on the shared replay surface and keeps field policy local
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

### Control And Side Channels

- [`control.py`](../src/decepticons/control.py)
- [`gating.py`](../src/decepticons/gating.py)
- [`routing.py`](../src/decepticons/routing.py)
- [`modulation.py`](../src/decepticons/modulation.py)
- [`predictive_surprise.py`](../src/decepticons/predictive_surprise.py)

### Memory, Latents, And Views

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

### Readouts, Experts, And Runtime

- [`readout.py`](../src/decepticons/readout.py)
- [`readouts.py`](../src/decepticons/readouts.py)
- [`experts.py`](../src/decepticons/experts.py)
- [`runtime.py`](../src/decepticons/runtime.py)
- [`eval.py`](../src/decepticons/eval.py)
- [`span_selection.py`](../src/decepticons/span_selection.py)
- [`train_eval.py`](../src/decepticons/train_eval.py)
- [`train_modes.py`](../src/decepticons/train_modes.py)
- [`artifacts.py`](../src/decepticons/artifacts.py)

### Adapters And Presets

- [`adapters.py`](../src/decepticons/adapters.py)
- [`causal_predictive.py`](../src/decepticons/causal_predictive.py)
- [`bridge_export.py`](../src/decepticons/bridge_export.py)
- [`oracle_analysis.py`](../src/decepticons/oracle_analysis.py)
- [`model.py`](../src/decepticons/model.py)
- [`presets.py`](../src/decepticons/presets.py)
- [`cli.py`](../src/decepticons/cli.py)

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
- `ExactContextCache` / `StatisticalBackoffCache`

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

## Causal-Bank Configuration Surface

The causal-bank family (`causal_bank.py`) is the most actively explored
descendant family. Its configuration surface includes:

### Input Projection Schemes (`input_proj_scheme`)

- `random` — default, Gaussian scaled by 1/sqrt(embedding_dim)
- `orthogonal_rows` — QR-factorized orthogonal basis
- `split_banks` — separate subspaces for oscillatory and non-oscillatory modes
  - `_orthogonal_rows_in_proj` shape fix applied for high `osc_frac` values
- `kernel_energy` — energy-weighted by mode RMS

### Oscillatory Scheduling (`oscillatory_schedule`)

- `logspace` — simple logarithmic spacing of periods and half-lives
- `mincorr_greedy` — greedy selection minimizing pairwise correlation among candidate pairs
- `period_bucket_greedy` — bucketed half-lives with greedy period selection within each bucket

### Substrate Modes (`substrate_mode`)

- `frozen` — fixed random projection, no gradient through substrate
- `learnable_decays` — decay rates are gradient-tracked parameters
- `learnable_mixing` — mixing weights are gradient-tracked parameters
- `learned_recurrence` — full selective scan with Mamba-style input-dependent B/C projections; chunked parallel scan implementation
- `gated_retention` — learned multi-head matrix memory becomes the primary substrate instead of an additive augment

### Memory Attachment (`memory_kind` / `MemoryAttachmentConfig`)

- `none` — no auxiliary memory
- `ngram` — n-gram prior lookup
- `exact_context` — exact context cache
- `statistical_backoff` — backoff hierarchy over n-gram and exact-context layers

`OnlineCausalMemory` is a runtime n-gram accumulator with a 7-feature query
interface that updates incrementally during training without a separate build step.

### Stacked Substrate Blocks

- `num_blocks` — number of stacked substrate blocks (default 1)
- `block_mixing_ratio` — bottleneck mixing fraction between blocks
- `block_stride` — multi-timescale striding: block `i` operates at stride `block_stride^i`

### Selective Scan Dimensions

- `state_dim` — inner recurrent state width for the learned-state path
- `state_impl` — learned-state implementation (`scan` or `retention`)
- `num_heads` — number of learned-state heads
- under `gated_retention`, this state surface is promoted from an augment to the primary substrate

### Readout Geometry

- `readout_bands` — split readout by timescale band to reduce interference across the fixed/learned bank mixture

### Byte-to-Patch Encoding

- `patch_size` — number of raw bytes per patch (1 = no patching)
- `patch_causal_decoder` — decoder applied after patch embedding:
  - `none` — no patch decoder
  - `autoregressive` — left-to-right patch token prediction
  - `mlp_factored` — independent per-byte MLP heads
  - `hybrid` — global SSM over patches plus local window over raw bytes

### Fast/Slow State Splitting

- `num_hemispheres` — number of hemispheres (2 = fast + slow)
- `fast_hemisphere_ratio` — fraction of state allocated to the fast hemisphere
- `fast_lr_mult` — learning rate multiplier applied to fast-hemisphere parameters

### Polynomial Feature Expansion

- `local_poly_order` — NVAR polynomial expansion order on local window features
- `substrate_poly_order` — polynomial expansion order on substrate output before readout

### Stability Controls

- `training_noise` — Gaussian noise injected into substrate state during forward
- `adaptive_reg` — automatically scales regularization based on live gradient statistics
- Decay regularization term added to training loss

### Readout Kinds

The `CAUSAL_BANK_READOUT_KINDS` registry now includes `gru` (recurrent GRU readout)
in addition to the existing linear and MLP variants.

### Validation Helpers

- `learnable_substrate_keys()` — returns the set of config keys that are
  gradient-tracked under the given `substrate_mode`
- Period and half-life range validation with descriptive error messages on
  out-of-range values

### Downstream Threading

When a new config knob is added to `CausalBankConfig`, the Chronohorn training
CLI must wire it through its scan system (`_training_spec()` and
`_torch_train_command()`). Chronohorn maintains a consistency test that validates
this wiring.

## Immediate Architectural Direction

The current pressure from the descendant runtime is narrower and more concrete:

- make learned memory do more of the real work
- keep the path O(n) in sequence length
- separate readout wins from actual substrate wins
- only promote mechanisms that survive scale and longer context

For the active causal-bank line that means:

- `readout_bands` to reduce timescale interference at readout
- `scan` and `retention` as head-factored learned-state paths
- `gated_retention` as the first primary learned substrate replacing the fixed bank on the cheap lane
- keeping fleet policy, promotion logic, and benchmark claims in `chronohorn`, not here

The older shared-contract work still matters, but the immediate kernel question is simpler:
can a reusable learned-memory substrate beat the fixed-bank ceiling without giving up O(n) execution?
