# Kernel Matrix

What lives in the kernel today, organized by capability area. Use this as a
quick map of what the kernel offers without reading every module.

For the design language behind each capability, see
[`related_work.md`](./related_work.md). For the boundary rule that controls what
gets in, see [`architecture.md`](./architecture.md).

## Capability Matrix

| Area | What it provides | Module |
| --- | --- | --- |
| `substrates.echo_state` | fixed recurrent baseline substrate | `reservoir.py`, `substrates.py` |
| `substrates.delay` | deterministic fading-memory delay line | `delay.py`, `substrates.py` |
| `substrates.linear_memory` | frozen multiscale decay-bank memory | `linear_memory.py`, `substrates.py` |
| `substrates.mixed_memory` | recurrent + delay hybrid | `mixed_memory.py`, `substrates.py` |
| `substrates.hierarchical` | fast/mid/slow multi-timescale substrate | `hierarchical.py`, `substrates.py` |
| `substrates.oscillatory_memory` | exponential + damped-oscillatory mode bank | `oscillatory_memory.py`, `substrates.py` |
| `factories.substrates` | config-driven substrate construction | `factories.py` |
| `controllers.summary` | summary contract shared by gates and routing | `control.py` |
| `controllers.predictive` | latent commit, prediction, surprise, residual | `controllers.py`, `predictive_surprise.py` |
| `controllers.learned_segmentation` | learned boundary probability + target-rate patching | `learned_segmentation.py` |
| `controllers.gating` | fast→mid and mid→slow pathway gates | `gating.py` |
| `controllers.routing` | causal substrate / path selection over branch summaries | `routing.py` |
| `controllers.modulation` | hormone / modulation side channels | `modulation.py` |
| `memory.exact_context` | exact-history experts (exact1/2/3) | `exact_context.py` |
| `memory.ngram` | smoothed n-gram statistical tables | `ngram_memory.py` |
| `memory.statistical_backoff` | fitted backoff mixture with prefix-time fallback | `statistical_backoff.py` |
| `memory.cache_views` | unified prediction records over exact + backoff | `memory_cache.py` |
| `memory.online_ngram` | runtime n-gram accumulator with 7-feature query | `online_memory.py` |
| `views.hierarchical` | pooled and predictive views over fast/mid/slow banks | `hierarchical_views.py` |
| `views.linear_memory` | features over decay-bank memory state | `linear_views.py` |
| `views.byte_latent` | residual + patch-summary + latent feature view | `views.py` |
| `views.sampled_readout` | deterministic sampled bands over multiscale state | `sampled_readout.py` |
| `views.probability_diagnostics` | summaries over one or two probability sources | `probability_diagnostics.py` |
| `views.bridge_features` | probability-to-feature bridge | `bridge_features.py` |
| `analysis.bidirectional_context` | noncausal left/right context determinism probe | `bidirectional_context.py` |
| `runtime.span_selection` | scored-position to contiguous-span grouping | `span_selection.py` |
| `blocks.patch_latent_local` | local byte encoder, patch pooler, global-to-local bridge | `patch_latent_blocks.py` |
| `readouts.closed_form` | trained ridge readout over frozen state | `readouts.py` |
| `experts.frozen_readout` | frozen substrate + feature-function expert | `experts.py` |
| `runtime.trace_reporting` | sequence traces, fit reports, accounting | `runtime.py`, `artifacts.py` |
| `runtime.eval_light` | next-step and rollout scoring | `eval.py` |
| `runtime.train_eval` | weighted dataset eval, rollout curves, transfer probes | `train_eval.py` |
| `runtime.train_modes` | detached vs through-state semantics, sparse slow updates | `train_modes.py` |
| `runtime.artifacts_audits` | legality / replay / artifact-boundary helpers | `artifacts_audits.py` |
| `adapters.causal_predictive` | causal predictive / compressive runtime adapter | `causal_predictive.py` |
| `adapters.noncausal_reconstructive` | document-field replay adapter | `noncausal_reconstructive.py` |
| `adapters.oracle_analysis` | bidirectional structure analysis adapter | `oracle_analysis.py` |
| `adapters.bridge_export` | offline-teacher to causal-export adapter | `bridge_export.py` |
| `adapters.teacher_export` | paired teacher/student export over shared diagnostics | `teacher_export.py` |
| `adapters.byte_latent` | byte-patch latent adapter | `model.py`, `adapters.py` |
| `causal_bank` | family metadata, deterministic substrate construction | `causal_bank.py` |
| `presets` | reproducible named bundles over primitives | `presets.py` |

## Promotion Rule

Code is promoted into `src/` only when all three hold:

1. It is a mechanism, not a project policy.
2. At least two descendants want the same thing.
3. The generalized API is simpler than keeping the duplication.

This is the main defense against turning the kernel into a renamed collection
of branches.

## What's Out of Scope

The kernel deliberately does not own:

- one descendant's training recipe
- one descendant's artifact format
- one descendant's leaderboard or frontier story
- one descendant's legality or audit policy
- one descendant's fleet/runtime orchestration

If a mechanism can be named without reference to a specific descendant and used
unchanged by more than one downstream system, it belongs here. Otherwise it
stays in the descendant.
