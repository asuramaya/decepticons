# Kernel Matrix

This matrix lays out the kernel-first build order for `open-predictive-coder` using three sources of guidance:

- the upstream workspace thesis that the reusable center is the literal core from which the downstream systems evolve
- the generalized downstream pattern language in [`downstream_patterns.md`](./downstream_patterns.md)
- the research anchors in [`related_work.md`](./related_work.md)

The point is not to preserve every historical branch name. The point is to extract the smallest reusable kernel from
the upstream workspace that can still support causal, noncausal, oracle, bridge, and byte-latent downstream systems.

This matrix is written against the current kernel, including the recent shared-contract wave:

- a family-neutral noncausal adapter
- a paired teacher/export contract above the shared bridge layer
- a generic artifact-boundary audit helper

Use [`lineage.md`](./lineage.md) for the attribution rule behind the upstream workspace paths named below.
Use [`frontier_pass.md`](./frontier_pass.md) for the current descendant-frontier read across the live sibling repos.

## Source Anchors

### Upstream Workspace

- `README.md#L8` in the upstream workspace:
  the working thesis is a rich substrate plus a learned side-channel and readout
- `README.md#L21` in the upstream workspace:
  the live fixed-substrate thread is hierarchical and multi-timescale
- `README.md#L29` in the upstream workspace:
  the harness is meant to compare substrate designs, not just masks
- `ORIGIN.md#L15` in the upstream workspace:
  the original design language was subtraction, predictive coding, and a reservoir plus controller plus readout stack

### Generalized downstream framing

- [`downstream_patterns.md`](./downstream_patterns.md):
  the kernel should be organized as substrate dynamics, side-channels, views, readouts, and runtime
- [`downstream_patterns.md`](./downstream_patterns.md):
  causal systems need legality, replay, and packed-artifact discipline
- [`downstream_patterns.md`](./downstream_patterns.md):
  noncausal systems need field selectors, replay, and side-data accounting
- [`downstream_patterns.md`](./downstream_patterns.md):
  oracle systems need analysis-only context and attack accounting
- [`downstream_patterns.md`](./downstream_patterns.md):
  bridge systems need explicit runtime contracts

### Research anchors

- [`related_work.md`](./related_work.md):
  predictive coding implies local residuals and error-carrying views
- [`related_work.md`](./related_work.md):
  reservoir computing anchors the fixed-substrate side of the kernel
- [`related_work.md`](./related_work.md):
  rate-distortion and information bottleneck thinking justify explicit latent commits
- [`related_work.md`](./related_work.md):
  sequence predictive coding and continual modeling motivate the runtime and rollout surfaces
- [`related_work.md`](./related_work.md):
  recurrent refinement and byte-level latent compression motivate the byte-latent adapter layer

## Matrix

| Kernel Area | Why It Exists | Upstream source | Research anchor | Current status |
| --- | --- | --- | --- | --- |
| `substrates.echo_state` | fixed recurrent baseline substrate | `carving_machine/reservoir.py#L85` | Jaeger 2001, Maass et al. 2002 | Implemented |
| `substrates.delay` | deterministic fading-memory control line | `carving_machine/models.py#L384` | fading-memory / liquid-state intuition | Implemented |
| `substrates.linear_memory` | frozen linear multiscale decay-bank memory | linear-correction and residual-repair reconstruction from primitives | short/medium-horizon linear memory scaffold | Implemented |
| `substrates.mixed_memory` | recurrent plus delay hybrid | `carving_machine/models.py#L484`, `carving_machine/models.py#L609` | sequence memory and continual modeling | Implemented |
| `substrates.hierarchical` | fast/mid/slow multi-timescale substrate | `carving_machine/models.py#L224` | predictive coding hierarchies, multi-timescale memory | Implemented |
| `substrates.oscillatory_memory` | frozen exponential plus damped-oscillatory mode bank | later linear-memory / oscillatory substrate experiments in the upstream workspace | multi-timescale linear dynamical memory and oscillatory state summaries | Implemented |
| `factories.substrates` | config-driven substrate construction and adapter dispatch | upstream harness principle: compare substrate designs | engineering bridge from research kernel to adapters | Implemented |
| `controllers.summary` | generic summary contract shared by gates and routing | `carving_machine/models.py#L224`, `carving_machine/models.py#L1129` | control-side summary views over substrate state | Implemented |
| `controllers.predictive` | latent commit, prediction, surprise, residual paths | `carving_machine/models.py#L224`, `carving_machine/models.py#L609` | Rao and Ballard 1999, Friston 2005 | Partial: generic predictive/surprise primitive implemented |
| `controllers.learned_segmentation` | reusable learned boundary probability and target-rate patching | learned patch-latent reconstruction from primitives | rate-distortion intuition without transformer-specific policy | Implemented |
| `memory.exact_context` | causal exact-history experts over exact1/exact2/exact3 style contexts | causal descendant docs and early exact-count branches | count-based language modeling, causal support-aware correction | Implemented |
| `memory.ngram` | smoothed unigram/bigram/trigram statistical tables | later causal packed-memory descendants | classical n-gram language modeling and lightweight causal memory | Implemented |
| `memory.statistical_backoff` | fitted global mixture over unigram/bigram/trigram priors with prefix-time fallback semantics | generalized packed-memory descendants rebuilt from primitives | memory-first backoff modeling and lightweight prior mixing | Implemented |
| `controllers.gating` | reusable fast-to-mid and mid-to-slow pathway gates | `carving_machine/models.py#L224` | adaptive control over multiscale substrate paths | Implemented |
| `controllers.routing` | causal substrate/path selection over branch summaries | `carving_machine/models.py#L1129` | adaptive control over substrate views | Implemented |
| `controllers.modulation` | hormone/modulation paths | `carving_machine/models.py#L1354` | side-channel modulation over substrate | Implemented as primitive |
| `views.hierarchical` | pooled and predictive views over fast/mid/slow banks | `carving_machine/models.py#L224` | predictive coding hierarchies and surprise-style residuals | Implemented |
| `views.sampled_readout` | deterministic sampled bands over multiscale state | `ablations.py`, `v6.py` | sampled multiscale readout over fixed state | Implemented |
| `views.byte_latent` | residual + patch-summary + latent feature construction | current library | Rao and Ballard 1999, BLT 2025 | Implemented |
| `views.probability_diagnostics` | family-neutral summaries over one or two probability sources | bridge export, causal packed-memory control, and causal program-controller descendants | confidence and overlap diagnostics over scored distributions | Implemented |
| `views.bridge_features` | probability-to-feature bridge between proxy and runtime surfaces | bridge/export descendant framing | causal proxy features from offline or higher-order distributions | Implemented |
| `analysis.bidirectional_context` | noncausal left/right context determinism probe | oracle-analysis descendant framing | bidirectional context probing and support-size analysis | Implemented |
| `runtime.span_selection` | shared score-array to replay-span grouping seam | bridge export and noncausal field-reconstruction descendants | reusable scored-position to contiguous-span selection | Implemented |
| `blocks.patch_latent_local` | local byte encoding, patch pooling, and learned global-to-local bridge | learned patch-latent reconstruction from primitives | local predictive coding over shorter latent streams | Implemented |
| `views.linear_memory` | reusable features over decay-bank memory state | linear-correction and residual-repair reconstruction from primitives | short/medium-horizon linear summaries | Implemented |
| `readouts.closed_form` | simple trained readout over frozen state | current library | Jaeger 2001 | Implemented |
| `experts.frozen_readout` | frozen substrate plus feature-function expert wrapper | causal variant reconstruction from primitives | mixture-of-experts over fixed dynamical state | Implemented |
| `runtime.trace_reporting` | sequence traces, reports, accounting | current library | Shannon 1948 | Implemented |
| `runtime.eval_light` | next-step and single-rollout scoring over current adapters | `carving_machine/training.py#L64` | sequence predictive coding and continual modeling | Implemented |
| `runtime.train_eval` | weighted dataset eval, checkpointed rollout curves, and transfer probes | `carving_machine/training.py#L54`, `carving_machine/training.py#L111`, `carving_machine/experiments.py#L493` | sequence predictive coding and continual modeling | Implemented |
| `runtime.train_modes` | detached vs through-state semantics, sparse slow updates, rollout checkpoints | `bptt_test.py` | BPTT versus detached-state runtime regimes | Implemented |
| `runtime.artifacts_audits` | legality, replay, artifact-boundary surfaces | downstream causal docs | compression/accounting discipline | Implemented as generic audit helpers over artifact accounting; legality policy stays downstream |
| `adapters.byte_latent` | first concrete downstream adapter | current library | BLT 2025, UT 2018 | Implemented |
| `adapters.causal_predictive` | causal predictive/compressive runtime systems | causal descendant docs | sequence predictive coding | Implemented |
| `adapters.noncausal_reconstructive` | document-field replay systems | noncausal descendant docs | reconstruction and side-data economics | Implemented |
| `adapters.oracle_analysis` | bidirectional structure analysis | oracle analysis docs | predictive coding as analysis, not runtime | Implemented |
| `adapters.bridge_export` | offline teacher to causal export layer | bridge-export docs | explicit boundary discipline | Implemented |
| `adapters.teacher_export` | paired teacher/student export records over shared diagnostics | bridge-side teacher-export descendants | explicit offline teacher-label export without attack policy | Implemented |
| `presets` | reproducible named bundles over primitives | `carving_machine/catalog.py#L1` | engineering convenience, not theory | Not started |
| `examples.reference_projects` | thin project-shaped models over the kernel, used for smoke/dev loops | upstream ancestor docs, causal descendant docs, oracle-analysis docs, and byte-latent downstream docs as downstream shapes | engineering bridge from primitives to dev/test targets | Implemented |

## Priority

### P0: finish the remaining kernel spine

- integrate the new modulation and predictive-surprise primitives into more than one descendant before widening them further
- richer exact-context expert families beyond the current exact-history memory core
- full training harness and optimizer-facing loop extraction beyond the current eval/transfer scaffold
- tighter hierarchy/controller integration beyond pooled predictive views and the compact predictive-surprise primitive

### P1: finish the causal kernel contract

- rollout evaluation
- replay-safe reporting
- packed-artifact and legality hooks
- broader memory-first expert/cache interfaces beyond the current statistical backoff layer

### P2: grow sideways into noncausal and bridge work

- noncausal field selectors and replay accounting
- oracle analysis helpers
- bridge/export schemas
- payload-choice helpers and teacher-export surfaces once they repeat across multiple descendants

### P3: only then stabilize presets

- named preset bundles copied from the upstream workspace
- compatibility aliases for historical branch names if still useful

## Current Read

The kernel is no longer just an echo-state toy. It now has:

- fixed recurrent substrate
- delay substrate
- linear memory substrate
- oscillatory memory substrate
- mixed-memory substrate
- hierarchical multiscale substrate
- substrate factory and `substrate_kind` dispatch
- controller-summary contract
- predictive controller primitive plus compact predictive/surprise primitive
- learned boundary-scoring and learned segmenter primitive
- pathway-gate primitive
- summary router primitive
- hormone modulation primitive
- exact-context memory and support-weighted blending primitives
- smoothed n-gram memory primitive
- fitted statistical backoff layer over order-wise n-gram priors
- feature-view primitives for both byte-latent and hierarchical state
- bridge feature utilities over probability arrays
- family-neutral probability diagnostics over probability arrays
- bidirectional context analysis primitive
- scored-span selection primitive
- local byte encoder, patch pooler, and learned global-to-local bridge primitives
- deterministic sampled multiscale readout
- reusable linear-memory feature view
- reporting runtime surface
- lightweight rollout and next-step evaluation
- weighted dataset eval, rollout curves, and transfer probes
- frozen readout expert primitive
- first shared causal adapter
- first shared oracle-analysis adapter
- first shared bridge-export adapter
- byte-latent adapter
- example-project smoke surfaces for the hierarchical ancestor path and early exact-context repair builds
- causal mixture/correction/repair replica projects built from primitives
- bridge proxy-feature and feature-export consumers built from primitives
- a first shared noncausal reconstruction adapter
- a first shared paired teacher/export contract
- a first generic artifact-boundary audit helper over artifact accounting

The line to preserve is:

- if a mechanism is required by multiple descendants, it belongs here
- if a choice is really a downstream task policy, legality rule, artifact boundary, or evaluation claim, it does not
  belong here

That is why the upstream workspace remains the main source anchor even now that `src/` contains both `byte-latent`
and a first `causal_predictive` adapter. The line still matters: descendants can thin around the causal contract
without forcing descendant-specific policy back into the kernel.

That is enough to justify continuing kernel-first rather than inventing downstream codenames too early.
