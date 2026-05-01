# Downstream Patterns

Five problem shapes the kernel is designed to support. Each names what
descendant systems in the family typically need from a shared library, and
which lessons keep repeating across them.

## Pattern: Causal Predictive / Compressive Systems

Problem shape:

- prefix-only runtime
- next-byte or next-token scoring
- strict legality constraints
- real artifact-boundary accounting

What these systems need from the kernel:

- causal state updates
- score-before-update discipline
- exact-history residual hooks
- packed-memory or cache interfaces
- serializers that distinguish regenerated substrate from true payload
- audit hooks for trainable-vs-frozen structure

Recurring lessons:

- bridge metrics are for search, not claims
- full held-out fresh-process evaluation is mandatory
- packed artifacts, not helper estimates, define the real runtime object
- invalid branches still teach if they are documented honestly
- exact experts work better as residual correction than as full-distribution competitors

## Pattern: Noncausal Reconstructive Systems

Problem shape:

- treat the document as a field, not only a stream
- remove bytes only when they can be reconstructed
- replay removal rounds in reverse
- count side-data economics honestly

What these systems need from the kernel:

- noncausal context access
- patch or field selectors
- round controllers
- sparse and dense mask encoders
- dictionary packers and factoring modes
- exact replay validators
- break-even accounting for side data

Recurring lessons:

- side-data cost can dominate even when removal fractions look strong
- adaptive dense-vs-sparse payload formats matter
- dictionary factoring across rounds can be as important as position payloads
- candidate generation and typed pruning should be explicit subsystems

## Pattern: Oracle Analysis Systems

Problem shape:

- analysis-only pass
- bidirectional local context
- measure structural determinism, not deploy a runtime codec

What these systems need from the kernel:

- bidirectional neighborhood extraction
- leave-one-out corpus maps
- candidate-set statistics
- self-inclusion and future-context attack utilities
- rulebook-cost lower-bound estimates

Recurring lessons:

- exact uniqueness is often the wrong target
- small candidate-set size is usually the stronger oracle label
- future-context uplift can dominate the apparent signal
- raw removable fraction is not a codec claim

## Pattern: Bridge Export Systems

Problem shape:

- boundary layer between noncausal discovery and causal runtime
- offline teacher data on one side
- strictly causal exported features on the other

What these systems need from the kernel:

- feature schemas for causal exports
- offline teacher-label serialization
- replay and audit adapters
- explicit boundary contracts between analysis and runtime modules

Recurring lessons:

- oracle outputs should be treated as offline teacher or probe data
- runtime consumers must not depend on live right-context scoring
- the bridge should be a first-class subsystem, not an implicit handoff

## Pattern: Byte-Latent Systems

Problem shape:

- bytes remain the visible interface
- shorter internal latent stream carries local summaries
- recurrent or reservoir-style state integrates those summaries over time

What these systems need from the kernel:

- patchers and segmenters
- latent commit policies
- fixed or semi-fixed substrate builders
- local byte decoders
- latent-stream metrics
- optional quantization and export hooks

The current `ByteLatentPredictiveCoder` is the reference byte-latent adapter in
this repo.

## Cross-Cutting Constraints

Across all five patterns, the same constraints keep repeating:

- separate model legality from artifact-boundary legality
- keep search metrics separate from claim metrics
- evaluate the real packed object, not only a training checkpoint
- document negative results instead of deleting them from the story
- treat invalid branches as falsifiers and teachers, not as frontier claims
- make the runtime contract explicit whenever offline oracle analysis exists

These are not just repo habits — they are the shape of a serious reusable
library surface for this family of work.

## How These Map to the Kernel

| Pattern | Kernel adapter |
| --- | --- |
| causal predictive | [`CausalPredictiveAdapter`](../src/decepticons/causal_predictive.py) |
| noncausal reconstructive | [`NoncausalReconstructiveAdapter`](../src/decepticons/noncausal_reconstructive.py) |
| oracle analysis | [`OracleAnalysisAdapter`](../src/decepticons/oracle_analysis.py) |
| bridge export | [`BridgeExportAdapter`](../src/decepticons/bridge_export.py) |
| paired teacher/export | [`TeacherExportAdapter`](../src/decepticons/teacher_export.py) |
| byte-latent | [`ByteLatentPredictiveCoder`](../src/decepticons/model.py) |
