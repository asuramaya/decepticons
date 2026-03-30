# Downstream Patterns

This document extracts reusable library shapes from related descendant projects, but renames them by idea instead of
by codename. The goal is to keep `open-predictive-coder` legible as a library surface even if the surrounding project
names change.

Use [`lineage.md`](./lineage.md) for the attribution rule behind the non-vendored sources named here.

## Naming Rule

The public library should organize itself along a few explicit axes:

- `causal`, `noncausal`, `oracle`, `bridge`
- `byte`, `token`, `patch`, `field`, `sequence`
- `predictive`, `reconstructive`, `compressive`, `export`, `probe`
- `reservoir`, `delay`, `mixed_memory`, `routed`, `modulated`

That vocabulary says what a module is for without requiring knowledge of internal project history.

## Core Kernel

The upstream experiment family suggests that the reusable center is not a single branch outcome, but the upstream core
from which several descendants grow. The kernel is therefore made of:

- substrate dynamics
- learned side-channels
- state views and sampled summaries
- readouts
- runtime and audit utilities

The library should therefore present a reusable kernel first, with downstream adapters layered on top of it. In this
framing, named descendants are not peers of the kernel. They are evidence about where the line should be drawn.

## Pattern: Causal Predictive Or Compressive Systems

Primary descendant sources:

- causal runtime frontier notes from the broader workspace
- causal validity notes from the broader workspace
- causal negative-results notes from the broader workspace

Problem shape:

- prefix-only runtime
- next-byte or next-token scoring
- strict legality constraints
- real artifact-boundary accounting

What these systems need from the library:

- causal state updates
- score-before-update discipline
- exact-history residual hooks
- packed-memory or cache interfaces
- serializers that distinguish regenerated substrate from true payload
- audit hooks for trainable-vs-frozen structure

Key lessons from the docs:

- bridge metrics are for search, not claims
- full held-out fresh-process evaluation is mandatory
- packed artifacts, not helper estimates, define the real runtime object
- invalid branches can still teach useful lessons if they are documented honestly
- exact experts work better as residual correction over a strong base than as full-distribution competitors
- memory-first restarts are a legitimate downstream direction even when the first packed-memory build is weak

## Pattern: Noncausal Reconstructive Systems

Primary descendant sources:

- noncausal reconstruction notes from the broader workspace
- noncausal replay/accounting notes from the broader workspace

Problem shape:

- treat the document as a field, not only a stream
- remove bytes only when they can be reconstructed
- replay removal rounds in reverse
- count side-data economics honestly

What these systems need from the library:

- noncausal context access
- patch or field selectors
- round controllers
- sparse and dense mask encoders
- dictionary packers and factoring modes
- exact replay validators
- break-even accounting for side data

Key lessons from the docs:

- side-data cost can dominate even when removal fractions look strong
- adaptive dense-vs-sparse payload formats matter
- dictionary factoring across rounds can be as important as position payloads
- candidate generation and typed pruning should be explicit subsystems, not afterthoughts

## Pattern: Oracle Analysis Systems

Primary descendant source:

- oracle-analysis notes from the broader workspace

Problem shape:

- analysis-only pass
- bidirectional local context
- measure structural determinism, not deploy a runtime codec

What these systems need from the library:

- bidirectional neighborhood extraction
- leave-one-out corpus maps
- candidate-set statistics
- self-inclusion and future-context attack utilities
- rulebook-cost lower-bound estimates

Key lessons from the docs:

- exact uniqueness is often the wrong target
- small candidate-set size is usually the stronger oracle label
- future-context uplift can dominate the apparent signal
- raw removable fraction is not a codec claim

## Pattern: Bridge Export Systems

Primary descendant sources:

- bridge/export notes from the broader workspace
- bridge architecture notes from the broader workspace

Problem shape:

- boundary layer between noncausal discovery and causal runtime
- offline teacher data on one side
- strictly causal exported features on the other

What these systems need from the library:

- feature schemas for causal exports
- offline teacher-label serialization
- replay and audit adapters
- explicit boundary contracts between analysis and runtime modules

Key lessons from the docs:

- oracle outputs should be treated as offline teacher or probe data
- runtime consumers must not depend on live right-context scoring
- the bridge should be a first-class subsystem rather than an implicit handoff

## Pattern: Byte-Latent Systems

Primary sources:

- this repo's current implementation
- the broader upstream extraction direction
- public byte-latent downstream repos as adaptations in the same family

Problem shape:

- bytes remain the visible interface
- shorter internal latent stream carries local summaries
- recurrent or reservoir-style state integrates those summaries over time

What these systems need from the library:

- patchers and segmenters
- latent commit policies
- fixed or semi-fixed substrate builders
- local byte decoders
- latent-stream metrics
- optional quantization and export hooks

Key lessons for the current repo:

- the current `OpenPredictiveCoder` model is a reasonable reference adapter
- it should not be the long-term center of the library if the goal is to extract the broader kernel
- the durable public language is `byte`, `patch`, `latent`, `causal`, `oracle`, and `bridge`, not local branch names

## Cross-Cutting Constraints

Across the sibling docs, a few reusable constraints keep repeating:

- separate model legality from artifact-boundary legality
- keep search metrics separate from claim metrics
- evaluate the real packed object, not only a training checkpoint
- document negative results instead of deleting them from the story
- treat invalid branches as falsifiers and teachers, not as frontier claims
- make the runtime contract explicit whenever offline oracle analysis exists

These are not just repo-specific habits. They are the shape of a serious reusable library surface for this family of
work.

## Suggested Public Module Names

If the library is generalized, the obvious public surfaces are:

- `substrates`
- `controllers`
- `views`
- `readouts`
- `runtime`
- `adapters`
- `presets`

Likely adapter names:

- `causal_predictive`
- `noncausal_reconstructive`
- `oracle_analysis`
- `bridge_export`
- `byte_latent`

## Lineage Note

Named descendants still exist in the broader workspace and in the repo lineage note. They are useful for attribution,
but they should stay secondary to the idea-based library surface.
