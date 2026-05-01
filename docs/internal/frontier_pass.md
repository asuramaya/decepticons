# Frontier Pass

This document captures the current frontier pass against the live sibling descendants:

- `conker`
- `blinx`
- `giddy-up`

The point is not to copy those repos into `decepticons`. The point is to read the current frontier and ask:

1. what is still descendant-only policy
2. what is now repeated enough to be a candidate kernel mechanism
3. what should first be rebuilt here as another example descendant before promotion

Use [`lineage.md`](./lineage.md) for the attribution rule behind the non-vendored repo names in this pass.

## Current Frontier Read

### Causal Frontier

The live `Conker` frontier is memory-first and controller-heavy.

Current branch read:

- `Conker-10`
  - packed unigram / bigram / hashed trigram memory
  - normalized posterior backoff
  - learned mixer with the neural base
  - explicit causal structure proxy features
- `Conker-11`
  - recursive routing over legal lag buckets and residual source ownership
- `Conker-12`
  - compact higher-order controller tensor over lag, source, program slot, and effect
- `Conker-13`
  - higher-order controller over lag groups, linear mode groups, local offsets, program slot, and effect

So the causal frontier is currently pushing on:

- packed statistical memory
- confidence-aware base-vs-memory mixing
- recursive routing
- higher-order controller tensors
- direct control over frozen linear-mode groups and local residual windows

### Noncausal Frontier

The live `BLINX` frontier is payload economics.

Current branch read:

- `BLINX-5`
  - adaptive dense-vs-sparse position encoding
- `BLINX-6`
  - direct-vs-shared dictionary factoring across rounds
- `BLINX-7`
  - typed pruning over globally decodable context dictionaries

So the noncausal frontier is currently pushing on:

- position encoding choice
- dictionary layout choice
- reusable cross-round payload factoring
- typed candidate pruning
- side-data break-even accounting

### Bridge Frontier

The live `Giddy-Up` frontier is the legality boundary:

- oracle analysis
- attack / uplift accounting
- strictly causal proxy features
- export surfaces that causal runtime can consume legally

So the bridge frontier is currently pushing on:

- teacher-label export
- attack-aware oracle reports
- causal proxy feature schemas
- explicit separation between offline oracle and runtime features

## What OPC Already Covers

The current kernel and descendant examples already cover a meaningful part of that frontier:

- packed statistical memory in simplified form:
  - [`ngram_memory.py`](../src/decepticons/ngram_memory.py)
- exact support-aware causal repair:
  - [`exact_context.py`](../src/decepticons/exact_context.py)
- bridge proxy features and export:
  - [`bridge_features.py`](../src/decepticons/bridge_features.py)
  - [`bridge_export.py`](../src/decepticons/bridge_export.py)
- oracle-side bidirectional support probing:
  - [`bidirectional_context.py`](../src/decepticons/bidirectional_context.py)
- score-to-span reuse:
  - [`span_selection.py`](../src/decepticons/span_selection.py)
- descendant rebuilds:
  - `examples/projects/causal/*`
  - `examples/projects/causal/packed_memory_controller`
  - `examples/projects/bridge/*`
  - `examples/projects/noncausal/field_reconstruction`

So the repo is no longer missing the whole frontier. It is missing the next sharper cuts of it.

## What Still Looks Kernel-Worthy

These are the strongest current candidates.

### 1. Packed Statistical Memory Refinement

Current `ngram_memory` is the right direction, but the live causal frontier is pushing a narrower stronger shape:

- explicit posterior backoff chain
- hashed higher-order bucketed memory
- confidence / mass / support features from those tables

Likely extraction path:

- rebuild a stronger causal statistical-memory example first
- only then decide whether `ngram_memory` should grow into a richer packed-memory kernel module

### 2. Cache Or Prior Confidence Features

The `Conker-10` read makes this explicit: memory is not enough by itself; what matters is exposing when memory should be trusted.

This seam is now in `src/` as a small family-neutral probability diagnostics surface.

Kernel seam:

- a small probability-source diagnostics surface that reports things like:
  - entropy
  - peak
  - top-k mass
  - base-vs-memory agreement

This is more plausible now because both bridge export and causal memory-first work want similar “confidence about a scored distribution” surfaces.

It is family-neutral, mechanism-level, and already repeated across:

- bridge feature export
- causal packed-memory control
- oracle-side probability comparison

What should not follow it into `src/`:

- payload-wire policy
- teacher-export policy
- higher-order causal program/controller policy

### 3. Payload-Choice Helpers

The `BLINX-5/6/7` line suggests the noncausal frontier is really about choosing among payload formats, not only discovering candidates.

Kernel candidate, but only after another noncausal example:

- location encoding choice helper
- dictionary layout choice helper
- typed pruning helper

These are not yet ready for `src/` today. They should first appear in at least one more noncausal descendant here.

### 4. Teacher Export And Attack Reports

The bridge frontier is now broader than just proxy features:

- export records
- leave-one-out / future-context uplift
- rulebook-cost lower bounds

The current oracle and bridge surfaces are close, but they do not yet expose a generalized export-label contract for downstream training or auditing.

## What Still Belongs In Examples

These are still too policy-shaped to promote now:

- `Conker-11/12/13` higher-order controller tensors
- direct routing policy over residual source families
- direct routing policy over linear mode groups and local offsets
- `BLINX-5/6/7` exact wire formats
- exact dictionary packing rules
- exact attack score composition

Those should first exist as rebuilt descendants or example-local controllers in this repo.

## Best Next Rebuilds

If the repo keeps following the descendant-first method, the next clean rebuilds are:

1. `examples/projects/noncausal/payload_choice`
   - noncausal descendant that chooses dense vs sparse position payloads
   - keep dictionary policy local at first

2. `examples/projects/bridge/teacher_export`
   - bridge descendant focused on export labels and attack-aware reporting

3. a stronger causal controller/program descendant
   - pressure-test higher-order controller shapes without promoting them into `src/`

Those rebuilds would pressure the frontier without locking the wrong `src/` contracts too early.

## Current Extraction Rule

The frontier does not change the basic rule:

- repeated mechanism goes to the kernel
- descendant policy stays in example space
- the current frontier only defines where to look next
- probability diagnostics is now a shared seam and should be pressure-tested, not immediately widened
- higher-order causal controller policy stays out of `src/` for now

That means:

- `Conker` defines the causal imagination limit for now
- `BLINX` defines the noncausal imagination limit for now
- `Giddy-Up` defines the legal bridge/oracle imagination limit for now

`decepticons` should keep rebuilding those shapes as example descendants and only extracting the seams that survive repetition.
