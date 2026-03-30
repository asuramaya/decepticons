# Ecosystem Landscape

This snapshot was assembled on March 29, 2026 to answer a narrow question:

- does a healthy open library already exist for predictive coding over byte sequences with reservoir-style latent memory?

The short answer is no. The ecosystem is real, but split.

## Summary

The nearby space breaks into three groups:

- active predictive-coding toolkits that focus on general PCNs
- mature reservoir libraries that do not target predictive coding
- thin, partial, or stalled projects that cover pieces of the idea but not the full surface

The gap this repo targets is the overlap:

- predictive coding
- sequence or byte modeling
- adaptive patching or compressed latent summaries
- frozen recurrent or reservoir-style memory
- a small usable Python package surface

## Active Predictive-Coding Libraries

### [PCX](https://github.com/liukidar/pcx) and [PyPI `pcx`](https://pypi.org/project/pcx/)

Observed state:

- GitHub repo pushed on March 28, 2026
- PyPI package `0.6.3` uploaded on March 26, 2025
- JAX-based, documented, packaged, and clearly alive

What it already does well:

- serious predictive-coding package surface
- configurable PCNs
- documentation, examples, and packaging

What it does not appear to target:

- byte-level sequence compression
- reservoir substrates
- adaptive patch commits over raw byte streams

Takeaway:

- this is the strongest adjacent project, but it occupies the "general JAX predictive coding library" slot

### [pcn-torch](https://github.com/emv-dev/pcn-torch) and [PyPI `pcn-torch`](https://pypi.org/project/pcn-torch/)

Observed state:

- GitHub repo created on February 20, 2026
- PyPI `1.1.0` uploaded on February 27, 2026
- has `src/`, `tests/`, `examples`, and a cleaner packaging surface than many PCN repos

What it already does well:

- practical PyTorch-native PCN implementation
- readable API for layered predictive coding
- packaging and testing discipline

What looks incomplete or narrow:

- public scope is still mostly supervised PCNs
- examples and framing center classification rather than sequence modeling
- no reservoir or compression-first surface

Takeaway:

- active and useful, but not the same product shape as this repo

### [Pyromancy](https://github.com/mdominijanni/pyromancy) and [PyPI `pyromancy-ai`](https://pypi.org/project/pyromancy-ai/)

Observed state:

- PyPI `0.0.1` uploaded on June 24, 2025
- GitHub last pushed on October 1, 2025
- explicitly described as a compact predictive-coding library

What it already does well:

- clean positioning
- docs and tests
- modern package layout

What still looks early:

- alpha-stage package
- narrow surface area
- no visible sequence, byte, reservoir, or compression-first wedge

Takeaway:

- promising foundation, but still early and aimed at core PCN building blocks

### [PRECO](https://github.com/bjornvz/PRECO) and [PyPI `preco`](https://pypi.org/project/preco/)

Observed state:

- GitHub pushed on March 13, 2026
- GitHub frames it as a library for PCNs and predictive coding graphs
- PyPI `preco` exists, but the published package is still `0.0.0` with the summary "Just a placeholder for now"

What it already does well:

- active research-facing repository
- ties directly to survey/tutorial work

What still looks incomplete:

- package distribution surface lags the repo
- public repo layout still reads more like package-plus-notebooks than a broad stable platform
- no reservoir or byte-sequence focus

Takeaway:

- alive as a research codebase, but incomplete as a polished distribution surface

## Thin Or Possibly Stalled Predictive-Coding Projects

### [PCLib](https://github.com/joeagriffith/pclib) and [PyPI `pclib`](https://pypi.org/project/pclib/)

Observed state:

- GitHub last pushed on April 24, 2024
- PyPI latest is `2.0.0b2` from March 6, 2024
- PyPI still labels it beta

What it already does:

- torch-like API for building predictive coding networks
- fully connected and convolutional variants

What looks incomplete:

- package activity appears to have stopped after early beta releases
- PyPI description says the CNN helper is not customizable in shape
- repo root contains committed build outputs and packaging artifacts like `build/`, `dist/`, `pclib.egg-info`, and `None.pth`

Takeaway:

- useful signal that people want this kind of library, but it does not look like a maintained, production-grade package surface

### [pypc](https://github.com/infer-actively/pypc)

Observed state:

- GitHub last pushed on March 30, 2024
- very small root layout and minimal README
- no obvious packaging file in the repo root from the inspected contents

What it already does:

- lightweight predictive-coding implementation in PyTorch

What looks incomplete or abandoned:

- minimal docs
- minimal package surface
- unclear installation/distribution story

Important naming note:

- [PyPI `pypc`](https://pypi.org/project/pypc/) is unrelated and is an old 2015 package called "Python3 Package Creator"

Takeaway:

- interesting code artifact, but not a strong living package surface

## Active Reservoir Libraries

### [ReservoirPy](https://github.com/reservoirpy/reservoirpy) and [PyPI `reservoirpy`](https://pypi.org/project/reservoirpy/)

Observed state:

- GitHub pushed on March 26, 2026
- large project, active discussions, docs, examples, tutorials, and packaging

What it already does well:

- mature reservoir-computing ecosystem
- good package hygiene
- broad ESN-oriented tooling

What it does not target:

- predictive coding
- byte-level latent compression
- adaptive patch-based sequence abstractions

Takeaway:

- clearly alive, but solves a different problem

### [PyRCN](https://github.com/PlasmaControl/PyRCN) and [PyPI `pyrcn`](https://pypi.org/project/pyrcn/)

Observed state:

- GitHub pushed on July 17, 2024
- PyPI `0.0.18` uploaded on July 16, 2024
- repo and package are real and usable

What it already does well:

- scikit-learn-compatible reservoir API
- concrete task support for ESNs and ELMs

What still looks incomplete:

- package still classifies itself as pre-alpha
- roadmap text still points to future work rather than a finished broader platform
- no predictive-coding layer

Takeaway:

- a healthy reservoir library, but not a predictive-coding one

## Stalled Reservoir Surfaces

### [easyesn](https://github.com/kalekiu/easyesn) and [PyPI `easyesn`](https://pypi.org/project/easyesn/)

Observed state:

- GitHub last pushed on January 4, 2021
- PyPI latest `0.1.6.1` uploaded on January 4, 2021
- 11 open GitHub issues at inspection time

What it already does:

- easy-to-use ESN interface
- hyperparameter tuning story

What looks stalled:

- no recent development activity
- package text still notes that the gradient optimizer does not fully work

Takeaway:

- still useful historically, but not the place to build a new hybrid predictive-coding surface on top of

## What Seems Missing

Across the projects above, the missing combined surface is:

- a package that treats bytes or generic sequences as the main visible interface
- predictive-coding style residual features rather than only generic PCN hierarchies
- adaptive patching or latent commits that shorten the internal sequence
- reservoir-style fixed recurrent memory instead of only trainable latent stacks
- a small readable reference implementation with package docs, tests, and CLI

That is the wedge for `open-predictive-coder`.

## Positioning For This Repo

The most accurate framing is not:

- "a general predictive coding framework"

The more defensible framing is:

- "a small reference library for byte-level predictive coding with reservoir latents"
- "a sequence-compression-oriented predictive coding toolkit"
- "an open bridge between predictive coding, reservoir computing, and adaptive byte patching"

The broader pattern language extracted from the sibling projects is documented in
[`docs/downstream_patterns.md`](./downstream_patterns.md). That document deliberately uses idea-based names like
`causal`, `noncausal`, `oracle`, `bridge`, and `byte-latent` rather than project codenames.

## Local Sibling Patterns

Within the current workspace, there are already sibling projects that illustrate the sort of work this library is for.
They should be treated as example downstream patterns, not as code ancestors for this repo.
See [`lineage.md`](./lineage.md) for the attribution rule and why these are named here without dead local links.

### `giddy-up`

- generalized pattern: `bridge.export`
- boundary layer between oracle-side discovery and causal exported features
- good example of a project that needs predictive summaries and strict boundary discipline

### `blinx`

- generalized patterns: `noncausal.field.reconstructive` and `oracle.analysis`
- noncausal lossless-compression research over whole-document reconstruction
- good example of the "document as field, not only stream" side of the design space

### `conker`

- generalized pattern: `causal.byte.compressive`
- causal compression/runtime research with memory-first experiments
- good example of the stricter runtime and artifact-boundary side of the design space

The intended interpretation is:

- `open-predictive-coder` is a reusable library layer for work in this family
- those repos are examples of downstream pattern types the library is meant to serve
- the current implementation only covers the `byte-latent` reference path directly
- this repo is not presented as an adaptation of those projects even though it learns from their documentation

## Naming Notes

The exact surface `open-predictive-coder` did not show up as an existing obvious GitHub or PyPI collision during this pass.
However, shorter names are already crowded or ambiguous:

- `pypc` is taken on PyPI by an unrelated package
- `preco` exists on PyPI but is currently just a placeholder surface
- `pcx` is already an active predictive-coding package

That makes the more explicit repo/package name a better choice.
