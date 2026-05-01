# Decepticons — Public Copy Style

This file locks the canonical copy that should appear consistently across
public surfaces (README, pyproject, site, CLAUDE.md, package docstring,
CHANGELOG, social blurbs). Update this file before changing any of those
surfaces, so the canonical block stays the source of truth.

Internal-facing docs under `docs/*.md` have their own established voice and
are not bound by this file.

The sibling runtime project has its own STYLE.md at
[`asuramaya/chronohorn/STYLE.md`](https://github.com/asuramaya/chronohorn/blob/main/STYLE.md);
keep cross-project phrasing aligned (sibling-project framing, ecosystem
diagram, dependency rule).

## Slogan

> O(n) attention is deception.

The hero slogan. Always written **`O(n)`** — never `O(1)`, never `O(n log n)`.
Period at the end is part of the line.

## Tagline (one line)

> A backend-neutral kernel of predictive primitives — substrates, memory, gating, routing, readouts.

This is the one-liner every public surface should lead with. PyPI
description is the more compact:

> A backend-neutral kernel of predictive primitives for descendant systems.

## Pitch (paragraph)

> `decepticons` is the shared mechanism layer for predictive descendants:
> substrate dynamics, controller summaries, memory primitives, feature views,
> readouts, and runtime helpers extracted from a broader experiment family so
> downstream systems can specialize without forking the kernel.

## Naming

- **Decepticons** — capitalized in prose ("Decepticons is a kernel…").
- **decepticons** — lowercase as the package name, the CLI binary, the brand
  glyph in the site nav, and inside code blocks.

## The dependency rule (always quote it the same way)

> decepticons never imports its descendants.

Enforced by `tests/test_dependency_firewall.py` (AST scan, catches any
chronohorn or descendant-package import).

## The promotion rule (always quote it the same way)

Code moves into `src/` only when **all three** hold:

1. It is a mechanism, not a project policy.
2. At least two descendants want the same thing.
3. The generalized API is simpler than keeping the duplication.

## Three-layer ecosystem (always draw it the same way)

```
decepticons  ──→  chronohorn  ──→  heinrich
   kernel          runtime          forensics
 (this repo)   training · fleet   geometry · audit
```

Dependencies flow left-to-right. Never reverse the direction.

## Key facts (numbers / one-liners)

- **numpy>=1.26** — only hard dependency for the kernel
- **Python ≥ 3.11**
- **MIT** license
- Backends: **torch** and **mlx** (optional extras)
- Causality verified by `tests/test_causality.py` (frozen, learnable_decays,
  learnable_mixing, learned_recurrence, selective scan augment, readout_bands,
  routed experts)

## Sibling project: `chronohorn`

- **Technical surfaces** (README, pyproject, CLAUDE.md, docs):
  *"the runtime descendant"* / *"chronohorn (the runtime)"* /
  *"that's chronohorn"*. Always link to <https://chronohorn.com> on first
  mention.
- **Site**: hyperlink the four existing chronohorn references to
  [chronohorn.com](https://chronohorn.com). Don't import chronohorn's
  dream-style — decepticons keeps its industrial / hazard / toxic-green
  aesthetic.

## Don't

- Don't say "Non-transformer primitives kernel" — superseded.
- Don't say "O(1) attention is deception" — typo.
- Don't say `decepticons` is a transformer alternative library, or a model
  zoo, or a training framework. It is a **kernel** — primitives only.
- Don't add hard dependencies beyond numpy. Backend-specific code goes
  behind extras.
- Don't import any descendant package from anywhere in this repo —
  the firewall test will fail.

## Surfaces this file governs

| Surface | Canonical line lives at |
|---|---|
| `README.md` | First blockquote (slogan + pitch) |
| `pyproject.toml` | `description = ...` |
| `CLAUDE.md` | First paragraph |
| `site/index.html` | `<title>`, `<meta name="description">`, `og:description`, hero `<h1>` |
| `src/decepticons/__init__.py` | First docstring line |
| `CHANGELOG.md` | Plain-prose preamble |

## Surfaces this file does NOT govern

- `docs/architecture.md`, `docs/chronohorn_boundary.md`,
  `docs/kernel_matrix.md`, `docs/related_work.md`, `docs/landscape.md`,
  `docs/lineage.md`, `docs/downstream_patterns.md` — internal doctrine
  documents with their own established voice.
- Test docstrings, internal module docstrings.
- `examples/` — they're allowed to demonstrate without re-stating doctrine.
