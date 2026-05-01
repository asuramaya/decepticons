# Decepticons

<p align="center">
  <img src="https://raw.githubusercontent.com/asuramaya/decepticons/main/docs/logo.webp" alt="Decepticons" width="520">
</p>

<p align="center">
  <a href="https://pypi.org/project/decepticons/"><img alt="PyPI" src="https://img.shields.io/pypi/v/decepticons.svg"></a>
  <a href="https://github.com/asuramaya/decepticons/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/asuramaya/decepticons/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/asuramaya/decepticons/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-alpha-orange">
</p>

<p align="center">
  <a href="https://decepticons.win"><b>Website</b></a> ·
  <a href="https://github.com/asuramaya/decepticons/blob/main/docs/architecture.md">Architecture</a> ·
  <a href="https://github.com/asuramaya/decepticons/blob/main/docs/kernel_matrix.md">Kernel matrix</a> ·
  <a href="https://github.com/asuramaya/decepticons/tree/main/examples">Examples</a> ·
  <a href="https://github.com/asuramaya/decepticons/blob/main/docs/related_work.md">Related work</a>
</p>

> **O(n) attention is deception.** A backend-neutral kernel of predictive
> primitives — substrates, memory, gating, routing, readouts — that downstream
> systems combine into trained models without forking the kernel itself.

`decepticons` is the shared mechanism layer for predictive descendants:
substrate dynamics, controller summaries, memory primitives, feature views,
readouts, and runtime helpers extracted from a broader experiment family so
downstream systems can specialize without forking the kernel.

## Install

Python ≥ 3.11. Numpy is the only hard dependency for the kernel.

```bash
pip install decepticons
```

For the model backends:

```bash
pip install "decepticons[torch]"   # PyTorch CausalBankModel + routed readouts
pip install "decepticons[metal]"   # Apple MLX backend
```

For development from source:

```bash
git clone https://github.com/asuramaya/decepticons
cd decepticons
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -v
```

## Quickstart

```python
from decepticons import ByteCodec, ByteLatentPredictiveCoder

text = "predictive coding likes repeated structure.\n" * 64
model = ByteLatentPredictiveCoder()
report = model.fit(text)

prompt = ByteCodec.encode_text("predictive ")
sample = model.generate(prompt, steps=40, greedy=True)

print(report.train_bits_per_byte)
print(ByteCodec.decode_text(sample))
```

CLI:

```bash
decepticons fit --input ./corpus.txt --prompt "predictive " --generate 80
```

A complete worked example lives in
[`examples/quickstart.py`](https://github.com/asuramaya/decepticons/blob/main/examples/quickstart.py).
For descendant-shaped projects, see
[`examples/projects/`](https://github.com/asuramaya/decepticons/tree/main/examples/projects).

## What's in the kernel

| Area | Highlights |
| --- | --- |
| **Substrates** | recurrent, delay, linear-memory, oscillatory, mixed, hierarchical |
| **Control** | controller summaries, pathway gates, summary routing, hormone modulation, predictive surprise |
| **Memory** | exact-context, n-gram, statistical-backoff, online n-gram, cache views |
| **Views** | byte-latent, hierarchical, linear-memory, sampled multiscale, bridge features, probability diagnostics |
| **Readouts** | ridge, frozen-readout expert, sampled multiscale, GRU recurrent, routed squared-ReLU |
| **Adapters** | causal predictive, oracle analysis, bridge export, noncausal reconstructive, paired teacher/export |
| **Runtime** | traces, fit reports, rollout evaluation, transfer probes, train-mode checkpoints, artifact accounting |
| **Causal-bank** | family metadata + deterministic substrate construction (frozen / learnable-decays / learnable-mixing / learned-recurrence / gated-retention) |
| **Backends** | numpy-only kernel; PyTorch and MLX `CausalBankModel` implementations |

Full capability matrix: [`docs/kernel_matrix.md`](https://github.com/asuramaya/decepticons/blob/main/docs/kernel_matrix.md).

## Architecture

```
decepticons  ──→  chronohorn  ──→  heinrich
  kernel          runtime          forensics
 (this repo)   training · fleet   geometry · audit
```

Three layers inside this repo:

1. **Kernel** — `src/decepticons/`. Public package. Reusable mechanisms only.
2. **Project descendants** — `examples/projects/`. Pressure-tests the kernel
   boundary with concrete descendant shapes (causal · oracle · bridge · noncausal · byte-latent).
3. **Tooling** — `examples/tools/`. Development and analysis scripts. Not part
   of the public package.

Code moves into `src/` only when **all three** hold:

1. It is a mechanism, not a project policy.
2. At least two descendants want the same thing.
3. The generalized API is simpler than keeping the duplication.

This rule is the main defense against turning the kernel into a renamed
collection of branches. Full detail in
[`docs/architecture.md`](https://github.com/asuramaya/decepticons/blob/main/docs/architecture.md)
and the boundary against the runtime in
[`docs/chronohorn_boundary.md`](https://github.com/asuramaya/decepticons/blob/main/docs/chronohorn_boundary.md).

## Causality is verified

All substrate modes are verified by
[`tests/test_causality.py`](https://github.com/asuramaya/decepticons/blob/main/tests/test_causality.py):
it feeds two identical sequences up to position *t*, different after *t*. If
logits at position *t* differ, causality is violated and CI fails. Modes
verified: `frozen`, `learnable_mixing`, `learnable_decays`, selective scan
augment (`state_dim > 0`), `readout_bands`, routed experts.

decepticons never imports its descendants — enforced by an AST scan in
[`tests/test_dependency_firewall.py`](https://github.com/asuramaya/decepticons/blob/main/tests/test_dependency_firewall.py).

## Docs

- [Architecture](https://github.com/asuramaya/decepticons/blob/main/docs/architecture.md) — package map, three-layer model, promotion rule
- [Kernel matrix](https://github.com/asuramaya/decepticons/blob/main/docs/kernel_matrix.md) — capability matrix
- [Chronohorn boundary](https://github.com/asuramaya/decepticons/blob/main/docs/chronohorn_boundary.md) — boundary against the runtime descendant
- [Downstream patterns](https://github.com/asuramaya/decepticons/blob/main/docs/downstream_patterns.md) — causal, noncausal, oracle, bridge, byte-latent patterns
- [Related work](https://github.com/asuramaya/decepticons/blob/main/docs/related_work.md) — research anchors and prior art
- [Landscape](https://github.com/asuramaya/decepticons/blob/main/docs/landscape.md) — ecosystem snapshot (March 2026)
- [Lineage](https://github.com/asuramaya/decepticons/blob/main/docs/lineage.md) — source attribution
- [Examples](https://github.com/asuramaya/decepticons/tree/main/examples) — example descendants and tooling
- [Tests](https://github.com/asuramaya/decepticons/tree/main/tests) — verification surface

## Scope

This is a research kernel and reference implementation. The current pressure
from descendants is O(n) causal-bank architecture search — cheap ablation lanes
to separate mechanisms before promotion, with scale and context survival
checked in the descendant runtime.

It is not a frontier runtime, a production compression stack, or a benchmark
claim. It exists to keep the shared mechanism layer reusable and legible.

## Contributing

See [`CONTRIBUTING.md`](https://github.com/asuramaya/decepticons/blob/main/CONTRIBUTING.md).
Issues and pull requests welcome.

## License

MIT — see [`LICENSE`](https://github.com/asuramaya/decepticons/blob/main/LICENSE).
