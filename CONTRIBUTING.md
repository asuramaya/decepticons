# Contributing

Thanks for your interest in `decepticons`. This is an alpha-stage research
kernel — the API will change. Bug reports, small fixes, and discussions are
welcome; please file an issue before starting larger changes so we can talk
about scope and the kernel boundary.

## Quickstart for contributors

```bash
git clone https://github.com/asuramaya/decepticons
cd decepticons
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest -v
```

For the model backends:

```bash
pip install -e ".[torch]"
pip install -e ".[metal]"   # Apple Silicon
```

## What belongs in the kernel

`decepticons` is a kernel of mechanisms, not a runtime. Read
[`docs/architecture.md`](./docs/architecture.md) and
[`docs/chronohorn_boundary.md`](./docs/chronohorn_boundary.md) before proposing
new public surfaces.

The promotion rule: code moves into `src/` only when **all three** hold:

1. It is a mechanism, not a project policy.
2. At least two descendants want the same thing.
3. The generalized API is simpler than keeping the duplication in project code.

If you're adding a new primitive, a checklist:

- [ ] backend-neutral (numpy-only) for kernel code; backend-specific code lives under `src/decepticons/models/`
- [ ] config dataclasses are frozen (`@dataclass(frozen=True)`)
- [ ] no hardcoded downstream names — use dependency injection
- [ ] one-line module docstring at the top of the file
- [ ] tests in `tests/`
- [ ] if the primitive processes sequences, a causality test in
      [`tests/test_causality.py`](./tests/test_causality.py)
- [ ] decepticons does not import its descendants — the AST scan in
      [`tests/test_dependency_firewall.py`](./tests/test_dependency_firewall.py)
      will catch you if it does

## Style

- Python ≥ 3.11. Type-annotated.
- `ruff` is the only linter — config in `pyproject.toml`. `ruff check .` should
  pass before you push.
- Line length 120, but format is handled by ruff — don't fight it.

## Tests

```bash
pytest -v                                  # full suite
pytest tests/test_causality.py -v          # causality only — runs after any architecture change
pytest tests/test_dependency_firewall.py   # the firewall — runs free, run it
```

For backend-specific tests, install the extra:

```bash
pip install -e ".[torch]" && pytest tests/test_causal_bank_torch.py -v
```

## Pull requests

- One concern per PR. Architectural cleanups and bug fixes shouldn't ride together.
- Reference the issue you're closing.
- If your PR changes a public API in `src/decepticons/__init__.py`, update
  `CHANGELOG.md` and the relevant doc in `docs/`.
- A passing CI run is required.

## Reporting bugs

Open an issue at <https://github.com/asuramaya/decepticons/issues>. A minimal
reproduction (a `python` snippet or a failing test) is worth more than a long
description.

## License

By contributing, you agree your contributions are licensed under the MIT
License — see [`LICENSE`](./LICENSE).
