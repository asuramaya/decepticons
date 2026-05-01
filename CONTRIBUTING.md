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

## Releasing

Releases are one command. The script in [`scripts/release.sh`](./scripts/release.sh)
bumps the version everywhere, commits, tags, and pushes. The push triggers
[`.github/workflows/release.yml`](./.github/workflows/release.yml) which builds,
publishes to PyPI, and creates a GitHub Release — all automatic.

### Before tagging

While you work, log changes under `## [Unreleased]` in [`CHANGELOG.md`](./CHANGELOG.md).
This is what becomes the release notes for the next version. Group entries by
`### Added`, `### Changed`, `### Fixed`, `### Removed`.

### Cut a release

```bash
scripts/release.sh 0.1.1     # bump patch
scripts/release.sh 0.2.0     # bump minor
scripts/release.sh 1.0.0     # first stable
scripts/release.sh 1.1.0-rc1 # pre-release
```

The script:

1. Checks the working tree is clean and `main` is in sync with `origin/main`.
2. Bumps the version in `pyproject.toml`, `CHANGELOG.md` (moves `[Unreleased]`
   entries under a new `[X.Y.Z] — <today>` header), and `site/index.html`
   (kicker + footer).
3. Shows the diff and asks for confirmation.
4. Commits as `chore(release): vX.Y.Z`, tags `vX.Y.Z`, pushes both.

### What happens after the push

Pushing the tag fires `release.yml`:

| Job | Does |
| --- | --- |
| `build` | Verifies the tag matches `pyproject.toml` version, builds sdist + wheel, runs `twine check`, smoke-imports the wheel in a fresh venv, uploads `dist/` as an artifact. |
| `publish` | Publishes to PyPI via OIDC trusted publishing. No tokens required — the `pypi` GitHub environment authorizes the run. |
| `github-release` | Extracts the matching section from `CHANGELOG.md`, creates a GitHub Release at `vX.Y.Z` with those notes, and attaches the wheel + sdist. |

In parallel, the push of the bump commit (which touched `site/index.html`)
fires [`.github/workflows/pages.yml`](./.github/workflows/pages.yml) and
redeploys <https://decepticons.win> with the new version visible in the kicker
and footer.

### Versioning rules (semver)

While `< 1.0.0`:
- `0.1.0 → 0.1.1` — bug fix only, no API change
- `0.1.x → 0.2.0` — breaking changes are allowed pre-1.0
- `0.x.x → 1.0.0` — first stable API. After this, breaking changes need a major bump.

### If something goes wrong

| Failure | Recovery |
| --- | --- |
| `build` fails on `twine check` | Fix locally, delete the bad tag (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`), bump to the next patch, run `scripts/release.sh` again. |
| `publish` fails | Re-run the failed job from the GitHub Actions UI. The build artifact is still there. |
| Released a broken version | You cannot re-upload to PyPI. Yank with `twine yank decepticons==X.Y.Z -m "reason"` and ship `X.Y.(Z+1)` with the fix. |

## Reporting bugs

Open an issue at <https://github.com/asuramaya/decepticons/issues>. A minimal
reproduction (a `python` snippet or a failing test) is worth more than a long
description.

## License

By contributing, you agree your contributions are licensed under the MIT
License — see [`LICENSE`](./LICENSE).
