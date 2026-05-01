<!-- Thanks for the contribution! Brief is good. -->

## Summary

<!-- One sentence: what does this change and why. -->

## The promotion rule

If this PR adds or moves code into `src/decepticons/`, please confirm:

- [ ] It is a mechanism, not a project policy.
- [ ] At least two descendants want the same thing.
- [ ] The generalized API is simpler than keeping the duplication.

## Checklist

- [ ] One concern per PR (architectural cleanups and bug fixes don't ride together)
- [ ] `pytest -v` passes locally
- [ ] `ruff check .` passes locally
- [ ] If this changes a public API in `src/decepticons/__init__.py`: `CHANGELOG.md` updated under `[Unreleased]` and the relevant `docs/` page touched
- [ ] If this is a sequence-processing primitive: a causality test in `tests/test_causality.py`
- [ ] No descendant (e.g. `chronohorn`) imports — `tests/test_dependency_firewall.py` still passes

## Related issues

<!-- Closes #N, refs #M -->
