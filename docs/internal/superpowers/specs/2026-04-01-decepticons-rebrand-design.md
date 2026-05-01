# Decepticons Rebrand — Design Spec

## Summary

Rename the library from "Open Predictive Coder" / "OPC" to "Decepticons". This is a mechanical rename of the project identity — package name, module paths, CLI entry point, and the one class directly named after the project. All other class names, terminology, and architecture remain unchanged.

## Substitution Map

| Pattern | Replacement | Scope |
|---------|-------------|-------|
| `decepticons` | `decepticons` | Python module name, all imports, internal references |
| `decepticons` | `decepticons` | Package name (pyproject.toml, hyphenated references in docs) |
| `OpenPredictiveCoderConfig` | `DecepticonsConfig` | Class name + all usages |
| `OpenPredictiveCoder` | `Decepticons` | Only where it appears as part of the Config class name — not as free text in docs describing the project generically |
| `opc` (CLI entry point) | `decepticons` | pyproject.toml `[project.scripts]`, cli.py references, docs/README CLI examples |
| `OPC` / `Open Predictive Coder` (prose) | `Decepticons` | README, docs, comments that name the project |

## What Changes

### 1. Directory rename
- `src/decepticons/` → `src/decepticons/`

### 2. pyproject.toml
- `name = "decepticons"` → `name = "decepticons"`
- `[project.scripts]`: `opc = ...` → `decepticons = ...`
- Package directory references

### 3. All Python files under `src/decepticons/`
- Update any self-referencing imports (e.g., `from decepticons.x import y` → `from decepticons.x import y`)
- Rename `OpenPredictiveCoderConfig` → `DecepticonsConfig` in `config.py` and all usages

### 4. Tests (`tests/`)
- All `import decepticons` → `import decepticons`
- All `from decepticons` → `from decepticons`

### 5. Examples (`examples/`)
- All imports updated
- CLI invocation examples updated (`opc` → `decepticons`)

### 6. Docs (`docs/`)
- Project name references updated
- CLI examples updated

### 7. README.md
- Project title, description, install/usage examples

### 8. CLAUDE.md (if present)
- Project name references

## What Does NOT Change

- **Class names**: `ByteLatentPredictiveCoder`, `EchoStateReservoir`, `CausalPredictiveAdapter`, `TokenSubstrate`, all adapters, all substrates, all memory types, all control primitives, all readouts — all stay as-is
- **Terminology**: descendants, ancestors, kernel, adapters, substrates, views, readouts, experts, artifacts
- **Architecture**: layer structure, file organization within the package, test structure, example project structure
- **The only class that changes**: `OpenPredictiveCoderConfig` → `DecepticonsConfig` (because it is literally named after the project)

## Risks and Edge Cases

- **`opc` as substring**: The string "opc" may appear inside other identifiers or words. The CLI rename must target only the entry-point definition and CLI invocation examples, not arbitrary occurrences.
- **Git history**: This is a large rename. A single commit keeps `git log --follow` useful.
- **Reinstall required**: After the directory rename, users will need `pip install -e .` again.

## Execution Order

1. Rename the directory `src/decepticons/` → `src/decepticons/`
2. Find-and-replace `decepticons` → `decepticons` across all `.py` files
3. Find-and-replace `decepticons` → `decepticons` in pyproject.toml and docs
4. Rename `OpenPredictiveCoderConfig` → `DecepticonsConfig` across all files
5. Update CLI entry point (`opc` → `decepticons`) in pyproject.toml and docs
6. Update prose references (README, docs, comments) from "Open Predictive Coder" / "OPC" to "Decepticons"
7. Run tests to verify nothing broke
8. Single commit
