"""Naming and dependency invariants. Tests enforce what docs can't."""
from __future__ import annotations

import ast
import re
from pathlib import Path


def _all_python_files() -> list[Path]:
    src = Path(__file__).resolve().parent.parent / "src" / "decepticons"
    return sorted(src.rglob("*.py"))


def _all_source_text() -> list[tuple[Path, str]]:
    return [(p, p.read_text()) for p in _all_python_files()]


def test_no_chronohorn_imports():
    """No source file in decepticons/ may import from chronohorn."""
    violations = []
    for path, source in _all_source_text():
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "chronohorn" in alias.name:
                        violations.append(f"{path.relative_to(path.parents[3])}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and "chronohorn" in node.module:
                violations.append(f"{path.relative_to(path.parents[3])}:{node.lineno}: from {node.module}")
    assert violations == [], "decepticons must not import chronohorn:\n" + "\n".join(violations)


def test_no_stale_opc_references():
    """The rename from open-predictive-coder/opc to decepticons is complete."""
    stale_patterns = [
        re.compile(r'\bopen.predictive.coder\b', re.IGNORECASE),
        re.compile(r'\bprog="opc"'),
        re.compile(r"'opc'"),
    ]
    violations = []
    for path, source in _all_source_text():
        for i, line in enumerate(source.splitlines(), 1):
            for pattern in stale_patterns:
                if pattern.search(line):
                    violations.append(f"{path.relative_to(path.parents[3])}:{i}: {line.strip()}")
    assert violations == [], "stale opc/open-predictive-coder references:\n" + "\n".join(violations)
