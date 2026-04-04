from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = REPO_ROOT / "examples" / "projects"


def run_example(relative_path: str) -> str:
    script = REPO_ROOT / relative_path
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.stdout


class ExampleProjectTests(unittest.TestCase):
    def test_hierarchical_predictive_probe_runs(self) -> None:
        output = run_example("examples/projects/ancestor/hierarchical_predictive/probe.py")
        self.assertIn("state_dim:", output)
        self.assertIn("feature_dim:", output)
        self.assertIn("bank_slices:", output)

    def test_hierarchical_predictive_smoke_runs(self) -> None:
        output = run_example("examples/projects/ancestor/hierarchical_predictive/smoke.py")
        self.assertIn("train bits/byte:", output)
        self.assertIn("score bits/byte:", output)

    def test_exact_context_repair_smoke_runs(self) -> None:
        output = run_example("examples/projects/causal/exact_context_repair/smoke.py")
        self.assertIn("base bits/byte:", output)
        self.assertIn("mixed bits/byte:", output)
        values: dict[str, float] = {}
        for line in output.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            try:
                values[key] = float(value)
            except ValueError:
                continue
        self.assertLess(values["mixed bits/byte"], values["base bits/byte"])

    def test_memory_stability_smoke_runs(self) -> None:
        output = run_example("examples/projects/causal/memory_stability/smoke.py")
        self.assertIn("project: memory_stability", output)
        self.assertIn("mixed score bits/byte:", output)

    def test_linear_correction_smoke_runs(self) -> None:
        output = run_example("examples/projects/causal/linear_correction/smoke.py")
        self.assertIn("project: linear_correction", output)
        self.assertIn("mixed score bits/byte:", output)

    def test_residual_repair_smoke_runs(self) -> None:
        output = run_example("examples/projects/causal/residual_repair/smoke.py")
        self.assertIn("project: residual_repair", output)
        self.assertIn("corrected score bits/byte:", output)

    def test_causal_variant_readmes_record_boundary_decisions(self) -> None:
        variant_1_readme = (PROJECTS_ROOT / "causal" / "memory_stability" / "README.md").read_text(encoding="utf-8")
        variant_2_readme = (PROJECTS_ROOT / "causal" / "linear_correction" / "README.md").read_text(encoding="utf-8")
        variant_3_readme = (PROJECTS_ROOT / "causal" / "residual_repair" / "README.md").read_text(encoding="utf-8")

        self.assertIn("stays in the kernel", variant_1_readme)
        self.assertIn("already live in the kernel", variant_2_readme)
        self.assertIn("already live in the kernel", variant_3_readme)


if __name__ == "__main__":
    unittest.main()
