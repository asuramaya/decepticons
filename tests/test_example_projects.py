from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


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
    def test_carving_machine_like_probe_runs(self) -> None:
        output = run_example("examples/projects/carving_machine_like/probe.py")
        self.assertIn("state_dim:", output)
        self.assertIn("feature_dim:", output)
        self.assertIn("bank_slices:", output)

    def test_carving_machine_like_smoke_runs(self) -> None:
        output = run_example("examples/projects/carving_machine_like/smoke.py")
        self.assertIn("train bits/byte:", output)
        self.assertIn("score bits/byte:", output)

    def test_conker_early_like_smoke_runs(self) -> None:
        output = run_example("examples/projects/conker_early_like/smoke.py")
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

    def test_conker_1_like_smoke_runs(self) -> None:
        output = run_example("examples/projects/conker_1_like/smoke.py")
        self.assertIn("project: conker_1_like", output)
        self.assertIn("mixed score bits/byte:", output)

    def test_conker_2_like_smoke_runs(self) -> None:
        output = run_example("examples/projects/conker_2_like/smoke.py")
        self.assertIn("project: conker_2_like", output)
        self.assertIn("mixed score bits/byte:", output)

    def test_conker_3_like_smoke_runs(self) -> None:
        output = run_example("examples/projects/conker_3_like/smoke.py")
        self.assertIn("project: conker_3_like", output)
        self.assertIn("corrected score bits/byte:", output)

    def test_conker_readmes_record_boundary_decisions(self) -> None:
        conker_1_readme = (PROJECTS_ROOT / "conker_1_like" / "README.md").read_text(encoding="utf-8")
        conker_2_readme = (PROJECTS_ROOT / "conker_2_like" / "README.md").read_text(encoding="utf-8")
        conker_3_readme = (PROJECTS_ROOT / "conker_3_like" / "README.md").read_text(encoding="utf-8")

        self.assertIn("kernel now has sampled readout", conker_1_readme)
        self.assertIn("runtime rollout and slow-update knobs already live in the kernel", conker_2_readme)
        self.assertIn("rollout mode switches already live in the kernel", conker_3_readme)


if __name__ == "__main__":
    unittest.main()
