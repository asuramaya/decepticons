from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "noncausal" / "payload_choice"


def load_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_example(relative_path: str) -> str:
    script = REPO_ROOT / relative_path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    return result.stdout


module = load_module("payload_choice_model", "examples/projects/noncausal/payload_choice/model.py")
PayloadChoiceConfig = module.PayloadChoiceConfig
PayloadChoiceModel = module.PayloadChoiceModel


class PayloadChoiceExampleTests(unittest.TestCase):
    def test_trace_exposes_dense_and_sparse_layouts(self) -> None:
        model = PayloadChoiceModel.build()
        corpus = (
            "payload choice keeps dictionary layout policy local.\n"
            "payload choice keeps dictionary layout policy local.\n"
            "dense and sparse payloads should both remain visible.\n"
        ) * 3

        trace = model.trace(corpus)

        self.assertEqual(trace.dense_layout.kind, "dense")
        self.assertEqual(trace.sparse_layout.kind, "sparse")
        self.assertGreater(trace.dense_layout.payload_bytes, 0)
        self.assertGreater(trace.sparse_layout.payload_bytes, 0)
        self.assertGreater(trace.dense_layout.positions.size, 0)
        self.assertGreater(trace.sparse_layout.positions.size, 0)
        self.assertTrue(np.all(trace.position_scores >= 0.0))
        self.assertTrue(np.all(trace.position_scores <= 1.0))

    def test_dense_bias_can_force_dense_layout(self) -> None:
        config = PayloadChoiceConfig(dense_bias=5.0)
        model = PayloadChoiceModel.build(config=config)
        corpus = "dense layout should win when the local policy strongly prefers it.\n" * 2

        report = model.report(corpus)

        self.assertEqual(report.selected_layout, "dense")
        self.assertGreater(report.dense_payload_bytes, 0)
        self.assertGreaterEqual(report.coverage_ratio, 0.0)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/noncausal/payload_choice/probe.py")
        smoke_output = run_example("examples/projects/noncausal/payload_choice/smoke.py")

        self.assertIn("selected layout:", probe_output)
        self.assertIn("coverage ratio:", smoke_output)


if __name__ == "__main__":
    unittest.main()
