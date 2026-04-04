from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "bridge" / "agreement_export"


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
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.stdout


class AgreementExportExampleTests(unittest.TestCase):
    def test_trace_and_report_shapes(self) -> None:
        module = load_module("agreement_export_model_test", "examples/projects/bridge/agreement_export/model.py")
        model = module.AgreementExportModel(module.AgreementExportConfig())
        corpus = (
            "agreement export compares two local streams.\n"
            "the shared bridge transform stays generic.\n"
        ) * 4

        trace = model.trace(corpus[:192])
        report = model.report(corpus)
        summary = model.summary(corpus)

        self.assertEqual(trace.left_probs.shape, trace.right_probs.shape)
        self.assertEqual(trace.features.entropy.shape[0], trace.left_probs.shape[0])
        self.assertGreater(trace.tokens, trace.steps)
        self.assertGreaterEqual(report.mean_entropy, 0.0)
        self.assertLessEqual(report.mean_agreement, 1.0)
        self.assertGreaterEqual(report.mean_consensus_ratio, 0.0)
        self.assertGreaterEqual(report.mean_disagreement, 0.0)
        self.assertLessEqual(report.mean_disagreement, 1.0)
        self.assertAlmostEqual(summary["mean_agreement"], report.mean_agreement, places=12)
        self.assertAlmostEqual(summary["mean_consensus_ratio"], report.mean_consensus_ratio, places=12)

    def test_trace_arrays_are_probability_like(self) -> None:
        module = load_module("agreement_export_model_prob_test", "examples/projects/bridge/agreement_export/model.py")
        model = module.AgreementExportModel(module.AgreementExportConfig())
        trace = model.trace("agreement export sample text with repeated structure." * 3)

        self.assertTrue(np.allclose(trace.left_probs.sum(axis=-1), 1.0, atol=1e-8))
        self.assertTrue(np.allclose(trace.right_probs.sum(axis=-1), 1.0, atol=1e-8))
        self.assertTrue(np.all(trace.features.entropy >= 0.0))
        self.assertTrue(np.all(trace.features.peak <= 1.0))

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/bridge/agreement_export/probe.py")
        smoke_output = run_example("examples/projects/bridge/agreement_export/smoke.py")

        self.assertIn("mean consensus ratio:", probe_output)
        self.assertIn("mean disagreement:", smoke_output)


if __name__ == "__main__":
    unittest.main()
