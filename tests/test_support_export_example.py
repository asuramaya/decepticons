from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


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


class SupportExportExampleTests(unittest.TestCase):
    def test_support_export_uses_shared_teacher_export_surface(self) -> None:
        module = load_module(
            "support_export_model_test",
            "examples/projects/bridge/support_export/model.py",
        )
        model = module.SupportExportModel.build()
        corpus = (
            "support export compares an exact teacher against a mixed backoff student.\n"
            "support export keeps the paired probability streams and support summaries generic.\n"
        ) * 4

        fit = model.fit(corpus)
        trace = model.trace(corpus[:192])
        report = model.report(corpus[:192])

        self.assertGreater(fit.backoff.ngram.tokens, 0)
        self.assertGreater(fit.exact.tokens, 0)
        self.assertEqual(trace.teacher_probs.shape, trace.student_probs.shape)
        self.assertEqual(trace.targets.shape[0], trace.teacher_probs.shape[0])
        self.assertTrue(np.allclose(trace.teacher_probs.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(trace.student_probs.sum(axis=1), 1.0))
        self.assertGreaterEqual(report.teacher_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.student_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.mean_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.label_flip_rate, 0.0)
        self.assertLessEqual(report.label_flip_rate, 1.0)
        self.assertGreaterEqual(report.mean_exact_support, 0.0)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/bridge/support_export/probe.py")
        smoke_output = run_example("examples/projects/bridge/support_export/smoke.py")

        self.assertIn("project: support_export", probe_output)
        self.assertIn("mean bits/byte:", probe_output)
        self.assertIn("label flip rate:", smoke_output)
        self.assertIn("mean exact support:", smoke_output)


if __name__ == "__main__":
    unittest.main()
