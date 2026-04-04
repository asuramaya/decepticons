from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

from decepticons.teacher_export import TeacherExportAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "bridge" / "teacher_export"


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


class TeacherExportExampleTests(unittest.TestCase):
    def test_export_labels_and_probability_diagnostics(self) -> None:
        module = load_module("teacher_export_model_test", "examples/projects/bridge/teacher_export/model.py")
        model = module.TeacherExportModel(module.TeacherExportConfig(hidden_dim=16))
        corpus = (
            "teacher export labels stay local to the example.\n"
            "attack-aware reporting compares clean and attacked streams.\n"
        )

        trace = model.export(corpus[:192])
        report = model.report(corpus)
        summary = model.summary(corpus)
        shared_report = model.teacher_export.export(trace.teacher_probs, trace.student_probs, targets=trace.teacher_labels)

        self.assertIsInstance(model.teacher_export, TeacherExportAdapter)
        self.assertEqual(trace.teacher_probs.shape, trace.student_probs.shape)
        self.assertEqual(trace.teacher_probs.shape, trace.attacked_probs.shape)
        self.assertEqual(trace.teacher_labels.shape[0], trace.steps)
        self.assertEqual(trace.attacked_labels.shape[0], trace.steps)
        self.assertGreater(trace.tokens, trace.steps)
        self.assertTrue(np.any(trace.clean_tokens != trace.attacked_tokens))
        self.assertGreaterEqual(report.teacher_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.student_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.attack_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.label_flip_rate, 0.0)
        self.assertLessEqual(report.label_flip_rate, 1.0)
        self.assertGreaterEqual(report.attack_mutation_rate, 0.0)
        self.assertLessEqual(report.attack_mutation_rate, 1.0)
        self.assertGreaterEqual(report.mean_teacher_margin, 0.0)
        self.assertGreaterEqual(report.mean_attack_margin, 0.0)
        self.assertGreaterEqual(report.clean_deterministic_fraction, 0.0)
        self.assertLessEqual(report.clean_deterministic_fraction, 1.0)
        self.assertGreaterEqual(report.attacked_deterministic_fraction, 0.0)
        self.assertLessEqual(report.attacked_deterministic_fraction, 1.0)
        self.assertAlmostEqual(summary["teacher_bits_per_byte"], report.teacher_bits_per_byte, places=12)
        self.assertAlmostEqual(summary["label_flip_rate"], report.label_flip_rate, places=12)
        self.assertAlmostEqual(summary["mean_teacher_margin"], report.mean_teacher_margin, places=12)
        self.assertAlmostEqual(summary["mean_attack_margin"], report.mean_attack_margin, places=12)
        self.assertAlmostEqual(report.teacher_bits_per_byte, float(shared_report.teacher_bits_per_byte), places=12)
        self.assertAlmostEqual(report.student_bits_per_byte, float(shared_report.student_bits_per_byte), places=12)
        self.assertAlmostEqual(report.student_label_disagreement, shared_report.label_flip_rate, places=12)
        self.assertAlmostEqual(report.mean_candidate4, float(np.mean(trace.features.top_k_mass)), places=12)
        self.assertAlmostEqual(report.mean_agreement, float(np.mean(trace.features.overlap)), places=12)
        self.assertAlmostEqual(report.mean_agreement_mass, float(np.mean(trace.features.shared_top_k_mass)), places=12)
        self.assertTrue(np.all(trace.features.top2_margin >= 0.0))

    def test_trace_arrays_are_probability_like(self) -> None:
        module = load_module("teacher_export_model_prob_test", "examples/projects/bridge/teacher_export/model.py")
        model = module.TeacherExportModel(module.TeacherExportConfig(hidden_dim=16))
        trace = model.export("teacher export example text with repeated structure.")

        self.assertTrue(np.allclose(trace.teacher_probs.sum(axis=-1), 1.0, atol=1e-8))
        self.assertTrue(np.allclose(trace.student_probs.sum(axis=-1), 1.0, atol=1e-8))
        self.assertTrue(np.allclose(trace.attacked_probs.sum(axis=-1), 1.0, atol=1e-8))
        self.assertTrue(np.all(trace.features.entropy >= 0.0))
        self.assertTrue(np.all(trace.features.peak <= 1.0))
        self.assertTrue(np.all(trace.features.top_k_mass <= 1.0))
        self.assertTrue(np.all(trace.features.overlap <= 1.0))
        self.assertTrue(np.all(trace.features.shared_top_k_mass <= 1.0))

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/bridge/teacher_export/probe.py")
        smoke_output = run_example("examples/projects/bridge/teacher_export/smoke.py")

        self.assertIn("teacher bits/byte:", probe_output)
        self.assertIn("label flip rate:", probe_output)
        self.assertIn("attack bits/byte:", smoke_output)
        self.assertIn("attacked deterministic fraction:", smoke_output)


if __name__ == "__main__":
    unittest.main()
