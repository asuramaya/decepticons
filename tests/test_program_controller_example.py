from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "examples" / "projects" / "causal" / "program_controller"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decepticons.probability_diagnostics import (
    ProbabilityDiagnosticsConfig,
    probability_diagnostics,
)


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


class ProgramControllerExampleTests(unittest.TestCase):
    def test_fit_trace_and_score_expose_controller_behavior(self) -> None:
        module = load_module(
            "program_controller_model_test",
            "examples/projects/causal/program_controller/model.py",
        )
        model = module.ProgramControllerModel.build()
        corpus = (
            "controller programs should choose between prior, exact, and repair.\n"
            "controller programs should choose between prior, exact, and repair.\n"
            "the repeated prefix gives the exact route a chance to win.\n"
        ) * 3

        fit = model.fit(corpus)
        trace = model.trace(corpus[:224])
        score = model.score(corpus[:224])
        prompt = model.predict_proba(corpus[:64])

        self.assertGreater(fit.ngram.tokens, 0)
        self.assertGreater(fit.exact.tokens, 0)
        self.assertEqual(len(fit.route_names), fit.route_weights.shape[1])
        self.assertEqual(len(fit.feature_names), fit.route_weights.shape[0] - 1)
        self.assertEqual(trace.feature_names, fit.feature_names)
        self.assertEqual(trace.route_names, fit.route_names)
        self.assertGreaterEqual(fit.route_accuracy, 0.0)
        self.assertLessEqual(fit.route_accuracy, 1.0)
        self.assertGreaterEqual(fit.repair_accuracy, 0.0)
        self.assertLessEqual(fit.repair_accuracy, 1.0)
        self.assertGreaterEqual(fit.mean_repair_strength, 0.0)
        self.assertLessEqual(fit.mean_repair_strength, 1.0)
        self.assertGreaterEqual(fit.repair_span_count, 0)

        self.assertEqual(trace.prior_probs.shape, trace.exact_probs.shape)
        self.assertEqual(trace.repair_probs.shape, trace.prior_probs.shape)
        self.assertEqual(trace.mixed_probs.shape, trace.prior_probs.shape)
        self.assertEqual(trace.route_probs.shape[1], len(trace.route_names))
        self.assertEqual(trace.controller_features.shape[1], len(trace.feature_names))
        self.assertEqual(prompt.shape, (256,))
        self.assertTrue(np.all(np.isfinite(trace.mixed_probs)))
        self.assertTrue(np.allclose(trace.mixed_probs.sum(axis=1), 1.0))
        self.assertTrue(np.all((trace.route_probs >= 0.0) & (trace.route_probs <= 1.0)))
        self.assertGreaterEqual(len(trace.repair_spans), 0)

        self.assertGreaterEqual(score.prior_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.exact_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.repair_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.mixed_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.mean_route_entropy, 0.0)
        self.assertLessEqual(score.mean_route_entropy, 1.0)
        self.assertGreaterEqual(score.mean_repair_strength, 0.0)
        self.assertLessEqual(score.mean_repair_strength, 1.0)

    def test_feature_vector_uses_shared_probability_diagnostics(self) -> None:
        module = load_module(
            "program_controller_model_diagnostics_test",
            "examples/projects/causal/program_controller/model.py",
        )
        model = module.ProgramControllerModel.build(candidate_count=2)
        prefix = np.asarray([10, 11, 12, 13], dtype=np.uint8)
        prior_probs = model._ngram_distribution(prefix)
        exact_probs, exact_support, exact_order = model._exact_distribution(prefix)
        diagnostics = probability_diagnostics(
            prior_probs[None, :],
            exact_probs[None, :],
            config=ProbabilityDiagnosticsConfig(top_k=2),
        )
        vector = model._feature_vector(
            prior_probs,
            exact_probs,
            exact_support=exact_support,
            exact_order=exact_order,
            prefix_fraction=0.5,
        )

        self.assertEqual(vector.shape, (len(model.FEATURE_NAMES),))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(vector[0], float(diagnostics.entropy[0]), places=12)
        self.assertAlmostEqual(vector[1], float(diagnostics.peak[0]), places=12)
        self.assertAlmostEqual(vector[2], float(diagnostics.top_k_mass[0]), places=12)
        self.assertAlmostEqual(vector[3], float(diagnostics.overlap[0]), places=12)
        self.assertAlmostEqual(vector[4], float(diagnostics.shared_top_k_mass[0]), places=12)
        self.assertAlmostEqual(vector[-1], 1.0)

    def test_scripts_run(self) -> None:
        probe_output = run_example("examples/projects/causal/program_controller/probe.py")
        smoke_output = run_example("examples/projects/causal/program_controller/smoke.py")

        self.assertIn("project: program_controller", probe_output)
        self.assertIn("route accuracy:", probe_output)
        self.assertIn("repair spans:", smoke_output)
        self.assertIn("mixed bits/byte:", smoke_output)


if __name__ == "__main__":
    unittest.main()
