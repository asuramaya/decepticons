from __future__ import annotations

import unittest

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PROJECT = ROOT / "examples/projects/causal/packed_memory_controller"

from decepticons.probability_diagnostics import (
    ProbabilityDiagnosticsConfig,
    probability_diagnostics,
)

import importlib.util
_spec = importlib.util.spec_from_file_location("packed_memory_controller_model", PROJECT / "model.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
PackedMemoryControllerModel = _mod.PackedMemoryControllerModel


class PackedMemoryControllerExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = PackedMemoryControllerModel.build()
        self.corpus = (
            "packed memory gives the controller several reusable priors.\n"
            "exact context repair still matters when local support is strong.\n"
            "agreement and candidate mass should help decide which source to trust.\n"
        ) * 3

    def test_fit_learns_controller_weights(self) -> None:
        fit = self.model.fit(self.corpus)

        self.assertGreater(fit.ngram.tokens, 0)
        self.assertGreater(fit.exact.tokens, 0)
        self.assertEqual(len(fit.feature_names), fit.controller_weights.shape[0])
        self.assertGreaterEqual(fit.exact_win_rate, 0.0)
        self.assertLessEqual(fit.exact_win_rate, 1.0)

    def test_trace_returns_normalized_distributions_and_trust(self) -> None:
        self.model.fit(self.corpus)
        trace = self.model.trace(self.corpus[:192])

        self.assertGreater(trace.steps, 0)
        self.assertEqual(trace.prior_probs.shape, trace.exact_probs.shape)
        self.assertEqual(trace.prior_probs.shape, trace.mixed_probs.shape)
        self.assertEqual(trace.controller_features.shape[1], len(trace.feature_names))
        self.assertTrue(((trace.memory_trust >= 0.0) & (trace.memory_trust <= 1.0)).all())
        self.assertTrue(abs(trace.mixed_probs.sum(axis=1) - 1.0).max() < 1e-9)

    def test_controller_vector_uses_shared_probability_diagnostics(self) -> None:
        model = PackedMemoryControllerModel.build(candidate_count=2)
        prior = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        exact = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)

        diagnostics = probability_diagnostics(
            prior[None, :],
            exact[None, :],
            config=ProbabilityDiagnosticsConfig(top_k=2),
        )
        vector = model._controller_vector(
            prior,
            exact,
            exact_support=0.75,
            exact_order=2,
        )

        self.assertEqual(vector.shape, (8,))
        self.assertAlmostEqual(vector[0], float(diagnostics.entropy[0]), places=12)
        self.assertAlmostEqual(vector[1], float(diagnostics.peak[0]), places=12)
        self.assertAlmostEqual(vector[2], float(diagnostics.top_k_mass[0]), places=12)
        self.assertAlmostEqual(vector[3], float(diagnostics.overlap[0]), places=12)
        self.assertAlmostEqual(vector[4], float(diagnostics.shared_top_k_mass[0]), places=12)

    def test_score_exposes_memory_first_metrics(self) -> None:
        self.model.fit(self.corpus)
        trace = self.model.trace(self.corpus[:192])
        score = self.model.score(self.corpus[:192])

        self.assertGreater(score.tokens, 0)
        self.assertGreaterEqual(score.prior_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.exact_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.mixed_bits_per_byte, 0.0)
        self.assertGreaterEqual(score.mean_memory_trust, 0.0)
        self.assertLessEqual(score.mean_memory_trust, 1.0)
        self.assertGreaterEqual(score.mean_agreement_mass, 0.0)
        self.assertAlmostEqual(
            score.mean_agreement_mass,
            float(np.mean(trace.controller_features[:, self.model.FEATURE_NAMES.index("agreement_mass")])),
            places=12,
        )
        self.assertAlmostEqual(
            score.mean_candidate4,
            float(np.mean(trace.controller_features[:, self.model.FEATURE_NAMES.index("candidate4")])),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
