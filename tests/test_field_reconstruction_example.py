from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PROJECT = ROOT / "examples/projects/noncausal/field_reconstruction"

import importlib.util

_spec = importlib.util.spec_from_file_location("field_reconstruction_model", PROJECT / "model.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
FieldReconstructionModel = _mod.FieldReconstructionModel


class FieldReconstructionExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FieldReconstructionModel.build()
        self.corpus = (
            "noncausal reconstruction uses both left and right context.\n"
            "noncausal reconstruction uses both left and right context.\n"
            "the replay span should capture repeated field structure.\n"
        ) * 3

    def test_model_wraps_shared_adapter_policy(self) -> None:
        self.assertTrue(hasattr(self.model, "adapter"))
        self.assertEqual(self.model.adapter.artifact_name, self.model.config.artifact_name)
        self.assertAlmostEqual(self.model.adapter.config.blend_temperature, self.model.config.blend_temperature)
        self.assertAlmostEqual(self.model.adapter.config.replay_threshold, self.model.config.replay_threshold)
        self.assertAlmostEqual(self.model.adapter.config.agreement_threshold, self.model.config.agreement_threshold)

    def test_fit_tracks_bidirectional_replay_surface(self) -> None:
        fit = self.model.fit(self.corpus)

        self.assertGreater(fit.forward.tokens, 0)
        self.assertGreater(fit.reverse.tokens, 0)
        self.assertGreater(fit.bidirectional_context.neighborhood_count, 0)
        self.assertGreaterEqual(fit.accounting.replay_span_count, 0)

    def test_score_returns_reconstruction_and_replay_metrics(self) -> None:
        report = self.model.score(self.corpus[:192])

        self.assertGreater(report.tokens, 0)
        self.assertEqual(report.tokens, report.steps)
        self.assertGreaterEqual(report.blended_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.agreement_rate, 0.0)
        self.assertLessEqual(report.agreement_rate, 1.0)
        self.assertGreaterEqual(report.replay_rate, 0.0)
        self.assertIsInstance(report.reconstructed_text, str)
        self.assertGreater(len(report.reconstructed_text), 0)

    def test_reconstruct_preserves_length(self) -> None:
        reconstructed = self.model.reconstruct(self.corpus[:160])

        self.assertGreater(reconstructed.size, 0)
        self.assertEqual(reconstructed.dtype.kind, "u")
        self.assertEqual(reconstructed.size, len(self.corpus[:160].encode("utf-8")))


if __name__ == "__main__":
    unittest.main()
