from __future__ import annotations

import unittest

from decepticons.artifacts import ArtifactAccounting
from decepticons.noncausal_reconstructive import (
    NoncausalReconstructiveAdapter,
    NoncausalReconstructiveConfig,
    NoncausalReconstructiveFitReport,
    NoncausalReconstructiveReport,
    NoncausalReconstructiveTrace,
)


class NoncausalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = NoncausalReconstructiveAdapter(
            NoncausalReconstructiveConfig(
                bidirectional_left_order=1,
                bidirectional_right_order=1,
                replay_threshold=0.25,
                agreement_threshold=0.25,
            )
        )
        self.corpus = (
            "field reconstruction uses both left and right context.\n"
            "field reconstruction uses both left and right context.\n"
            "the replay span should capture repeated field structure.\n"
        ) * 3

    def test_surface_exposes_noncausal_methods(self) -> None:
        for name in ("fit", "trace", "reconstruct", "score"):
            self.assertTrue(hasattr(self.model, name), f"missing expected method {name}")

    def test_fit_trace_and_score_round_trip(self) -> None:
        fit = self.model.fit(self.corpus)
        trace = self.model.trace(self.corpus[:192])
        report = self.model.score(self.corpus[:192])
        reconstructed = self.model.reconstruct(self.corpus[:160])

        self.assertIsInstance(fit, NoncausalReconstructiveFitReport)
        self.assertIsInstance(trace, NoncausalReconstructiveTrace)
        self.assertIsInstance(report, NoncausalReconstructiveReport)
        self.assertIsInstance(fit.accounting, ArtifactAccounting)
        self.assertEqual(fit.accounting.artifact_name, "noncausal_reconstructive")
        self.assertGreater(fit.forward.tokens, 0)
        self.assertGreater(fit.reverse.tokens, 0)
        self.assertGreater(fit.bidirectional_context.neighborhood_count, 0)
        self.assertGreaterEqual(trace.tokens, 1)
        self.assertEqual(trace.tokens, trace.steps)
        self.assertEqual(trace.left_probs.shape, trace.right_probs.shape)
        self.assertEqual(trace.left_probs.shape, trace.blended_probs.shape)
        self.assertEqual(reconstructed.dtype.kind, "u")
        self.assertEqual(reconstructed.size, len(self.corpus[:160].encode("utf-8")))
        self.assertGreaterEqual(report.blended_bits_per_byte, 0.0)
        self.assertGreaterEqual(report.agreement_rate, 0.0)
        self.assertLessEqual(report.agreement_rate, 1.0)
        self.assertGreaterEqual(report.replay_rate, 0.0)
        self.assertGreaterEqual(report.replay_span_count, 0)
        self.assertIsInstance(report.bidirectional_context, type(fit.bidirectional_context))
        self.assertIsInstance(report.accounting, ArtifactAccounting)
        self.assertIsInstance(report.reconstructed_text, str)
        self.assertGreater(len(report.reconstructed_text), 0)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.fit("")
        with self.assertRaises(ValueError):
            self.model.trace("")

    def test_accounting_accessor_tracks_latest_fit(self) -> None:
        fit = self.model.fit(self.corpus)
        accounting = self.model.accounting()

        self.assertEqual(accounting, fit.accounting)
        self.assertGreaterEqual(accounting.artifact_bytes, 0)
        self.assertGreaterEqual(accounting.replay_bytes, 0)


if __name__ == "__main__":
    unittest.main()
