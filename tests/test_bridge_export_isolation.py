from __future__ import annotations

import unittest
from dataclasses import fields

from decepticons import (
    ByteCodec,
    CausalPredictiveAdapter,
    CausalPredictiveFitReport,
    CausalPredictiveScore,
)
from decepticons.bridge_features import BridgeFeatureArrays, BridgeFeatureConfig
from decepticons.exact_context import ExactContextConfig, ExactContextMemory
from decepticons.ngram_memory import NgramMemory, NgramMemoryConfig


def _field_names(cls: type[object]) -> set[str]:
    return {field.name for field in fields(cls)}


class BridgeExportIsolationTests(unittest.TestCase):
    def test_causal_reports_do_not_pick_up_bridge_or_oracle_fields(self) -> None:
        fit_fields = _field_names(CausalPredictiveFitReport)
        score_fields = _field_names(CausalPredictiveScore)

        forbidden = {
            "bridge_features",
            "bridge_export",
            "candidate4",
            "candidate_count",
            "entropy",
            "agreement",
            "agreement_mass",
            "left_context_count",
            "right_context_count",
            "pair_context_count",
            "neighborhoods",
        }

        self.assertFalse(fit_fields & forbidden)
        self.assertFalse(score_fields & forbidden)

    def test_causal_adapter_runtime_objects_do_not_expose_bridge_oracle_payloads(self) -> None:
        exact_config = ExactContextConfig(vocabulary_size=256, max_order=2, alpha=0.0)
        ngram_config = NgramMemoryConfig(vocabulary_size=256, bigram_alpha=0.0, trigram_alpha=0.0)
        adapter = CausalPredictiveAdapter(
            exact_context=ExactContextMemory(exact_config),
            ngram_memory=NgramMemory(ngram_config),
        )
        adapter.fit((ByteCodec.encode_text("abababab"),))
        score = adapter.score("abab")
        fit_report = adapter.fit(("abab",))

        self.assertFalse(hasattr(score, "bridge_features"))
        self.assertFalse(hasattr(score, "bridge_export"))
        self.assertFalse(hasattr(score, "bidirectional_context"))
        self.assertFalse(hasattr(fit_report, "bridge_features"))
        self.assertFalse(hasattr(fit_report, "bidirectional_context"))
        self.assertIsNone(score.__dict__.get("bridge_features"))
        self.assertIsNone(fit_report.__dict__.get("bridge_features"))

    def test_bridge_features_stay_in_their_own_surface(self) -> None:
        config = BridgeFeatureConfig(candidate_count=4)
        self.assertEqual(config.candidate_count, 4)

        bridge_fields = _field_names(BridgeFeatureArrays)
        self.assertEqual(
            bridge_fields,
            {"entropy", "peak", "candidate4", "agreement", "agreement_mass"},
        )
        self.assertNotIn("bits_per_byte", bridge_fields)
        self.assertNotIn("exact_support", bridge_fields)


if __name__ == "__main__":
    unittest.main()
