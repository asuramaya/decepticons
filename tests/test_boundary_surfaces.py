from __future__ import annotations

import unittest
from dataclasses import fields

import decepticons as opc
from decepticons.bidirectional_context import BidirectionalContextConfig, BidirectionalContextStats
from decepticons.bridge_features import BridgeFeatureArrays, BridgeFeatureConfig
from decepticons.factories import create_substrate
from decepticons.ngram_memory import NgramMemoryConfig


def _field_names(cls: type[object]) -> set[str]:
    return {field.name for field in fields(cls)}


class BoundarySurfaceTests(unittest.TestCase):
    def test_substrate_factory_rejects_bridge_and_analysis_configs(self) -> None:
        for config in (
            BridgeFeatureConfig(),
            BidirectionalContextConfig(),
            NgramMemoryConfig(),
        ):
            with self.subTest(config_type=type(config).__name__), self.assertRaises(TypeError):
                create_substrate(config)

    def test_causal_reports_do_not_absorb_bridge_or_right_context_fields(self) -> None:
        fit_fields = _field_names(opc.CausalPredictiveFitReport)
        score_fields = _field_names(opc.CausalPredictiveScore)
        bridge_fields = _field_names(BridgeFeatureArrays)

        forbidden = {
            "bridge_features",
            "candidate4",
            "candidate_count",
            "candidate_sizes",
            "entropy",
            "agreement",
            "agreement_mass",
            "left_context",
            "left_context_count",
            "neighborhoods",
            "pair_context_count",
            "peak",
            "right_context",
            "right_context_count",
        }

        self.assertTrue(bridge_fields <= {"entropy", "peak", "candidate4", "agreement", "agreement_mass"})
        self.assertFalse(fit_fields & forbidden)
        self.assertFalse(score_fields & forbidden)

    def test_bidirectional_context_surface_stays_analysis_only(self) -> None:
        stats_fields = _field_names(BidirectionalContextStats)

        self.assertIn("left_context_count", stats_fields)
        self.assertIn("right_context_count", stats_fields)
        self.assertIn("pair_context_count", stats_fields)
        self.assertIn("neighborhoods", stats_fields)
        self.assertNotIn("bits_per_byte", stats_fields)
        self.assertNotIn("exact_support", stats_fields)


if __name__ == "__main__":
    unittest.main()
