from __future__ import annotations

import inspect
import unittest
from dataclasses import fields

import decepticons as opc


def _field_names(cls: type[object]) -> set[str]:
    return {field.name for field in fields(cls)}


class NoncausalBridgeBoundaryTests(unittest.TestCase):
    def test_public_kernel_exports_only_generic_noncausal_contract(self) -> None:
        public_names = set(dir(opc))
        noncausal_names = {name for name in public_names if name.startswith("Noncausal")}
        allowed_noncausal_names = {
            "NoncausalReconstructiveAdapter",
            "NoncausalReconstructiveConfig",
            "NoncausalReconstructiveFitReport",
            "NoncausalReconstructiveReport",
            "NoncausalReconstructiveTrace",
        }

        self.assertEqual(noncausal_names, allowed_noncausal_names)
        self.assertTrue(hasattr(opc, "NoncausalReconstructiveAdapter"))
        self.assertFalse(hasattr(opc, "NoncausalFieldReconstructionModel"))
        self.assertFalse(any("field" in name.lower() for name in noncausal_names))
        self.assertFalse(any("payload" in name.lower() for name in noncausal_names))

    def test_causal_oracle_and_bridge_constructors_stay_disjoint(self) -> None:
        causal_params = tuple(inspect.signature(opc.CausalPredictiveAdapter.__init__).parameters)
        noncausal_params = tuple(inspect.signature(opc.NoncausalReconstructiveAdapter.__init__).parameters)
        oracle_params = tuple(inspect.signature(opc.OracleAnalysisAdapter.__init__).parameters)
        bridge_params = tuple(inspect.signature(opc.BridgeExportAdapter.__init__).parameters)

        self.assertNotIn("bridge", causal_params)
        self.assertNotIn("bridge_features", causal_params)
        self.assertNotIn("right_context", causal_params)
        self.assertNotIn("noncausal", causal_params)

        self.assertNotIn("bridge", oracle_params)
        self.assertNotIn("bridge_features", oracle_params)
        self.assertNotIn("exact_context", oracle_params)
        self.assertNotIn("noncausal", oracle_params)

        self.assertNotIn("payload", noncausal_params)
        self.assertNotIn("payload_policy", noncausal_params)
        self.assertNotIn("bridge", noncausal_params)
        self.assertNotIn("teacher", noncausal_params)

        self.assertNotIn("exact_context", bridge_params)
        self.assertNotIn("experts", bridge_params)
        self.assertNotIn("bidirectional", bridge_params)
        self.assertNotIn("noncausal", bridge_params)

    def test_surface_reports_do_not_cross_the_family_boundary(self) -> None:
        causal_fit_fields = _field_names(opc.CausalPredictiveFitReport)
        causal_score_fields = _field_names(opc.CausalPredictiveScore)
        noncausal_report_fields = _field_names(opc.NoncausalReconstructiveReport)
        oracle_report_fields = _field_names(opc.OracleAnalysisReport)
        bridge_report_fields = _field_names(opc.BridgeExportReport)

        causal_forbidden = {
            "bridge_features",
            "bridge_export",
            "bidirectional_context",
            "left_context_count",
            "right_context_count",
            "pair_context_count",
            "neighborhoods",
        }
        oracle_forbidden = {
            "bridge_features",
            "bridge_export",
            "candidate4",
            "candidate_count",
            "exact_support",
            "ngram_bits_per_byte",
        }
        noncausal_forbidden = {
            "payload",
            "payload_bytes",
            "payload_format",
            "bridge_export",
            "teacher",
            "attack",
            "oracle",
        }
        bridge_forbidden = {
            "exact_support",
            "left_context_count",
            "right_context_count",
            "pair_context_count",
            "neighborhoods",
            "ngram_bits_per_byte",
        }

        self.assertFalse(causal_fit_fields & causal_forbidden)
        self.assertFalse(causal_score_fields & causal_forbidden)
        self.assertFalse(noncausal_report_fields & noncausal_forbidden)
        self.assertFalse(oracle_report_fields & oracle_forbidden)
        self.assertFalse(bridge_report_fields & bridge_forbidden)

        self.assertIn("reconstructed_text", noncausal_report_fields)
        self.assertIn("bidirectional_context", noncausal_report_fields)
        self.assertIn("accounting", noncausal_report_fields)
        self.assertIn("features", bridge_report_fields)
        self.assertIn("accounting", bridge_report_fields)
        self.assertIn("mean_candidate4", bridge_report_fields)
        self.assertIn("mean_agreement_mass", bridge_report_fields)
        self.assertIn("oracle_preference_rate", oracle_report_fields)
        self.assertIn("points", oracle_report_fields)


if __name__ == "__main__":
    unittest.main()
