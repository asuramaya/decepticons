from __future__ import annotations

import inspect
import sys
import unittest
from importlib import util
from pathlib import Path

import decepticons as opc

PROJECTS_ROOT = Path(__file__).resolve().parents[1] / "examples" / "projects"


def load_module(module_name: str, relative_path: str):
    path = PROJECTS_ROOT / relative_path
    spec = util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class KernelProjectSplitTests(unittest.TestCase):
    def test_kernel_exports_only_reusable_mechanisms(self) -> None:
        self.assertTrue(hasattr(opc, "LinearMemorySubstrate"))
        self.assertTrue(hasattr(opc, "LinearMemoryFeatureView"))
        self.assertTrue(hasattr(opc, "FrozenReadoutExpert"))
        self.assertTrue(hasattr(opc, "CausalPredictiveAdapter"))
        self.assertTrue(hasattr(opc, "OracleAnalysisAdapter"))
        self.assertTrue(hasattr(opc, "bits_per_byte_from_probabilities"))

        self.assertFalse(hasattr(opc, "ExpertMixtureModel"))
        self.assertFalse(hasattr(opc, "ResidualCorrectionModel"))
        self.assertFalse(hasattr(opc, "MemoryStabilityModel"))
        self.assertFalse(hasattr(opc, "LinearCorrectionModel"))
        self.assertFalse(hasattr(opc, "ResidualRepairModel"))
        self.assertFalse(hasattr(opc, "HierarchicalPredictiveModel"))
        self.assertFalse(hasattr(opc, "GateSource"))

    def test_causal_descendant_helpers_keep_policy_in_project_code(self) -> None:
        module = load_module("causal_shared_test", "causal/shared.py")
        self.assertTrue(hasattr(module, "ExpertMixtureModel"))
        self.assertTrue(hasattr(module, "ResidualCorrectionModel"))
        self.assertTrue(hasattr(module, "build_linear_memory_expert"))
        self.assertTrue(hasattr(module, "build_delay_local_expert"))
        self.assertTrue(hasattr(module, "build_echo_correction_expert"))

    def test_hierarchical_predictive_example_keeps_task_policy_local(self) -> None:
        module = load_module("hierarchical_predictive_model_test", "ancestor/hierarchical_predictive/model.py")
        self.assertTrue(hasattr(module, "HierarchicalPredictiveModel"))
        self.assertTrue(hasattr(module, "HierarchicalPredictiveConfig"))
        self.assertTrue(hasattr(module, "GateSource"))

        self.assertFalse(hasattr(opc, "HierarchicalPredictiveModel"))
        self.assertFalse(hasattr(opc, "HierarchicalPredictiveConfig"))
        self.assertFalse(hasattr(opc, "GateSource"))

    def test_bidirectional_analysis_example_thins_around_kernel_adapter(self) -> None:
        module = load_module("bidirectional_analysis_model_test", "oracle/bidirectional_analysis/model.py")
        self.assertTrue(hasattr(module, "BidirectionalAnalysisModel"))
        self.assertTrue(hasattr(module, "BidirectionalAnalysisConfig"))
        self.assertFalse(hasattr(opc, "BidirectionalAnalysisModel"))

    def test_causal_adapter_construction_stays_free_of_bridge_and_bidirectional_hooks(self) -> None:
        params = tuple(inspect.signature(opc.CausalPredictiveAdapter.__init__).parameters)

        self.assertEqual(
            params[1:],
            ("exact_context", "experts", "ngram_memory", "mixer", "artifact_name", "metadata"),
        )
        self.assertNotIn("bridge", params)
        self.assertNotIn("bridge_features", params)
        self.assertNotIn("bidirectional", params)
        self.assertNotIn("right_context", params)
        self.assertNotIn("left_context", params)


if __name__ == "__main__":
    unittest.main()
