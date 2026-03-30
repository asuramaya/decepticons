from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
import unittest

import open_predictive_coder as opc


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
        self.assertTrue(hasattr(opc, "bits_per_byte_from_probabilities"))

        self.assertFalse(hasattr(opc, "ExpertMixtureModel"))
        self.assertFalse(hasattr(opc, "ResidualCorrectionModel"))
        self.assertFalse(hasattr(opc, "Conker1Replica"))
        self.assertFalse(hasattr(opc, "Conker2Replica"))
        self.assertFalse(hasattr(opc, "Conker3Replica"))
        self.assertFalse(hasattr(opc, "CarvingMachineKernelAdapter"))
        self.assertFalse(hasattr(opc, "GateSource"))

    def test_conker_shared_keeps_policy_in_project_code(self) -> None:
        module = load_module("conker_shared_test", "conker_shared.py")
        self.assertTrue(hasattr(module, "ExpertMixtureModel"))
        self.assertTrue(hasattr(module, "ResidualCorrectionModel"))
        self.assertTrue(hasattr(module, "build_linear_memory_expert"))
        self.assertTrue(hasattr(module, "build_delay_local_expert"))
        self.assertTrue(hasattr(module, "build_echo_correction_expert"))

    def test_carving_machine_like_keeps_task_policy_local(self) -> None:
        module = load_module("carving_machine_like_model_test", "carving_machine_like/model.py")
        self.assertTrue(hasattr(module, "CarvingMachineKernelAdapter"))
        self.assertTrue(hasattr(module, "CarvingMachineKernelConfig"))
        self.assertTrue(hasattr(module, "GateSource"))

        self.assertFalse(hasattr(opc, "CarvingMachineKernelAdapter"))
        self.assertFalse(hasattr(opc, "CarvingMachineKernelConfig"))
        self.assertFalse(hasattr(opc, "GateSource"))


if __name__ == "__main__":
    unittest.main()
