from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
import unittest

import numpy as np

from open_predictive_coder import ByteCodec, TrainModeConfig


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


class CarvingMachineLikeFidelityTests(unittest.TestCase):
    def test_routed_hormonal_trace_exposes_control_surfaces(self) -> None:
        module = load_module("carving_machine_like_fidelity_model", "carving_machine_like/model.py")
        config = module.CarvingMachineKernelConfig(
            gate_source="routed",
            train_mode=TrainModeConfig(state_mode="through_state", slow_update_stride=2, rollout_checkpoint_stride=4),
        )
        model = module.CarvingMachineKernelAdapter(config)

        trace = model.trace(ByteCodec.encode_text("carving machine keeps fast, mid, and slow state aligned."))

        self.assertEqual(trace.gates.shape[0], trace.hormones.shape[0])
        self.assertEqual(trace.gates.shape[0], trace.routes.shape[0])
        self.assertEqual(trace.hormones.shape[1], model.hormone_modulator.output_count)
        self.assertEqual(trace.checkpoint_steps, config.train_mode.resolve_rollout_checkpoints(trace.tokens - 1))
        self.assertTrue(np.all((trace.routes >= 0) & (trace.routes <= 1)))
        self.assertGreater(np.bincount(trace.routes, minlength=2).min(), 0)

    def test_routed_hormonal_model_fits_and_generates(self) -> None:
        module = load_module("carving_machine_like_fidelity_model_fit", "carving_machine_like/model.py")
        config = module.CarvingMachineKernelConfig(
            gate_source="routed",
            train_mode=TrainModeConfig(state_mode="through_state", slow_update_stride=2, rollout_checkpoint_stride=4),
        )
        model = module.CarvingMachineKernelAdapter(config)

        corpus = (
            "carving machine uses a rich substrate with a learned readout.\n"
            "hierarchical views expose fast, mid, and slow dynamics.\n"
        ) * 8
        fit_report = model.fit(corpus)
        score_report = model.score(corpus)
        sample = model.generate(ByteCodec.encode_text("hierarchical "), steps=16, greedy=True)

        self.assertGreater(fit_report.train_bits_per_byte, 0.0)
        self.assertGreater(score_report.bits_per_byte, 0.0)
        self.assertGreaterEqual(sample.size, len(ByteCodec.encode_text("hierarchical ")))


if __name__ == "__main__":
    unittest.main()
