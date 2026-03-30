from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.control import ControllerSummary
from open_predictive_coder.modulation import HormoneModulationConfig, HormoneModulator


class HormoneModulationTests(unittest.TestCase):
    def test_project_emits_bounded_outputs(self) -> None:
        modulator = HormoneModulator(
            summary_dim=3,
            config=HormoneModulationConfig(
                hormone_count=4,
                output_indices=(0, 2),
                output_biases=(0.0, 0.25),
                output_scales=(1.0, 0.5),
                seed=17,
            ),
        )
        state = modulator.project(ControllerSummary(np.asarray([0.2, -0.1, 0.5], dtype=np.float64), name="slow"))
        self.assertEqual(state.hormones.shape, (4,))
        self.assertEqual(state.outputs.shape, (2,))
        self.assertTrue(np.all(state.outputs >= 0.0))
        self.assertTrue(np.all(state.outputs <= 1.0))
        self.assertTrue(state.refreshed)

    def test_refresh_stride_preserves_previous_values(self) -> None:
        modulator = HormoneModulator(summary_dim=2, config=HormoneModulationConfig(refresh_stride=2, seed=23))
        initial = modulator.initial_state()
        first = modulator.advance(initial, [1.0, 0.0], step=0)
        second = modulator.advance(first, [0.0, 1.0], step=1)
        self.assertTrue(second.refreshed is False)
        self.assertTrue(np.allclose(first.hormones, second.hormones))
        self.assertTrue(np.allclose(first.outputs, second.outputs))

    def test_summary_dimension_mismatch_is_rejected(self) -> None:
        modulator = HormoneModulator(summary_dim=2)
        with self.assertRaises(ValueError):
            modulator.project([1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
