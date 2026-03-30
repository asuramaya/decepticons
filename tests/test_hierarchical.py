from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import HierarchicalSubstrateConfig
from open_predictive_coder.hierarchical import HierarchicalSubstrate


def small_config(slow_stride: int = 2) -> HierarchicalSubstrateConfig:
    return HierarchicalSubstrateConfig(
        fast_size=12,
        mid_size=16,
        slow_size=20,
        vocabulary_size=32,
        fast_connectivity=0.2,
        mid_connectivity=0.15,
        slow_connectivity=0.1,
        fast_spectral_radius=0.8,
        mid_spectral_radius=0.9,
        slow_spectral_radius=0.95,
        fast_leak=0.4,
        mid_leak=0.3,
        slow_leak=0.2,
        input_scale=0.15,
        upward_scale=0.08,
        slow_update_stride=slow_stride,
        seed=17,
    )


class HierarchicalSubstrateTests(unittest.TestCase):
    def test_state_shape_and_zeros(self) -> None:
        substrate = HierarchicalSubstrate(config=small_config())
        state = substrate.initial_state()

        self.assertEqual(state.shape, (48,))
        self.assertTrue(np.all(state == 0.0))
        self.assertEqual(substrate.state_slices.fast, slice(0, 12))
        self.assertEqual(substrate.state_slices.mid, slice(12, 28))
        self.assertEqual(substrate.state_slices.slow, slice(28, 48))

    def test_finite_rollout(self) -> None:
        substrate = HierarchicalSubstrate(config=small_config())
        state = substrate.initial_state()
        for token in [0, 1, 2, 3, 4, 5, 6, 7] * 12:
            state = substrate.step(state, token)
        self.assertEqual(state.shape, (48,))
        self.assertTrue(np.all(np.isfinite(state)))

    def test_slow_stride_holds_bank_when_not_scheduled(self) -> None:
        substrate = HierarchicalSubstrate(config=small_config(slow_stride=3))
        state0 = substrate.initial_state()
        state1 = substrate.step(state0, 5)
        state2 = substrate.step(state1, 6)
        state3 = substrate.step(state2, 7)

        slow0 = state0[28:]
        slow1 = state1[28:]
        slow2 = state2[28:]
        slow3 = state3[28:]

        self.assertTrue(np.allclose(slow0, 0.0))
        self.assertTrue(np.allclose(slow1, slow0))
        self.assertTrue(np.allclose(slow2, slow1))
        self.assertFalse(np.allclose(slow3, slow2))


if __name__ == "__main__":
    unittest.main()
