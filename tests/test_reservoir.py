from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import ReservoirConfig
from open_predictive_coder.reservoir import EchoStateReservoir, build_recurrent_matrix, spectral_radius


class ReservoirTests(unittest.TestCase):
    def test_recurrent_matrix_respects_target_radius(self) -> None:
        config = ReservoirConfig(size=48, connectivity=0.2, spectral_radius=0.85, seed=5)
        matrix = build_recurrent_matrix(config)
        radius = spectral_radius(matrix)
        self.assertLess(abs(radius - 0.85), 0.08)

    def test_reservoir_step_stays_finite(self) -> None:
        reservoir = EchoStateReservoir(
            ReservoirConfig(size=64, connectivity=0.15, spectral_radius=0.9, leak=0.3, seed=9)
        )
        state = reservoir.initial_state()
        for token in [0, 1, 2, 3] * 25:
            state = reservoir.step(state, token)
        self.assertEqual(state.shape, (64,))
        self.assertTrue(np.all(np.isfinite(state)))


if __name__ == "__main__":
    unittest.main()
