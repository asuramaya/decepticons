from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.predictive_surprise import (
    PredictionState,
    PredictiveSurpriseConfig,
    PredictiveSurpriseController,
)


class PredictiveSurpriseTests(unittest.TestCase):
    def test_observe_computes_residual_and_surprise(self) -> None:
        controller = PredictiveSurpriseController(
            PredictiveSurpriseConfig(feature_mode="surprise")
        )
        state = controller.observe([1.0, 2.0, 3.0], [1.5, 1.5, 4.0], step=4, name="demo")
        self.assertIsInstance(state, PredictionState)
        np.testing.assert_allclose(state.residual, np.asarray([0.5, -0.5, 1.0], dtype=np.float64))
        np.testing.assert_allclose(state.surprise, np.asarray([0.5, 0.5, 1.0], dtype=np.float64))
        self.assertEqual(state.step, 4)
        self.assertEqual(state.summary.name, "demo")

    def test_feature_vector_is_compact_and_deterministic(self) -> None:
        controller = PredictiveSurpriseController(
            PredictiveSurpriseConfig(feature_mode="residual")
        )
        state = controller.observe([0.0, 1.0], [1.0, 3.0])
        feature = controller.feature_vector(state)
        self.assertEqual(feature.shape, (8,))
        self.assertEqual(controller.feature_dim, 8)
        self.assertGreater(feature[-1], 0.0)

    def test_shape_mismatch_raises(self) -> None:
        controller = PredictiveSurpriseController()
        with self.assertRaises(ValueError):
            controller.observe([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
