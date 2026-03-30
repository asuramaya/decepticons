from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.control import ControllerSummary, ControllerSummaryBuilder
from open_predictive_coder.routing import RoutingConfig, SummaryRouter


class RoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        builder = ControllerSummaryBuilder()
        self.low = builder.encode([0.1, 0.1, 0.1], name="low")
        self.mid = builder.encode([0.5, 0.4, 0.3], name="mid")
        self.high = builder.encode([2.0, 1.5, 1.0], name="high")

    def test_equal_mode_is_uniform(self) -> None:
        router = SummaryRouter(RoutingConfig(mode="equal"))
        decision = router.route([self.low, self.mid, self.high])

        self.assertEqual(decision.mode, "equal")
        self.assertTrue(np.allclose(decision.weights, [1 / 3, 1 / 3, 1 / 3]))
        self.assertEqual(decision.selected_index, 0)

    def test_static_mode_uses_biases_and_normalizes(self) -> None:
        router = SummaryRouter(
            RoutingConfig(
                mode="static",
                static_logits=(2.0, 0.0, -1.0),
            )
        )
        decision = router.route([self.low, self.mid, self.high], names=("low", "mid", "high"))

        self.assertEqual(decision.route_names, ("low", "mid", "high"))
        self.assertAlmostEqual(float(np.sum(decision.weights)), 1.0, places=10)
        self.assertEqual(decision.selected_index, 0)
        self.assertGreater(decision.weights[0], decision.weights[1])
        self.assertGreater(decision.weights[1], decision.weights[2])

    def test_projection_mode_prefers_high_magnitude_summary(self) -> None:
        router = SummaryRouter(
            RoutingConfig(
                mode="projection",
                projection_weights=(1.0, 1.0, 1.0),
            )
        )
        decision = router.route([self.low, self.mid, self.high])

        self.assertAlmostEqual(float(np.sum(decision.weights)), 1.0, places=10)
        self.assertEqual(decision.selected_index, 2)
        self.assertGreater(decision.weights[2], decision.weights[1])
        self.assertGreater(decision.weights[2], decision.weights[0])

    def test_rejects_mismatched_summary_shapes(self) -> None:
        router = SummaryRouter()
        bigger = ControllerSummary(values=np.array([1.0, 2.0, 3.0, 4.0]), name="bigger")

        with self.assertRaises(ValueError):
            router.route([self.low, bigger])


if __name__ == "__main__":
    unittest.main()
