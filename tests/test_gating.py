from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.control import ControllerSummary, ControllerSummaryBuilder, ControllerSummaryConfig
from open_predictive_coder.gating import PathwayGateConfig, PathwayGateController


class PathwayGateTests(unittest.TestCase):
    def test_gate_values_increase_with_summary_signal(self) -> None:
        controller = PathwayGateController(
            PathwayGateConfig(
                refresh_stride=1,
                fast_to_mid_index=0,
                mid_to_slow_index=1,
                fast_to_mid_bias=0.0,
                fast_to_mid_scale=1.0,
                mid_to_slow_bias=0.0,
                mid_to_slow_scale=1.0,
            )
        )
        state = controller.initial_state()

        low = controller.advance(state, ControllerSummary(np.array([-1.0, -0.5]), name="slow"), step=0)
        high = controller.advance(low, ControllerSummary(np.array([1.0, 2.0]), name="slow"), step=1)

        self.assertTrue(0.0 <= low.values.fast_to_mid <= 1.0)
        self.assertTrue(0.0 <= low.values.mid_to_slow <= 1.0)
        self.assertGreater(high.values.fast_to_mid, low.values.fast_to_mid)
        self.assertGreater(high.values.mid_to_slow, low.values.mid_to_slow)

    def test_refresh_stride_reuses_cached_values(self) -> None:
        controller = PathwayGateController(
            PathwayGateConfig(
                refresh_stride=3,
                fast_to_mid_index=0,
                mid_to_slow_index=1,
                fast_to_mid_bias=-1.0,
                fast_to_mid_scale=2.0,
                mid_to_slow_bias=-1.0,
                mid_to_slow_scale=2.0,
            )
        )
        state = controller.initial_state()
        strong = ControllerSummary(np.array([3.0, 3.0, 3.0]), name="slow")
        weak = ControllerSummary(np.array([0.0, 0.0, 0.0]), name="slow")

        refreshed = controller.advance(state, strong, step=0)
        cached = controller.advance(refreshed, weak, step=1)
        refreshed_again = controller.advance(cached, weak, step=3)

        self.assertTrue(refreshed.values.refreshed)
        self.assertFalse(cached.values.refreshed)
        self.assertAlmostEqual(cached.values.fast_to_mid, refreshed.values.fast_to_mid, places=12)
        self.assertAlmostEqual(cached.values.mid_to_slow, refreshed.values.mid_to_slow, places=12)
        self.assertTrue(refreshed_again.values.refreshed)
        self.assertLess(refreshed_again.values.fast_to_mid, refreshed.values.fast_to_mid)
        self.assertLess(refreshed_again.values.mid_to_slow, refreshed.values.mid_to_slow)

    def test_configured_indices_drive_separate_gates(self) -> None:
        controller = PathwayGateController(
            PathwayGateConfig(
                refresh_stride=1,
                fast_to_mid_index=0,
                mid_to_slow_index=1,
                fast_to_mid_bias=0.0,
                fast_to_mid_scale=1.0,
                mid_to_slow_bias=0.0,
                mid_to_slow_scale=1.0,
            )
        )
        state = controller.initial_state()

        small = controller.advance(state, ControllerSummary(np.array([0.1, 0.1]), name="summary"), step=0)
        large = controller.advance(small, ControllerSummary(np.array([5.0, 6.0]), name="summary"), step=1)

        self.assertGreater(large.values.fast_to_mid, small.values.fast_to_mid)
        self.assertGreater(large.values.mid_to_slow, small.values.mid_to_slow)

    def test_raw_signal_uses_summary_builder(self) -> None:
        builder = ControllerSummaryBuilder(ControllerSummaryConfig(reduction="mean_abs", normalize=False))
        controller = PathwayGateController(
            PathwayGateConfig(
                refresh_stride=1,
                fast_to_mid_index=0,
                mid_to_slow_index=0,
                fast_to_mid_bias=0.0,
                fast_to_mid_scale=1.0,
                mid_to_slow_bias=0.0,
                mid_to_slow_scale=1.0,
                summary=ControllerSummaryConfig(reduction="mean_abs", normalize=False),
            ),
            summary_builder=builder,
        )
        state = controller.initial_state()

        raw_signal = np.asarray([[-2.0, 2.0], [-2.0, 2.0]], dtype=np.float64)
        advanced = controller.advance(state, raw_signal, step=0)

        self.assertTrue(advanced.values.refreshed)
        self.assertGreater(advanced.values.fast_to_mid, 0.5)
        self.assertGreater(advanced.values.mid_to_slow, 0.5)


if __name__ == "__main__":
    unittest.main()
