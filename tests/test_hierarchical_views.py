from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import HierarchicalSubstrateConfig
from open_predictive_coder.hierarchical import HierarchicalSubstrate
from open_predictive_coder.hierarchical_views import HierarchicalFeatureView


def small_config() -> HierarchicalSubstrateConfig:
    return HierarchicalSubstrateConfig(
        fast_size=10,
        mid_size=14,
        slow_size=18,
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
        slow_update_stride=3,
        seed=17,
    )


class HierarchicalFeatureViewTests(unittest.TestCase):
    def test_bank_slices_match_config(self) -> None:
        config = small_config()
        view = HierarchicalFeatureView(config)

        self.assertEqual(view.bank_slices.fast, slice(0, 10))
        self.assertEqual(view.bank_slices.mid, slice(10, 24))
        self.assertEqual(view.bank_slices.slow, slice(24, 42))

    def test_split_and_pooled_summary_shapes(self) -> None:
        config = small_config()
        view = HierarchicalFeatureView(config)
        state = np.arange(config.state_dim, dtype=np.float64)

        fast, mid, slow = view.split(state)
        summary = view.pooled_summary(state)
        features = view.encode(state)

        self.assertEqual(fast.shape, (10,))
        self.assertEqual(mid.shape, (14,))
        self.assertEqual(slow.shape, (18,))
        self.assertEqual(summary.fast_mean.shape, (1,))
        self.assertEqual(summary.mid_mean.shape, (1,))
        self.assertEqual(summary.slow_mean.shape, (1,))
        self.assertEqual(features.shape, (view.feature_dim,))

    def test_predictive_features_are_finite_over_rollout(self) -> None:
        config = small_config()
        substrate = HierarchicalSubstrate(config)
        view = HierarchicalFeatureView(config)
        state = substrate.initial_state()
        previous = None

        for token in [0, 1, 2, 3, 4, 5, 6, 7] * 10:
            next_state = substrate.step(state, token)
            features = view.predictive_features(next_state, previous_state=previous)
            self.assertEqual(features.shape, (view.predictive_dim,))
            self.assertTrue(np.all(np.isfinite(features)))
            previous = state
            state = next_state


if __name__ == "__main__":
    unittest.main()
