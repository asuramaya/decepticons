from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import (
    AdaptiveSegmenter,
    ByteLatentFeatureView,
    ByteLatentPredictiveCoder,
    DelayLineConfig,
    LatentCommitter,
    OpenPredictiveCoderConfig,
)
from open_predictive_coder.config import LatentConfig, ReservoirConfig, SegmenterConfig


def small_config() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        segmenter=SegmenterConfig(mode="fixed", patch_size=3, min_patch_size=2, max_patch_size=6, novelty_threshold=0.2),
        reservoir=ReservoirConfig(size=64, connectivity=0.15, spectral_radius=0.9, leak=0.3, seed=13),
        latent=LatentConfig(latent_dim=16, global_dim=16, reservoir_features=12, readout_l2=1e-4),
    )


class PrimitiveExtractionTests(unittest.TestCase):
    def test_feature_view_matches_config_dim(self) -> None:
        config = small_config()
        self.assertEqual(ByteLatentFeatureView.feature_dim(config.latent), config.feature_dim)

    def test_committer_emits_feature_sized_observation(self) -> None:
        config = small_config()
        committer = LatentCommitter(
            config=config.latent,
            substrate_size=config.reservoir.size,
            seed=config.reservoir.seed + 101,
        )
        segmenter = AdaptiveSegmenter(config.segmenter)
        feature_view = ByteLatentFeatureView(max_patch_size=config.segmenter.max_patch_size)
        state = committer.initial_state()
        substrate_state = np.linspace(-1.0, 1.0, config.reservoir.size, dtype=np.float64)

        observation = None
        for _ in range(config.segmenter.patch_size):
            local_view = committer.sample(substrate_state)
            observation = committer.step(state, local_view, segmenter)

        assert observation is not None
        feature = feature_view.encode(observation)
        self.assertEqual(feature.shape, (config.feature_dim,))
        self.assertTrue(observation.boundary)

    def test_byte_latent_adapter_trace_exposes_boundaries(self) -> None:
        model = ByteLatentPredictiveCoder(config=small_config())
        trace = model.trace("abcabcabcabc")
        self.assertEqual(trace.features.shape[0], len(trace.targets))
        self.assertEqual(trace.boundaries.shape, (len(trace.targets),))
        self.assertGreaterEqual(trace.patches, 1)

    def test_adapter_can_use_delay_substrate_via_config(self) -> None:
        config = OpenPredictiveCoderConfig(
            substrate_kind="delay",
            delay=DelayLineConfig(
                history_length=3,
                embedding_dim=8,
                vocabulary_size=256,
                seed=19,
            ),
            segmenter=SegmenterConfig(mode="fixed", patch_size=3, min_patch_size=2, max_patch_size=6),
            latent=LatentConfig(latent_dim=8, global_dim=8, reservoir_features=8, readout_l2=1e-4),
        )
        model = ByteLatentPredictiveCoder(config=config)
        trace = model.trace("abababab")
        self.assertEqual(trace.features.shape[1], config.feature_dim)
        self.assertGreaterEqual(trace.patches, 1)


if __name__ == "__main__":
    unittest.main()
