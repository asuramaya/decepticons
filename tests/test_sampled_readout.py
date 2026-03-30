from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import SampledReadoutBandConfig, SampledReadoutConfig
from open_predictive_coder.sampled_readout import SampledMultiscaleReadout


class SampledReadoutTests(unittest.TestCase):
    def test_encode_concatenates_sampled_values_and_band_statistics(self) -> None:
        config = SampledReadoutConfig(
            state_dim=12,
            bands=(
                SampledReadoutBandConfig(
                    name="fast",
                    start=0,
                    stop=4,
                    sample_indices=(0, 2),
                    include_mean=True,
                    include_energy=True,
                    include_drift=True,
                ),
                SampledReadoutBandConfig(
                    name="mid",
                    start=4,
                    stop=8,
                    sample_indices=(1, 3),
                    include_mean=True,
                    include_energy=True,
                    include_drift=True,
                ),
                SampledReadoutBandConfig(
                    name="slow",
                    start=8,
                    stop=12,
                    include_mean=True,
                    include_energy=True,
                    include_drift=False,
                ),
            ),
            seed=13,
        )
        readout = SampledMultiscaleReadout(config)
        state = np.arange(12, dtype=np.float64)
        previous = state - 1.0

        summary = readout.summaries(state, previous_state=previous)
        features = readout.encode(state, previous_state=previous)

        self.assertEqual(readout.feature_dim, 16)
        self.assertEqual(features.shape, (16,))
        self.assertEqual(summary[0].name, "fast")
        np.testing.assert_array_equal(summary[0].indices, np.array([0, 2], dtype=np.int64))
        np.testing.assert_allclose(
            features,
            np.array(
                [
                    0.0,
                    2.0,
                    1.5,
                    3.5,
                    1.0,
                    5.0,
                    7.0,
                    5.5,
                    31.5,
                    1.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    9.5,
                    91.5,
                ],
                dtype=np.float64,
            ),
        )

    def test_seeded_sampling_is_deterministic(self) -> None:
        config = SampledReadoutConfig(
            state_dim=10,
            bands=(
                SampledReadoutBandConfig(
                    name="band",
                    start=0,
                    stop=10,
                    sample_count=3,
                    include_mean=False,
                    include_energy=False,
                    include_drift=False,
                ),
            ),
            seed=21,
        )
        left = SampledMultiscaleReadout(config)
        right = SampledMultiscaleReadout(config)
        state = np.linspace(0.0, 9.0, num=10, dtype=np.float64)

        np.testing.assert_array_equal(left.band_indices[0], right.band_indices[0])
        np.testing.assert_allclose(left.encode(state), right.encode(state))

    def test_rank_and_shape_mismatches_raise(self) -> None:
        config = SampledReadoutConfig(
            state_dim=6,
            bands=(
                SampledReadoutBandConfig(
                    name="band",
                    start=0,
                    stop=6,
                    sample_count=2,
                ),
            ),
        )
        readout = SampledMultiscaleReadout(config)

        with self.assertRaises(ValueError):
            readout.encode(np.zeros((2, 3), dtype=np.float64))
        with self.assertRaises(ValueError):
            readout.encode(np.zeros((5,), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
