from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import DelayLineConfig, MixedMemoryConfig, ReservoirConfig
from open_predictive_coder.mixed_memory import MixedMemorySubstrate


def small_config() -> MixedMemoryConfig:
    return MixedMemoryConfig(
        reservoir=ReservoirConfig(size=48, connectivity=0.2, spectral_radius=0.9, leak=0.3, seed=21),
        delay=DelayLineConfig(
            history_length=3,
            embedding_dim=5,
            vocabulary_size=16,
            input_scale=0.1,
            seed=11,
        ),
    )


class MixedMemorySubstrateTests(unittest.TestCase):
    def test_state_shape_and_dim(self) -> None:
        substrate = MixedMemorySubstrate(config=small_config())
        state = substrate.initial_state()

        self.assertEqual(substrate.state_dim, 48 + 15)
        self.assertEqual(state.shape, (substrate.state_dim,))
        self.assertTrue(np.all(state == 0.0))

    def test_step_keeps_state_finite(self) -> None:
        substrate = MixedMemorySubstrate(config=small_config())
        state = substrate.initial_state()

        for token in [0, 1, 2, 3, 4, 5, 6, 7] * 4:
            state = substrate.step(state, token)

        self.assertEqual(state.shape, (substrate.state_dim,))
        self.assertTrue(np.all(np.isfinite(state)))

    def test_delay_branch_shifts_previous_tokens(self) -> None:
        substrate = MixedMemorySubstrate(config=small_config())
        state0 = substrate.initial_state()
        state1 = substrate.step(state0, 2)
        state2 = substrate.step(state1, 7)

        delay0 = substrate.delay_view(state0)
        delay1 = substrate.delay_view(state1)
        delay2 = substrate.delay_view(state2)
        reservoir0 = substrate.reservoir_view(state0)
        reservoir1 = substrate.reservoir_view(state1)

        self.assertTrue(np.all(delay0 == 0.0))
        np.testing.assert_allclose(delay1[:5], substrate.delay._token_embeddings[2])
        self.assertTrue(np.allclose(delay1[5:], 0.0))
        np.testing.assert_allclose(delay2[:5], substrate.delay._token_embeddings[7])
        np.testing.assert_allclose(delay2[5:10], substrate.delay._token_embeddings[2])
        self.assertEqual(reservoir0.shape, (48,))
        self.assertEqual(reservoir1.shape, (48,))
        self.assertFalse(np.allclose(reservoir0, reservoir1))


if __name__ == "__main__":
    unittest.main()
