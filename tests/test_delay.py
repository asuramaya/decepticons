from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder.config import DelayLineConfig
from open_predictive_coder.delay import DelayLineSubstrate


class DelayLineSubstrateTests(unittest.TestCase):
    def test_initial_state_shape_and_zeros(self) -> None:
        substrate = DelayLineSubstrate(
            DelayLineConfig(history_length=3, embedding_dim=5, vocabulary_size=16, seed=11)
        )
        state = substrate.initial_state()
        self.assertEqual(state.shape, (15,))
        self.assertTrue(np.all(state == 0.0))

    def test_step_inserts_new_embedding_and_shifts_history(self) -> None:
        substrate = DelayLineSubstrate(
            DelayLineConfig(history_length=3, embedding_dim=4, vocabulary_size=32, seed=19)
        )
        state = substrate.initial_state()

        first = substrate.step(state, 7)
        second = substrate.step(first, 13)

        first_history = substrate.history_view(first)
        second_history = substrate.history_view(second)

        np.testing.assert_allclose(first_history[0], substrate._token_embeddings[7])
        np.testing.assert_allclose(first_history[1], 0.0)
        np.testing.assert_allclose(first_history[2], 0.0)

        np.testing.assert_allclose(second_history[0], substrate._token_embeddings[13])
        np.testing.assert_allclose(second_history[1], substrate._token_embeddings[7])
        np.testing.assert_allclose(second_history[2], 0.0)

    def test_step_remains_finite_over_long_rollout(self) -> None:
        substrate = DelayLineSubstrate(
            DelayLineConfig(history_length=4, embedding_dim=6, vocabulary_size=64, seed=23, decay=0.9)
        )
        state = substrate.initial_state()
        for token in [0, 1, 2, 3, 4, 5, 6, 7] * 20:
            state = substrate.step(state, token)
        self.assertEqual(state.shape, (24,))
        self.assertTrue(np.all(np.isfinite(state)))


if __name__ == "__main__":
    unittest.main()
