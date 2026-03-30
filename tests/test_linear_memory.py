from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import LinearMemoryConfig, LinearMemorySubstrate


class LinearMemoryTests(unittest.TestCase):
    def test_state_dim_matches_bank_shape(self) -> None:
        config = LinearMemoryConfig(embedding_dim=4, decays=(0.5, 0.8, 0.95))
        substrate = LinearMemorySubstrate(config)
        state = substrate.initial_state()
        view = substrate.state_view(state)
        self.assertEqual(state.shape, (12,))
        self.assertEqual(view.shape, (3, 4))

    def test_step_applies_same_token_to_all_decay_banks(self) -> None:
        config = LinearMemoryConfig(embedding_dim=3, decays=(0.25, 0.75), seed=13)
        substrate = LinearMemorySubstrate(config)
        state = substrate.initial_state()
        next_state = substrate.step(state, 65)
        next_view = substrate.state_view(next_state)
        self.assertTrue(np.allclose(next_view[0], next_view[1]))

        second_state = substrate.step(next_state, 66)
        second_view = substrate.state_view(second_state)
        self.assertFalse(np.allclose(second_view[0], second_view[1]))


if __name__ == "__main__":
    unittest.main()
