from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import (
    DelayLineConfig,
    DelayLineSubstrate,
    FrozenReadoutExpert,
    LinearMemoryConfig,
    LinearMemoryFeatureView,
    LinearMemorySubstrate,
    bits_per_byte_from_probabilities,
)


class ExpertPrimitiveTests(unittest.TestCase):
    def test_linear_memory_feature_view_matches_expected_dim(self) -> None:
        substrate = LinearMemorySubstrate(LinearMemoryConfig(embedding_dim=4, decays=(0.4, 0.8, 0.95)))
        view = LinearMemoryFeatureView(substrate)
        encoded = view.encode(substrate.initial_state())
        self.assertEqual(encoded.shape, (substrate.state_dim + (3 * substrate.bank_count),))

    def test_frozen_readout_expert_fit_and_score(self) -> None:
        substrate = DelayLineSubstrate(DelayLineConfig(history_length=3, embedding_dim=4, seed=17))

        def feature_fn(state: np.ndarray, previous_state: np.ndarray | None) -> np.ndarray:
            del previous_state
            return state

        expert = FrozenReadoutExpert(
            name="delay",
            substrate=substrate,
            feature_dim=substrate.state_dim,
            vocabulary_size=256,
            feature_fn=feature_fn,
            alpha=1e-4,
        )
        fit = expert.fit("abababababab")
        score = expert.score("abababab")
        self.assertEqual(fit.sequences, 1)
        self.assertGreater(fit.tokens, 0)
        self.assertGreater(score.tokens, 0)
        self.assertTrue(np.isfinite(fit.bits_per_byte))
        self.assertTrue(np.isfinite(score.bits_per_byte))

    def test_bits_from_probabilities_matches_simple_case(self) -> None:
        probabilities = np.asarray(
            [
                [0.75, 0.25],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )
        targets = np.asarray([0, 1], dtype=np.int64)
        value = bits_per_byte_from_probabilities(probabilities, targets)
        self.assertLess(value, 1.0)


if __name__ == "__main__":
    unittest.main()
