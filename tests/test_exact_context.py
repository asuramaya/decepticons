from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import (
    ExactContextConfig,
    ExactContextMemory,
    SupportMixConfig,
    SupportWeightedMixer,
)


class ExactContextTests(unittest.TestCase):
    def test_exact_context_prefers_repeating_continuation(self) -> None:
        memory = ExactContextMemory(ExactContextConfig(vocabulary_size=256, max_order=3, alpha=0.05))
        fit_report = memory.fit("abababababababab")
        self.assertEqual(fit_report.sequences, 1)
        self.assertGreater(fit_report.contexts_by_order[1], 0)

        predictions = {prediction.order: prediction for prediction in memory.experts("ab")}
        next_a = ord("a")
        next_b = ord("b")

        self.assertIn(1, predictions)
        self.assertIn(2, predictions)
        self.assertGreater(predictions[1].probabilities[next_a], predictions[1].probabilities[next_b])
        self.assertGreater(predictions[2].probabilities[next_a], predictions[2].probabilities[next_b])
        self.assertGreater(predictions[2].support, 0.0)

    def test_predictive_distribution_falls_back_to_unigram(self) -> None:
        memory = ExactContextMemory(ExactContextConfig(vocabulary_size=4, max_order=2, alpha=0.1))
        memory.fit([0, 1, 0, 1, 0, 1])
        probs = memory.predictive_distribution([3, 3])
        unigram = memory.unigram_probabilities()
        self.assertTrue(np.allclose(probs, unigram))

    def test_support_weighted_mixer_can_shift_mass_toward_exact_expert(self) -> None:
        memory = ExactContextMemory(ExactContextConfig(vocabulary_size=4, max_order=2, alpha=0.05))
        memory.fit([0, 1, 0, 1, 0, 1, 0, 1])
        experts = memory.experts([0, 1])
        mixer = SupportWeightedMixer(SupportMixConfig(base_bias=0.0, expert_bias=0.0, support_scale=1.0))
        base_probs = np.full(4, 0.25, dtype=np.float64)

        blend = mixer.mix(base_probs=base_probs, experts=experts, base_support=0.0)

        self.assertEqual(blend.component_names[0], "base")
        self.assertAlmostEqual(float(np.sum(blend.weights)), 1.0, places=6)
        self.assertEqual(int(np.argmax(blend.probabilities)), 0)
        self.assertGreater(blend.probabilities[0], base_probs[0])


if __name__ == "__main__":
    unittest.main()
