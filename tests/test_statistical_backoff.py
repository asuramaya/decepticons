from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import (
    NgramMemoryConfig,
    StatisticalBackoffConfig,
    StatisticalBackoffMemory,
)


class StatisticalBackoffTests(unittest.TestCase):
    def test_fit_learns_simplex_mixture(self) -> None:
        model = StatisticalBackoffMemory(
            StatisticalBackoffConfig(
                ngram=NgramMemoryConfig(vocabulary_size=8, trigram_bucket_count=32),
                mixture_steps=64,
            )
        )
        sequence = np.asarray([0, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)

        fit = model.fit(sequence)

        self.assertGreater(fit.ngram.tokens, 0)
        self.assertEqual(fit.mixture_weights.shape, (3,))
        self.assertTrue(np.all(fit.mixture_weights >= 0.0))
        self.assertAlmostEqual(float(np.sum(fit.mixture_weights)), 1.0, places=9)
        self.assertGreaterEqual(fit.unigram_bits_per_token, 0.0)
        self.assertGreaterEqual(fit.bigram_bits_per_token, 0.0)
        self.assertGreaterEqual(fit.trigram_bits_per_token, 0.0)
        self.assertGreaterEqual(fit.mixed_bits_per_token, 0.0)

    def test_predict_uses_prefix_fallbacks(self) -> None:
        model = StatisticalBackoffMemory(
            StatisticalBackoffConfig(
                ngram=NgramMemoryConfig(vocabulary_size=8, trigram_bucket_count=32),
                mixture_steps=0,
            )
        )
        model.fit([0, 1, 2, 1, 2, 1])

        empty = model.predict(np.asarray([], dtype=np.int64))
        one = model.predict(np.asarray([1], dtype=np.int64))
        two = model.predict(np.asarray([1, 2], dtype=np.int64))

        self.assertEqual(empty.context_order, 0)
        self.assertTrue(np.allclose(empty.unigram_probs, empty.bigram_probs))
        self.assertTrue(np.allclose(empty.bigram_probs, empty.trigram_probs))
        self.assertEqual(one.context_order, 1)
        self.assertTrue(np.allclose(one.bigram_probs, one.trigram_probs))
        self.assertEqual(two.context_order, 2)
        self.assertEqual(two.highest_order_probs.shape, (8,))
        self.assertAlmostEqual(float(np.sum(two.mixed_probs)), 1.0, places=9)

    def test_trace_and_score_are_normalized(self) -> None:
        model = StatisticalBackoffMemory(
            StatisticalBackoffConfig(
                ngram=NgramMemoryConfig(vocabulary_size=16, trigram_bucket_count=64),
                mixture_steps=32,
            )
        )
        sequence = np.asarray([10, 11, 12, 11, 12, 11, 12, 10], dtype=np.int64)
        model.fit(sequence)

        trace = model.trace(sequence)
        score = model.score(sequence)

        self.assertEqual(trace.steps, sequence.size)
        self.assertEqual(trace.unigram_probs.shape, (sequence.size, 16))
        self.assertEqual(trace.bigram_probs.shape, trace.unigram_probs.shape)
        self.assertEqual(trace.trigram_probs.shape, trace.unigram_probs.shape)
        self.assertEqual(trace.mixed_probs.shape, trace.unigram_probs.shape)
        self.assertTrue(np.allclose(np.sum(trace.mixed_probs, axis=1), 1.0))
        self.assertGreaterEqual(score.unigram_bits_per_token, 0.0)
        self.assertGreaterEqual(score.bigram_bits_per_token, 0.0)
        self.assertGreaterEqual(score.trigram_bits_per_token, 0.0)
        self.assertGreaterEqual(score.highest_order_bits_per_token, 0.0)
        self.assertGreaterEqual(score.mixed_bits_per_token, 0.0)
        self.assertEqual(score.mixture_weights.shape, (3,))

    def test_predictive_distribution_modes(self) -> None:
        model = StatisticalBackoffMemory(
            StatisticalBackoffConfig(
                ngram=NgramMemoryConfig(vocabulary_size=8, trigram_bucket_count=32),
                mixture_steps=0,
            )
        )
        model.fit([0, 1, 2, 3, 2, 3, 2])
        prefix = np.asarray([2, 3], dtype=np.int64)

        mixed = model.predictive_distribution(prefix, mode="mixed")
        highest = model.predictive_distribution(prefix, mode="highest_order")
        unigram = model.predictive_distribution(prefix, mode="unigram")

        self.assertEqual(mixed.shape, (8,))
        self.assertEqual(highest.shape, (8,))
        self.assertEqual(unigram.shape, (8,))
        self.assertAlmostEqual(float(np.sum(mixed)), 1.0, places=9)
        with self.assertRaises(ValueError):
            model.predictive_distribution(prefix, mode="unknown")


if __name__ == "__main__":
    unittest.main()
