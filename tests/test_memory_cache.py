from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import (
    ExactContextCache,
    ExactContextConfig,
    ExactContextMemory,
    StatisticalBackoffCache,
)


class MemoryCacheTests(unittest.TestCase):
    def test_exact_context_cache_exposes_active_prediction(self) -> None:
        cache = ExactContextCache(
            ExactContextMemory(ExactContextConfig(vocabulary_size=8, max_order=3, alpha=0.05))
        )
        cache.fit([0, 1, 2, 1, 2, 1, 2])

        summary = cache.prediction_summary([1, 2])

        self.assertEqual(summary.family, "exact_context")
        self.assertEqual(summary.highest_order_prediction.order, 2)
        self.assertEqual(summary.active_prediction.order, 2)
        self.assertEqual(summary.active_prediction.name, "exact2")
        self.assertAlmostEqual(float(np.sum(summary.predictive_distribution())), 1.0, places=9)

    def test_exact_context_cache_falls_back_when_inactive(self) -> None:
        cache = ExactContextCache(
            ExactContextMemory(ExactContextConfig(vocabulary_size=8, max_order=3, alpha=0.05))
        )
        cache.fit([0, 1, 2, 1, 2, 1, 2])

        summary = cache.prediction_summary([7, 7, 7])

        self.assertEqual(summary.highest_order_prediction.order, 3)
        self.assertEqual(summary.active_prediction.order, 0)
        self.assertEqual(summary.active_prediction.name, "unigram")

    def test_statistical_backoff_cache_exposes_order_records_and_mixed_mode(self) -> None:
        cache = StatisticalBackoffCache.from_vocabulary(16, trigram_bucket_count=64, mixture_steps=32)
        sequence = np.asarray([10, 11, 12, 11, 12, 11, 12, 10], dtype=np.int64)
        fit = cache.fit(sequence)
        summary = cache.prediction_summary([11, 12])

        self.assertGreater(fit.ngram.tokens, 0)
        self.assertEqual(summary.family, "statistical_backoff")
        self.assertEqual(len(summary.predictions), 3)
        self.assertEqual(summary.active_prediction.order, 2)
        self.assertEqual(summary.highest_order_prediction.order, 2)
        self.assertEqual(summary.mixture_weights.shape, (3,))
        self.assertAlmostEqual(float(np.sum(summary.mixture_weights)), 1.0, places=9)
        self.assertAlmostEqual(float(np.sum(summary.predictive_distribution(mode="mixed"))), 1.0, places=9)
        self.assertAlmostEqual(
            float(np.sum(summary.predictive_distribution(mode="highest_order"))),
            1.0,
            places=9,
        )

    def test_statistical_backoff_cache_falls_back_by_prefix_length(self) -> None:
        cache = StatisticalBackoffCache.from_vocabulary(8, trigram_bucket_count=32, mixture_steps=0)
        cache.fit([0, 1, 2, 1, 2, 1])

        empty = cache.prediction_summary([])
        one = cache.prediction_summary([1])

        self.assertEqual(empty.active_prediction.order, 0)
        self.assertEqual(one.active_prediction.order, 1)
        with self.assertRaises(ValueError):
            empty.predictive_distribution(mode="unknown")


if __name__ == "__main__":
    unittest.main()
