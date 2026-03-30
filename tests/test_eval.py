from __future__ import annotations

import unittest

import numpy as np

from open_predictive_coder import ByteLatentPredictiveCoder, OpenPredictiveCoderConfig
from open_predictive_coder.config import LatentConfig, ReservoirConfig, SegmenterConfig
from open_predictive_coder.eval import evaluate_rollout, score_next_step


def small_config() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        reservoir=ReservoirConfig(size=96, connectivity=0.12, spectral_radius=0.9, leak=0.35, seed=11),
        latent=LatentConfig(latent_dim=24, global_dim=24, reservoir_features=24, readout_l2=1e-5),
    )


class EvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ByteLatentPredictiveCoder(config=small_config())
        corpus = "predictive coding likes repeated structure.\n" * 64
        self.model.fit(corpus)

    def test_score_next_step_matches_model_score(self) -> None:
        sequence = "abababababab"
        report = score_next_step(self.model, sequence)
        direct = self.model.score(sequence)
        self.assertEqual(report.tokens, direct.tokens)
        self.assertAlmostEqual(report.bits_per_byte, direct.bits_per_byte, places=10)

    def test_teacher_forced_rollout_scores_full_sequence(self) -> None:
        prompt = "predictive "
        continuation = "coding likes repeated structure."
        evaluation = evaluate_rollout(
            self.model,
            prompt,
            continuation=continuation,
            mode="teacher_forced",
        )
        direct = self.model.score(prompt + continuation)

        self.assertEqual(evaluation.mode, "teacher_forced")
        self.assertEqual(bytes(evaluation.prompt_tokens), prompt.encode("utf-8"))
        self.assertEqual(bytes(evaluation.continuation_tokens), continuation.encode("utf-8"))
        self.assertEqual(evaluation.sequence_tokens.shape[0], len(prompt.encode("utf-8")) + len(continuation.encode("utf-8")))
        self.assertAlmostEqual(evaluation.bits_per_byte, direct.bits_per_byte, places=10)

    def test_closed_loop_rollout_scores_generated_sequence(self) -> None:
        prompt = "predictive "
        evaluation = evaluate_rollout(
            self.model,
            prompt,
            steps=12,
            mode="closed_loop",
            greedy=True,
        )
        direct = self.model.score(evaluation.sequence_tokens)

        self.assertEqual(evaluation.mode, "closed_loop")
        self.assertEqual(evaluation.sequence_tokens.shape[0], len(prompt.encode("utf-8")) + 12)
        self.assertEqual(evaluation.continuation_tokens.shape[0], 12)
        self.assertTrue(np.isfinite(evaluation.bits_per_byte))
        self.assertAlmostEqual(evaluation.bits_per_byte, direct.bits_per_byte, places=10)

    def test_closed_loop_accepts_continuation_length(self) -> None:
        prompt = "predictive "
        continuation = "coding"
        evaluation = evaluate_rollout(
            self.model,
            prompt,
            continuation=continuation,
            mode="closed_loop",
            greedy=True,
        )

        self.assertEqual(evaluation.continuation_tokens.shape[0], len(continuation.encode("utf-8")))
        self.assertEqual(evaluation.sequence_tokens.shape[0], len(prompt.encode("utf-8")) + len(continuation.encode("utf-8")))

    def test_teacher_forced_requires_continuation(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_rollout(self.model, "prompt", mode="teacher_forced")


if __name__ == "__main__":
    unittest.main()
