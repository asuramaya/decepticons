from __future__ import annotations

import unittest

from open_predictive_coder import (
    ByteLatentPredictiveCoder,
    OpenPredictiveCoderConfig,
    evaluate_dataset,
    evaluate_rollout_curve,
    evaluate_transfer_probe,
)
from open_predictive_coder.config import LatentConfig, ReservoirConfig, SegmenterConfig


def small_config() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        reservoir=ReservoirConfig(size=64, connectivity=0.12, spectral_radius=0.9, leak=0.35, seed=17),
        latent=LatentConfig(latent_dim=16, global_dim=16, reservoir_features=16, readout_l2=1e-5),
    )


class TrainEvalTests(unittest.TestCase):
    def test_evaluate_dataset_matches_weighted_sequence_scores(self) -> None:
        model = ByteLatentPredictiveCoder(config=small_config())
        sequences = ("abababababab", "abcabcabcabc")
        model.fit(sequences)

        dataset_report = evaluate_dataset(model, sequences)
        direct_reports = [model.score(sequence) for sequence in sequences]
        total_effective = sum(report.tokens - 1 for report in direct_reports)
        weighted = sum(report.bits_per_byte * (report.tokens - 1) for report in direct_reports) / total_effective

        self.assertEqual(dataset_report.sequences, 2)
        self.assertEqual(dataset_report.effective_tokens, total_effective)
        self.assertAlmostEqual(dataset_report.bits_per_byte, weighted, places=10)

    def test_teacher_forced_rollout_curve_tracks_checkpoints(self) -> None:
        model = ByteLatentPredictiveCoder(config=small_config())
        corpus = "abababababababab" * 8
        model.fit(corpus)

        evaluation = evaluate_rollout_curve(
            model,
            "a",
            continuation="babab",
            mode="teacher_forced",
            checkpoints=(1, 3, 5),
        )

        self.assertEqual(evaluation.mode, "teacher_forced")
        self.assertEqual([point.step for point in evaluation.checkpoints], [1, 3, 5])
        self.assertEqual(evaluation.generated_tokens.shape[0], 5)
        self.assertTrue(all(point.match_rate is not None for point in evaluation.checkpoints))
        direct = model.score("ababab")
        self.assertAlmostEqual(evaluation.checkpoints[-1].bits_per_byte, direct.bits_per_byte, places=10)

    def test_closed_loop_rollout_curve_generates_and_scores(self) -> None:
        model = ByteLatentPredictiveCoder(config=small_config())
        corpus = "0101010101010101" * 8
        model.fit(corpus)

        evaluation = evaluate_rollout_curve(
            model,
            "0",
            continuation="1010",
            mode="closed_loop",
            checkpoints=(2, 4),
            greedy=True,
        )

        self.assertEqual(evaluation.mode, "closed_loop")
        self.assertEqual(evaluation.generated_tokens.shape[0], 4)
        self.assertEqual([point.step for point in evaluation.checkpoints], [2, 4])
        self.assertGreaterEqual(evaluation.checkpoints[-1].match_rate or 0.0, 0.5)

    def test_transfer_probe_distinguishes_zero_shot_from_target_scratch(self) -> None:
        def factory() -> ByteLatentPredictiveCoder:
            return ByteLatentPredictiveCoder(config=small_config())

        report = evaluate_transfer_probe(
            factory,
            "abababababababab" * 8,
            target_train="cdcdcdcdcdcdcdcd" * 8,
        )

        self.assertLess(report.source_eval.bits_per_byte, report.target_zero_shot.bits_per_byte)
        self.assertIsNotNone(report.target_scratch_eval)
        assert report.target_scratch_eval is not None
        self.assertLess(report.target_scratch_eval.bits_per_byte, report.target_zero_shot.bits_per_byte)


if __name__ == "__main__":
    unittest.main()
