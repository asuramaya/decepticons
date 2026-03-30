from __future__ import annotations

import unittest

from open_predictive_coder import ByteCodec, OpenPredictiveCoder, OpenPredictiveCoderConfig
from open_predictive_coder.config import LatentConfig, ReservoirConfig, SegmenterConfig


def small_config() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        reservoir=ReservoirConfig(size=96, connectivity=0.12, spectral_radius=0.9, leak=0.35, seed=11),
        latent=LatentConfig(latent_dim=24, global_dim=24, reservoir_features=24, readout_l2=1e-5),
    )


class OpenPredictiveCoderTests(unittest.TestCase):
    def test_fit_and_score_repeating_text(self) -> None:
        model = OpenPredictiveCoder(config=small_config())
        corpus = "abababababababab" * 32
        fit_report = model.fit(corpus)
        score_report = model.score(corpus)
        contrast_report = model.score("abcdefghijklmnop" * 32)
        self.assertLess(fit_report.train_bits_per_byte, 8.0)
        self.assertLess(score_report.bits_per_byte, 8.0)
        self.assertLess(score_report.bits_per_byte, contrast_report.bits_per_byte)
        self.assertGreater(score_report.patches, 0)

    def test_greedy_generation_keeps_simple_pattern(self) -> None:
        model = OpenPredictiveCoder(config=small_config())
        corpus = "0101010101010101" * 32
        model.fit(corpus)
        sample = model.generate("0", steps=8, greedy=True)
        text = ByteCodec.decode_text(sample)
        self.assertEqual(text, "010101010")


if __name__ == "__main__":
    unittest.main()
