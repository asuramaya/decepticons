from __future__ import annotations

from open_predictive_coder import (
    ByteCodec,
    ByteLatentPredictiveCoder,
    OpenPredictiveCoderConfig,
    ReservoirConfig,
    SegmenterConfig,
)
from open_predictive_coder.config import LatentConfig


def quickstart_config() -> OpenPredictiveCoderConfig:
    return OpenPredictiveCoderConfig(
        segmenter=SegmenterConfig(mode="adaptive", patch_size=4, min_patch_size=2, max_patch_size=8, novelty_threshold=0.08),
        reservoir=ReservoirConfig(size=96, connectivity=0.12, spectral_radius=0.9, leak=0.35, seed=11),
        latent=LatentConfig(latent_dim=24, global_dim=24, reservoir_features=24, readout_l2=1e-5),
    )


def main() -> None:
    corpus = (
        "predictive coding compresses what is easy and spends effort on what is surprising.\n"
        "open predictive coder is a small reference library for that idea.\n"
    ) * 64

    model = ByteLatentPredictiveCoder(config=quickstart_config())
    fit_report = model.fit(corpus)
    print("train bits/byte:", round(fit_report.train_bits_per_byte, 4))

    prompt = ByteCodec.encode_text("predictive ")
    sample = model.generate(prompt, steps=80, greedy=True)
    print(ByteCodec.decode_text(sample))


if __name__ == "__main__":
    main()
