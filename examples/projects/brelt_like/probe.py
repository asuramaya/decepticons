from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import BreltLikeByteModel, BreltLikeConfig


def main() -> None:
    model = BreltLikeByteModel(BreltLikeConfig())
    config = model.config
    print("vocabulary_size:", config.vocabulary_size)
    print("token_view_dim:", config.token_view_dim)
    print("latent_dim:", config.latent.latent_dim)
    print("global_dim:", config.latent.global_dim)
    print("feature_dim:", model.feature_dim)
    print("patch_size:", config.segmenter.patch_size)
    print("min_patch_size:", config.segmenter.min_patch_size)
    print("max_patch_size:", config.segmenter.max_patch_size)
    print("novelty_threshold:", config.segmenter.novelty_threshold)


if __name__ == "__main__":
    main()
