from __future__ import annotations

import argparse
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from exact_context import EarlyConkerLikeModel


SMOKE_CORPUS = (
    "exact history should help when the local suffix is stable.\n"
    "the support mixer should keep the base model in charge when support is thin.\n"
) * 2


DEMO_CORPUS = (
    "predictive coding keeps the useful substrate and asks exact-context memory to repair local continuations.\n"
    "this early conker-like example mixes a byte model with exact1 exact2 exact3 count memories.\n"
    "support should matter more than lore and less than evidence.\n"
) * 8


def build_model() -> EarlyConkerLikeModel:
    return EarlyConkerLikeModel.build(reservoir_size=64, latent_dim=16, exact_order=2)


def run(mode: str) -> None:
    corpus = SMOKE_CORPUS if mode == "smoke" else DEMO_CORPUS
    score_corpus = corpus[:128] if mode == "smoke" else corpus[:384]
    model = build_model()
    fit_report = model.fit(corpus)
    score = model.score(score_corpus)

    print("mode:", mode)
    print("base train bits/byte:", round(fit_report["base_train_bits_per_byte"], 4))
    print("exact contexts:", int(fit_report["exact_contexts"]))
    print("exact supports:", int(fit_report["exact_supports"]))
    print("score tokens:", score.tokens)
    print("base bits/byte:", round(score.base_bits_per_byte, 4))
    print("exact bits/byte:", round(score.exact_bits_per_byte, 4))
    print("mixed bits/byte:", round(score.mixed_bits_per_byte, 4))
    print("exact support:", int(score.exact_support))
    print("best exact order:", score.exact_order)

    prompt = "exact "
    generated = model.generate(prompt, steps=80 if mode == "demo" else 16, greedy=True)
    print("prompt:", prompt)
    print("sample:", generated.tobytes().decode("utf-8", errors="replace"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the early conker-like example project.")
    parser.add_argument("--mode", choices=("smoke", "demo"), default="demo")
    args = parser.parse_args(argv)
    run(args.mode)


if __name__ == "__main__":
    main()
