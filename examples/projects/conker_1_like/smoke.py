from __future__ import annotations

from model import Conker1Replica


SMOKE_CORPUS = (
    "the stable branch should help when document structure repeats in calm runs.\n"
    "the memory branch should carry ordinary continuation pressure across the local stream.\n"
    "stable structure repeats, stable structure repeats, then a local variation appears.\n"
) * 3


def main() -> None:
    model = Conker1Replica.build()
    fit = model.fit(SMOKE_CORPUS)
    score = model.score(SMOKE_CORPUS[:192])

    print("project:", "conker_1_like")
    print("memory train bits/byte:", round(fit["memory_path"], 4))
    print("stability train bits/byte:", round(fit["stability_path"], 4))
    print("memory score bits/byte:", round(score.component_bits_per_byte["memory_path"], 4))
    print("stability score bits/byte:", round(score.component_bits_per_byte["stability_path"], 4))
    print("mixed score bits/byte:", round(score.mixed_bits_per_byte, 4))


if __name__ == "__main__":
    main()
