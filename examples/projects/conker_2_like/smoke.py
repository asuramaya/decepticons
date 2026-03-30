from __future__ import annotations

from model import Conker2Replica


SMOKE_CORPUS = (
    "linear memory should capture most of the easy continuation work.\n"
    "a smaller nonlinear correction path should mainly repair local misses and edge cases.\n"
    "the same scaffold repeats but punctuation and spacing produce small nonlinear corrections.\n"
) * 3


def main() -> None:
    model = Conker2Replica.build()
    fit = model.fit(SMOKE_CORPUS)
    score = model.score(SMOKE_CORPUS[:192])

    print("project:", "conker_2_like")
    print("linear train bits/byte:", round(fit["linear_path"], 4))
    print("correction train bits/byte:", round(fit["correction_path"], 4))
    print("linear score bits/byte:", round(score.component_bits_per_byte["linear_path"], 4))
    print("correction score bits/byte:", round(score.component_bits_per_byte["correction_path"], 4))
    print("mixed score bits/byte:", round(score.mixed_bits_per_byte, 4))


if __name__ == "__main__":
    main()
