from __future__ import annotations

from model import Conker3Replica


SMOKE_CORPUS = (
    "local windows should repair repeated short suffixes when the linear scaffold is almost right.\n"
    "the linear scaffold should still hold the main continuation score across the sequence.\n"
    "right after this phrase, the next phrase repeats; right after this phrase, the next phrase repeats.\n"
) * 3


def main() -> None:
    model = Conker3Replica.build()
    fit = model.fit(SMOKE_CORPUS)
    score = model.score(SMOKE_CORPUS[:192])

    print("project:", "conker_3_like")
    print("linear train bits/byte:", round(fit["linear_path"], 4))
    print("local train bits/byte:", round(fit["local_path"], 4))
    print("base score bits/byte:", round(score.base_bits_per_byte, 4))
    print("local score bits/byte:", round(score.local_bits_per_byte, 4))
    print("corrected score bits/byte:", round(score.corrected_bits_per_byte, 4))


if __name__ == "__main__":
    main()
