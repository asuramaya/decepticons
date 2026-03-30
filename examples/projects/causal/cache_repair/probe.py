from __future__ import annotations

from model import CacheRepairModel


def main() -> None:
    corpus = (
        "cache repair prefers exact context when support is strong.\n"
        "cache repair still needs a broad prior when support collapses.\n"
    ) * 4
    model = CacheRepairModel.build()
    fit = model.fit(corpus)
    score = model.score(corpus[:192])

    print("project: cache_repair")
    print("backoff train tokens:", fit.backoff.ngram.tokens)
    print("exact train tokens:", fit.exact.tokens)
    print("exact win rate:", round(fit.exact_win_rate, 4))
    print("mixed bits/byte:", round(score.mixed_bits_per_byte, 4))


if __name__ == "__main__":
    main()
