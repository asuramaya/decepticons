from __future__ import annotations

from model import CacheRepairModel


def main() -> None:
    corpus = (
        "cache repair prefers exact context when support is strong.\n"
        "cache repair still needs a broad prior when support collapses.\n"
    ) * 4
    model = CacheRepairModel.build()
    model.fit(corpus)
    trace = model.trace(corpus[:192])
    score = model.score(corpus[:192])

    print("repair strength:", round(float(trace.repair_strength.mean()), 4))
    print("exact support:", round(float(trace.exact_support.mean()), 4))
    print("mixed bits/byte:", round(score.mixed_bits_per_byte, 4))


if __name__ == "__main__":
    main()
