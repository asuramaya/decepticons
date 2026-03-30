from __future__ import annotations

from model import ReplayFieldsModel


def main() -> None:
    corpus = (
        "title:alpha|body:repeat repeat repeat|tag:x\n"
        "title:alpha|body:repeat repeat repeat|tag:x\n"
        "title:beta|body:variation here|tag:y\n"
    )
    model = ReplayFieldsModel.build()
    model.fit(corpus)
    trace = model.trace(corpus)
    report = model.score(corpus)

    print("replay spans:", len(trace.reconstruction.replay_spans))
    print("field spans:", len(trace.field_spans))
    print("blended bits/byte:", round(report.reconstruction.blended_bits_per_byte, 4))


if __name__ == "__main__":
    main()
