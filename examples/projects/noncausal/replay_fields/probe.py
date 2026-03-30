from __future__ import annotations

from model import ReplayFieldsModel


def main() -> None:
    corpus = (
        "title:alpha|body:repeat repeat repeat|tag:x\n"
        "title:alpha|body:repeat repeat repeat|tag:x\n"
        "title:beta|body:variation here|tag:y\n"
    )
    model = ReplayFieldsModel.build()
    fit = model.fit(corpus)
    report = model.score(corpus)

    print("project: replay_fields")
    print("forward tokens:", fit.reconstruction.forward.tokens)
    print("field spans:", report.field_span_count)
    print("replay field ratio:", round(report.replay_field_ratio, 4))


if __name__ == "__main__":
    main()
