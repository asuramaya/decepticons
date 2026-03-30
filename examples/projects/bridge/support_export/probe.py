from __future__ import annotations

from model import SupportExportModel


def main() -> None:
    corpus = (
        "support export compares an exact teacher against a mixed backoff student.\n"
        "support export keeps the paired probability streams and support summaries generic.\n"
    ) * 4
    model = SupportExportModel.build()
    fit = model.fit(corpus)
    report = model.report(corpus[:192])

    print("project: support_export")
    print("backoff train tokens:", fit.backoff.ngram.tokens)
    print("exact train tokens:", fit.exact.tokens)
    print("mean bits/byte:", round(report.mean_bits_per_byte, 4))


if __name__ == "__main__":
    main()
