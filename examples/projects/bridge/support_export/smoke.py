from __future__ import annotations

from model import SupportExportModel


def main() -> None:
    corpus = (
        "support export compares an exact teacher against a mixed backoff student.\n"
        "support export keeps the paired probability streams and support summaries generic.\n"
    ) * 4
    model = SupportExportModel.build()
    model.fit(corpus)
    report = model.report(corpus[:192])

    print("label flip rate:", round(report.label_flip_rate, 4))
    print("mean exact support:", round(report.mean_exact_support, 4))
    print("mean agreement mass:", round(report.mean_agreement_mass, 4))


if __name__ == "__main__":
    main()
