from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_predictive_coder import ByteCodec  # noqa: E402
from model import OracleAnalysisLikeConfig, OracleAnalysisLikeModel  # noqa: E402


def main() -> None:
    model = OracleAnalysisLikeModel(OracleAnalysisLikeConfig())
    corpus = (
        "oracle analysis keeps the causal prefix honest while inspecting the suffix.\n"
        "a reverse scan provides a noncausal comparison surface.\n"
    ) * 24
    report = model.analyze(ByteCodec.encode_text(corpus))

    print("tokens:", report.tokens)
    print("checkpoints:", report.checkpoints)
    print("oracle preference rate:", round(report.oracle_preference_rate, 4))
    print("mean alignment pearson:", round(report.mean_alignment_pearson, 4))
    print("mean alignment cosine:", round(report.mean_alignment_cosine, 4))
    print("mean alignment mae:", round(report.mean_alignment_mae, 4))
    for point in report.points[:3]:
        print(
            f"step={point.checkpoint} route={point.selected_route} "
            f"pearson={point.alignment_pearson:.4f} cosine={point.alignment_cosine:.4f} "
            f"slow_update_active={int(point.slow_update_active)}"
        )


if __name__ == "__main__":
    main()
