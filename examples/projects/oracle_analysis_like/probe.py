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
    config = OracleAnalysisLikeConfig()
    model = OracleAnalysisLikeModel(config)
    sample = ByteCodec.encode_text("oracle analysis compares prefix and suffix state.")
    report = model.analyze(sample)

    print("state_dim:", config.model.hierarchical.state_dim)
    print("feature_dim:", model.feature_view.feature_dim)
    print("sampled_readout_dim:", model.sampled_readout.feature_dim)
    print("train_mode_state:", config.train_mode.state_mode)
    print("train_mode_stride:", config.train_mode.slow_update_stride)
    print("checkpoints:", report.checkpoints)
    print("oracle_preference_rate:", round(report.oracle_preference_rate, 4))
    print("mean_alignment_cosine:", round(report.mean_alignment_cosine, 4))
    print("mean_alignment_mae:", round(report.mean_alignment_mae, 4))
    first = report.points[0]
    print("first_route:", first.selected_route)
    print("first_route_weights:", tuple(round(float(value), 4) for value in first.route_weights))


if __name__ == "__main__":
    main()
