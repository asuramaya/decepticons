from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from open_predictive_coder import (
    ByteCodec,
    HierarchicalFeatureView,
    HierarchicalSubstrate,
    TrainModeConfig,
    hierarchical_small,
)
from model import CarvingMachineKernelAdapter, CarvingMachineKernelConfig


def main() -> None:
    config = hierarchical_small()
    substrate = HierarchicalSubstrate(config.hierarchical)
    view = HierarchicalFeatureView(config.hierarchical)
    model = CarvingMachineKernelAdapter(
        CarvingMachineKernelConfig(
            model=config,
            gate_source="routed",
            train_mode=TrainModeConfig(state_mode="through_state", slow_update_stride=2, rollout_checkpoint_stride=8),
        )
    )

    sequence = ByteCodec.encode_text("carving machine keeps fast, mid, and slow state aligned.")
    state = substrate.initial_state()
    previous_state: np.ndarray | None = None

    for token in sequence[: min(len(sequence), 24)]:
        previous_state = state
        state = substrate.step(state, int(token))

    feature = view.encode(state, previous_state=previous_state)
    print("hierarchical state dim:", substrate.state_dim)
    print("feature dim:", feature.shape[0])
    print("predictive dim:", view.predictive_dim)
    print("sampled readout dim:", model.sampled_readout.feature_dim)
    print("feature mean:", round(float(np.mean(feature)), 6))

    corpus = (
        "carving machine uses a rich substrate with a learned readout.\n"
        "hierarchical views expose fast, mid, and slow dynamics.\n"
    ) * 32
    trace = model.trace(corpus[:128])
    fit_report = model.fit(corpus)
    score_report = model.score(corpus)
    print("train bits/byte:", round(fit_report.train_bits_per_byte, 4))
    print("score bits/byte:", round(score_report.bits_per_byte, 4))
    print("mean fast_to_mid gate:", round(float(np.mean(trace.gates[:, 0])), 4))
    print("mean mid_to_slow gate:", round(float(np.mean(trace.gates[:, 1])), 4))
    print("mean hormone outputs:", np.round(np.mean(trace.hormones, axis=0), 4).tolist())
    print("route histogram:", np.bincount(trace.routes, minlength=2).tolist())
    print("checkpoint steps:", trace.checkpoint_steps)

    sample = model.generate(ByteCodec.encode_text("hierarchical "), steps=48, greedy=True)
    print(ByteCodec.decode_text(sample))


if __name__ == "__main__":
    main()
