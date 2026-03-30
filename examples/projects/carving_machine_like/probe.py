from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from open_predictive_coder import HierarchicalFeatureView, HierarchicalSubstrate, hierarchical_small
from open_predictive_coder import TrainModeConfig
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

    print("state_dim:", substrate.state_dim)
    print("fast/mid/slow:", config.hierarchical.fast_size, config.hierarchical.mid_size, config.hierarchical.slow_size)
    print("feature_dim:", view.feature_dim)
    print("predictive_dim:", view.predictive_dim)
    print("bank_slices:", view.bank_slices)
    print("sampled_readout_dim:", model.sampled_readout.feature_dim)
    print("sampled_readout_bands:", tuple(band.name for band in model.sampled_readout.config.bands))
    print("sampled_readout_sizes:", tuple(band.resolved_sample_count for band in model.sampled_readout.config.bands))
    print("train_mode:", model.config.train_mode.state_mode, model.config.train_mode.slow_update_stride, model.config.train_mode.rollout_checkpoint_stride)
    print("hormone_count:", model.hormone_modulator.output_count)
    print("routing_mode:", model.summary_router.config.mode)
    print("gate_source:", model.config.gate_source)
    print("aux_source:", model.config.aux_source)
    print("gate_flags:", model.config.gate_fast_mid, model.config.gate_mid_slow)
    print("predictor_view_dim:", model.config.controller_view_dim)
    print("predictor_width:", model.config.controller_width)
    print("sample_sizes:", model.config.fast_sample_size, model.config.mid_sample_size, model.config.slow_sample_size)
    print("gate_indices:", model.config.pathway_gates.fast_to_mid_index, model.config.pathway_gates.mid_to_slow_index)


if __name__ == "__main__":
    main()
