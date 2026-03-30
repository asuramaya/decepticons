from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from conker_shared import ResidualCorrectionModel, ResidualScore, build_delay_local_expert, build_linear_memory_expert


@dataclass
class Conker3Replica:
    model: ResidualCorrectionModel

    @classmethod
    def build(cls) -> "Conker3Replica":
        linear = build_linear_memory_expert(
            name="linear_path",
            embedding_dim=14,
            decays=(0.18, 0.42, 0.68, 0.86, 0.95),
            seed=61,
            alpha=1e-4,
        )
        local = build_delay_local_expert(
            name="local_path",
            history_length=4,
            embedding_dim=18,
            seed=67,
            alpha=1e-4,
        )
        return cls(model=ResidualCorrectionModel(base_expert=linear, local_expert=local, alpha=1e-4))

    def fit(self, data: object) -> dict[str, float]:
        return self.model.fit(data)

    def score(self, sequence: object) -> ResidualScore:
        return self.model.score(sequence)
