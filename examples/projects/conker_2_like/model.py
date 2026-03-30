from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from conker_shared import ExpertMixtureModel, MixtureScore, build_echo_correction_expert, build_linear_memory_expert


@dataclass
class Conker2Replica:
    model: ExpertMixtureModel

    @classmethod
    def build(cls) -> "Conker2Replica":
        linear = build_linear_memory_expert(
            name="linear_path",
            embedding_dim=14,
            decays=(0.22, 0.45, 0.72, 0.88, 0.96),
            seed=51,
            alpha=1e-4,
        )
        correction = build_echo_correction_expert(
            name="correction_path",
            size=48,
            seed=53,
            alpha=1e-4,
        )
        return cls(model=ExpertMixtureModel((linear, correction), alpha=1e-4))

    def fit(self, data: object) -> dict[str, float]:
        return self.model.fit(data)

    def score(self, sequence: object) -> MixtureScore:
        return self.model.score(sequence)
