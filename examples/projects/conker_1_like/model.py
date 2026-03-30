from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PROJECTS_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from conker_shared import ExpertMixtureModel, MixtureScore, build_hierarchical_stability_expert, build_linear_memory_expert


@dataclass
class Conker1Replica:
    model: ExpertMixtureModel

    @classmethod
    def build(cls) -> "Conker1Replica":
        compressor = build_linear_memory_expert(
            name="memory_path",
            embedding_dim=12,
            decays=(0.32, 0.58, 0.8, 0.92),
            seed=41,
            alpha=1e-4,
        )
        stability = build_hierarchical_stability_expert(
            name="stability_path",
            seed=43,
            alpha=1e-4,
        )
        return cls(model=ExpertMixtureModel((compressor, stability), alpha=1e-4))

    def fit(self, data: object) -> dict[str, float]:
        return self.model.fit(data)

    def score(self, sequence: object) -> MixtureScore:
        return self.model.score(sequence)
