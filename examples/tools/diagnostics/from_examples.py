from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterator, Sequence

import numpy as np

from .ablation import AblationComparison, compare_ablation, compare_ablation_map
from .analysis import SignalSummary, format_signal_summary, summarize_signal
from .snapshots import SnapshotRecord, SnapshotSeries, capture_snapshot, format_snapshot_record, format_snapshot_series, summarize_snapshot_series


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
PROJECTS_ROOT = REPO_ROOT / "examples" / "projects"


@contextmanager
def _sys_path(*paths: Path) -> Iterator[None]:
    inserted: list[str] = []
    try:
        for path in reversed(paths):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
                inserted.append(value)
        yield
    finally:
        for value in inserted:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def _load_project_module(project: str, relative_path: str, module_name: str) -> ModuleType:
    project_dir = PROJECTS_ROOT / project
    module_path = project_dir / relative_path
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    with _sys_path(SRC_ROOT, PROJECTS_ROOT, project_dir):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


@dataclass(frozen=True)
class ExampleDiagnosticsReport:
    project: str
    flow: str
    snapshot: SnapshotRecord | None
    series: SnapshotSeries | None
    signal_summaries: tuple[SignalSummary, ...]
    ablations: tuple[AblationComparison, ...]
    notes: tuple[str, ...] = ()

    def format_lines(self) -> tuple[str, ...]:
        lines = [f"{self.project} [{self.flow}]"]
        if self.snapshot is not None:
            lines.append(format_snapshot_record(self.snapshot))
        if self.series is not None:
            lines.append(format_snapshot_series(self.series))
            for name in self.series.signal_names():
                lines.append(str(summarize_snapshot_series(self.series, signal_name=name)))
        for summary in self.signal_summaries:
            lines.append(format_signal_summary(summary))
        for ablation in self.ablations:
            lines.append(
                f"{ablation.name}: baseline={ablation.baseline:.4f} variant={ablation.variant:.4f} "
                f"delta={ablation.delta:+.4f} rel={ablation.relative_change:+.1f}%"
            )
        lines.extend(self.notes)
        return tuple(lines)


def _fit_trace_report(
    *,
    project: str,
    flow: str,
    trace: object,
    fit_bits_per_byte: float | None = None,
    score_bits_per_byte: float | None = None,
    ablations: tuple[AblationComparison, ...] = (),
    notes: tuple[str, ...] = (),
) -> ExampleDiagnosticsReport:
    trace_dict = {}
    for name in ("features", "gates", "targets", "prediction", "surprise"):
        if hasattr(trace, name):
            value = getattr(trace, name)
            if value is not None:
                trace_dict[name] = value
    snapshot = capture_snapshot(0, **trace_dict) if trace_dict else None
    series = None
    if hasattr(trace, "features") and hasattr(trace, "gates"):
        features = np.asarray(getattr(trace, "features"))
        gates = np.asarray(getattr(trace, "gates"))
        targets = np.asarray(getattr(trace, "targets")) if hasattr(trace, "targets") else None
        if features.ndim == 2 and gates.ndim == 2 and features.shape[0] == gates.shape[0]:
            records = []
            for step in range(features.shape[0]):
                payload = {
                    "features": features[step],
                    "gates": gates[step],
                }
                if targets is not None and targets.shape[0] == features.shape[0]:
                    payload["targets"] = np.asarray([targets[step]], dtype=np.float64)
                records.append(capture_snapshot(step, **payload))
            series = SnapshotSeries(tuple(records))

    signal_summaries: list[SignalSummary] = []
    if hasattr(trace, "features"):
        signal_summaries.append(summarize_signal(getattr(trace, "features"), name="features"))
    if hasattr(trace, "gates"):
        signal_summaries.append(summarize_signal(getattr(trace, "gates"), name="gates"))
    if hasattr(trace, "targets"):
        signal_summaries.append(summarize_signal(getattr(trace, "targets"), name="targets"))
    if fit_bits_per_byte is not None:
        signal_summaries.append(summarize_signal(np.asarray([fit_bits_per_byte]), name="fit_bits_per_byte"))
    if score_bits_per_byte is not None:
        signal_summaries.append(summarize_signal(np.asarray([score_bits_per_byte]), name="score_bits_per_byte"))

    return ExampleDiagnosticsReport(
        project=project,
        flow=flow,
        snapshot=snapshot,
        series=series,
        signal_summaries=tuple(signal_summaries),
        ablations=ablations,
        notes=notes,
    )


def diagnose_carving_machine_like(
    corpus: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] = (
        "carving machine uses a rich substrate with a learned readout.\n"
        "hierarchical views expose fast, mid, and slow dynamics.\n"
    )
    * 8,
) -> ExampleDiagnosticsReport:
    module = _load_project_module("carving_machine_like", "model.py", "diagnostics_examples.carving_machine_like.model")
    config = module.CarvingMachineKernelConfig()
    model = module.CarvingMachineKernelAdapter(config)
    fit_report = model.fit(corpus)
    trace = model.trace(corpus[:128] if isinstance(corpus, str) else corpus)
    score_report = model.score(corpus)
    ablation = compare_ablation(
        "fit bits/byte",
        fit_report.train_bits_per_byte,
        "score bits/byte",
        score_report.bits_per_byte,
        name="fit vs score",
    )
    return _fit_trace_report(
        project="carving_machine_like",
        flow="hierarchical_trace",
        trace=trace,
        fit_bits_per_byte=fit_report.train_bits_per_byte,
        score_bits_per_byte=score_report.bits_per_byte,
        ablations=(ablation,),
        notes=(
            f"gate_source={config.gate_source}",
            f"aux_source={config.aux_source}",
            f"sampled_readout_dim={model.sampled_readout.feature_dim}",
        ),
    )


def diagnose_conker_early_like(
    corpus: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] = (
        "exact history should help when the local suffix is stable.\n"
        "the support mixer should keep the base model in charge when support is thin.\n"
    )
    * 2,
) -> ExampleDiagnosticsReport:
    module = _load_project_module("conker_early_like", "run.py", "diagnostics_examples.conker_early_like.run")
    model = module.build_model()
    fit_report = model.fit(corpus)
    score = model.score(corpus[:128] if isinstance(corpus, str) else corpus)
    ablations = compare_ablation_map(
        "base",
        score.base_bits_per_byte,
        {
            "exact": score.exact_bits_per_byte,
            "mixed": score.mixed_bits_per_byte,
        },
    )
    snapshot = capture_snapshot(
        0,
        base_bits=np.asarray([score.base_bits_per_byte]),
        exact_bits=np.asarray([score.exact_bits_per_byte]),
        mixed_bits=np.asarray([score.mixed_bits_per_byte]),
        exact_support=np.asarray([score.exact_support], dtype=np.float64),
    )
    return ExampleDiagnosticsReport(
        project="conker_early_like",
        flow="exact_context_repair",
        snapshot=snapshot,
        series=None,
        signal_summaries=(
            summarize_signal(np.asarray([fit_report["base_train_bits_per_byte"]]), name="base_train_bits_per_byte"),
            summarize_signal(np.asarray([score.base_bits_per_byte]), name="base_bits_per_byte"),
            summarize_signal(np.asarray([score.exact_bits_per_byte]), name="exact_bits_per_byte"),
            summarize_signal(np.asarray([score.mixed_bits_per_byte]), name="mixed_bits_per_byte"),
        ),
        ablations=ablations,
        notes=(f"exact_order={score.exact_order}",),
    )


def _diagnose_conker_replica(
    project: str,
    *,
    corpus: str,
) -> ExampleDiagnosticsReport:
    module = _load_project_module(project, "model.py", f"diagnostics_examples.{project}.model")
    if project == "conker_1_like":
        model = module.Conker1Replica.build()
    elif project == "conker_2_like":
        model = module.Conker2Replica.build()
    elif project == "conker_3_like":
        model = module.Conker3Replica.build()
    else:
        raise ValueError(f"unsupported conker replica: {project}")
    model.fit(corpus)
    score = model.score(corpus[:128] if isinstance(corpus, str) else corpus)
    if project == "conker_3_like":
        ablations = (
            compare_ablation("base", score.base_bits_per_byte, "corrected", score.corrected_bits_per_byte, name="base vs corrected"),
            compare_ablation("local", score.local_bits_per_byte, "corrected", score.corrected_bits_per_byte, name="local vs corrected"),
        )
        snapshot = capture_snapshot(
            0,
            base_bits=np.asarray([score.base_bits_per_byte]),
            local_bits=np.asarray([score.local_bits_per_byte]),
            corrected_bits=np.asarray([score.corrected_bits_per_byte]),
        )
        signal_summaries = (
            summarize_signal(np.asarray([score.base_bits_per_byte]), name="base_bits_per_byte"),
            summarize_signal(np.asarray([score.local_bits_per_byte]), name="local_bits_per_byte"),
            summarize_signal(np.asarray([score.corrected_bits_per_byte]), name="corrected_bits_per_byte"),
        )
    else:
        component_items = tuple(score.component_bits_per_byte.items())
        ablations = tuple(
            compare_ablation(component_name, component_value, "mixed", score.mixed_bits_per_byte)
            for component_name, component_value in component_items
        )
        snapshot = capture_snapshot(
            0,
            mixed_bits=np.asarray([score.mixed_bits_per_byte]),
            **{name: np.asarray([value]) for name, value in component_items},
        )
        signal_summaries = (
            summarize_signal(np.asarray([score.mixed_bits_per_byte]), name="mixed_bits_per_byte"),
            *(
                summarize_signal(np.asarray([value]), name=f"{name}_bits_per_byte")
                for name, value in component_items
            ),
        )
    return ExampleDiagnosticsReport(
        project=project,
        flow="conker_replica",
        snapshot=snapshot,
        series=None,
        signal_summaries=tuple(signal_summaries),
        ablations=ablations,
        notes=(),
    )


def diagnose_conker_1_like(corpus: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] = (
    "memory should beat stability when the suffix is narrow.\n"
    "stability should win when the substrate is already clean.\n"
) * 2) -> ExampleDiagnosticsReport:
    return _diagnose_conker_replica("conker_1_like", corpus=corpus)


def diagnose_conker_2_like(corpus: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] = (
    "linear memory carries the main path while local correction stays smaller.\n"
    "the correction expert should only matter when the base path misses detail.\n"
) * 2) -> ExampleDiagnosticsReport:
    return _diagnose_conker_replica("conker_2_like", corpus=corpus)


def diagnose_conker_3_like(corpus: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] = (
    "local residual repair should stay narrow and selective.\n"
    "the base path should remain responsible for most of the distribution.\n"
) * 2) -> ExampleDiagnosticsReport:
    return _diagnose_conker_replica("conker_3_like", corpus=corpus)


def format_example_diagnostics(report: ExampleDiagnosticsReport) -> str:
    return "\n".join(report.format_lines())


__all__ = [
    "ExampleDiagnosticsReport",
    "diagnose_carving_machine_like",
    "diagnose_conker_1_like",
    "diagnose_conker_2_like",
    "diagnose_conker_3_like",
    "diagnose_conker_early_like",
    "format_example_diagnostics",
]
