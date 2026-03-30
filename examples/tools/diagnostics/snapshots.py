from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .analysis import SignalSummary, format_signal_summary, summarize_signal


@dataclass(frozen=True)
class SnapshotRecord:
    step: int
    signals: tuple[SignalSummary, ...]

    def signal_names(self) -> tuple[str, ...]:
        return tuple(signal.name for signal in self.signals)

    def get(self, name: str) -> SignalSummary:
        for signal in self.signals:
            if signal.name == name:
                return signal
        raise KeyError(name)


@dataclass(frozen=True)
class SnapshotSeries:
    records: tuple[SnapshotRecord, ...]

    def signal_names(self) -> tuple[str, ...]:
        if not self.records:
            return ()
        return self.records[0].signal_names()

    def latest(self) -> SnapshotRecord:
        if not self.records:
            raise ValueError("snapshot series is empty")
        return self.records[-1]


def capture_snapshot(step: int, /, **signals: object) -> SnapshotRecord:
    summaries = tuple(
        summarize_signal(value, name=name)
        for name, value in signals.items()
    )
    return SnapshotRecord(step=int(step), signals=summaries)


def summarize_snapshot_series(series: SnapshotSeries, *, signal_name: str) -> dict[str, float | int]:
    if not series.records:
        raise ValueError("snapshot series is empty")

    selected = [record.get(signal_name) for record in series.records]
    first = selected[0]
    last = selected[-1]
    return {
        "signal": signal_name,
        "count": len(selected),
        "first_mean": first.mean,
        "last_mean": last.mean,
        "delta_mean": last.mean - first.mean,
        "first_std": first.std,
        "last_std": last.std,
        "delta_std": last.std - first.std,
    }


def format_snapshot_record(record: SnapshotRecord) -> str:
    pieces = ", ".join(format_signal_summary(signal) for signal in record.signals)
    return f"step={record.step}: {pieces}"


def format_snapshot_series(series: SnapshotSeries) -> str:
    if not series.records:
        return "SnapshotSeries(empty)"
    first = series.records[0].step
    last = series.records[-1].step
    names = ", ".join(series.signal_names())
    return f"SnapshotSeries(steps={first}->{last}, signals=[{names}], records={len(series.records)})"
