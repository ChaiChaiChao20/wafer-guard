from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WaferData:
    features: np.ndarray
    labels: np.ndarray
    timestamps: tuple[datetime, ...]
    sensor_names: tuple[str, ...]


def _number(value: str | None) -> float:
    if value is None or not value.strip():
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric sensor value, got {value!r}") from exc


def load_dataset(path: str | Path) -> WaferData:
    """Load the challenge CSV without fitting or inferring any preprocessing."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        sensor_names = tuple(
            sorted(
                (name for name in columns if name.startswith("sensor_")),
                key=lambda name: int(name.removeprefix("sensor_")),
            )
        )
        if not sensor_names or "label" not in columns or "timestamp" not in columns:
            raise ValueError("CSV must contain sensor_N columns, label, and timestamp")

        rows: list[list[float]] = []
        labels: list[int] = []
        timestamps: list[datetime] = []
        for row_number, row in enumerate(reader, start=2):
            rows.append([_number(row.get(name)) for name in sensor_names])
            label = int(row["label"])
            if label not in (-1, 1):
                raise ValueError(f"Unexpected label {label} on CSV row {row_number}")
            labels.append(int(label == 1))
            stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            timestamps.append(stamp.astimezone(timezone.utc))

    if not rows:
        raise ValueError("CSV contains no wafer rows")
    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    # Values are normalized to UTC above; integer epoch microseconds avoid
    # NumPy's timezone-naive datetime conversion warning without changing order.
    order = np.argsort(np.asarray([int(stamp.timestamp() * 1_000_000) for stamp in timestamps], dtype=np.int64), kind="stable")
    return WaferData(
        features=x[order],
        labels=y[order],
        timestamps=tuple(timestamps[int(i)] for i in order),
        sensor_names=sensor_names,
    )


def chronological_split(
    timestamps: tuple[datetime, ...], train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition ordered rows into past-only train, validation, and test blocks."""
    n = len(timestamps)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    if train_end < 2 or validation_end >= n:
        raise ValueError("Dataset is too small for the requested chronological split")
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n),
    )

