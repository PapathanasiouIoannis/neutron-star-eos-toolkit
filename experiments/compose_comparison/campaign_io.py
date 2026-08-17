"""Small deterministic JSON and CSV writers used by the campaign."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        candidate = float(value)
        return candidate if math.isfinite(candidate) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def write_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Atomically write dictionaries in a declared CSV column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: "" if row.get(name) is None else row.get(name)
                    for name in fieldnames
                }
            )
    temporary.replace(path)


def write_columns(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    """Write equally sized numeric columns using round-trip float precision."""

    names = tuple(columns)
    lengths = {len(np.asarray(columns[name])) for name in names}
    if len(lengths) != 1:
        raise RuntimeError(f"unaligned columns for {path}")
    rows = (
        {
            name: (
                ""
                if not math.isfinite(float(np.asarray(columns[name])[index]))
                else f"{float(np.asarray(columns[name])[index]):.17g}"
            )
            for name in names
        }
        for index in range(next(iter(lengths)))
    )
    write_rows(path, names, rows)


__all__ = ["json_ready", "write_columns", "write_json", "write_rows"]
