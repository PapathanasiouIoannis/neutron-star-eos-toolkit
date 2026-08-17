"""Deterministic, non-overwriting result bundles.

This module owns presentation and serialization.  Scientific orchestration
remains in :mod:`neutron_star_eos.model` and the solver modules.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from neutron_star_eos.eos import EosInputError
from neutron_star_eos.stellar import SequenceResult, StarResult

if TYPE_CHECKING:
    from neutron_star_eos.model import EosModel


def _write_new_bundle(
    destination: str | Path,
    writer: Callable[[Path], None],
) -> Path:
    resolved = Path(destination).expanduser().resolve()
    if resolved.exists():
        raise EosInputError(f"output directory already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{resolved.name}-", dir=resolved.parent))
    try:
        writer(temporary)
        os.replace(temporary, resolved)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_model_files(directory: Path, model: "EosModel") -> None:
    (directory / "summary.txt").write_text(
        model.summary() + "\n", encoding="utf-8", newline="\n"
    )
    _write_json(directory / "report.json", model.report().to_dict())


def _thermodynamic_columns(
    model: "EosModel",
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    view = model.thermodynamics(curve_points=257)
    for role in (
        "native_thermodynamics",
        "source_nodes",
        "continuous_barotrope",
    ):
        if role in view.roles:
            series = view.series_for(role)
            names = tuple(
                name
                for name in series.column_names
                if not (role == "source_nodes" and name == "source_node_position")
            )
            return names, tuple(series.column(name) for name in names)
    raise EosInputError("thermodynamic table is unavailable")


def _write_thermodynamics(path: Path, model: "EosModel") -> None:
    names, columns = _thermodynamic_columns(model)
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise RuntimeError("thermodynamic columns have inconsistent lengths")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names)
        for row in zip(*columns):
            writer.writerow(
                "" if not np.isfinite(value) else format(float(value), ".17g")
                for value in row
            )


def write_inspection(model: "EosModel", output_directory: str | Path) -> Path:
    """Write a deterministic inspection bundle to a new directory."""

    def writer(temporary: Path) -> None:
        _write_model_files(temporary, model)
        if model.report().capability("thermodynamics").available:
            _write_thermodynamics(temporary / "thermodynamics.csv", model)

    return _write_new_bundle(output_directory, writer)


def write_star(
    model: "EosModel", output_directory: str | Path, result: "StarResult"
) -> Path:
    """Write one already-computed stellar background and its model report."""

    if not isinstance(result, StarResult):
        raise EosInputError("write_star requires a StarResult")
    model._require_matching_result(result.model_name, result.eos_provenance_sha256)

    def writer(temporary: Path) -> None:
        _write_model_files(temporary, model)
        _write_json(temporary / "star.json", result.to_dict())

    return _write_new_bundle(output_directory, writer)


def write_sequence(
    model: "EosModel", output_directory: str | Path, result: "SequenceResult"
) -> Path:
    """Write all sequence attempts, including failures, to a new directory."""

    if not isinstance(result, SequenceResult):
        raise EosInputError("write_sequence requires a SequenceResult")
    model._require_matching_result(result.model_name, result.eos_provenance_sha256)

    def writer(temporary: Path) -> None:
        _write_model_files(temporary, model)
        _write_json(temporary / "sequence.json", result.to_dict())
        with (temporary / "sequence.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            fieldnames = (
                "central_pressure_mev_fm3",
                "status",
                "reason_code",
                "reason",
                "mass_msun",
                "radius_km",
                "boundary_pressure_mev_fm3",
                "boundary_status",
            )
            csv_writer = csv.DictWriter(
                stream, fieldnames=fieldnames, lineterminator="\n"
            )
            csv_writer.writeheader()
            for attempt in result.attempts:
                star = attempt.star
                csv_writer.writerow(
                    {
                        "central_pressure_mev_fm3": format(
                            attempt.central_pressure_mev_fm3, ".17g"
                        ),
                        "status": attempt.status,
                        "reason_code": attempt.reason_code or "",
                        "reason": attempt.reason or "",
                        "mass_msun": (
                            "" if star is None else format(star.mass_msun, ".17g")
                        ),
                        "radius_km": (
                            "" if star is None else format(star.radius_km, ".17g")
                        ),
                        "boundary_pressure_mev_fm3": (
                            ""
                            if star is None
                            else format(star.boundary_pressure_mev_fm3, ".17g")
                        ),
                        "boundary_status": (
                            "" if star is None else star.boundary_status
                        ),
                    }
                )

    return _write_new_bundle(output_directory, writer)


__all__ = ["write_inspection", "write_sequence", "write_star"]
