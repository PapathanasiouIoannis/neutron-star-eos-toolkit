"""Write thermodynamic and stellar-sequence CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from neutron_star_eos.eos import EosInputError
from neutron_star_eos.stellar import SequenceResult

if TYPE_CHECKING:
    from neutron_star_eos.model import EosModel


def thermodynamic_columns(
    model: EosModel,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Choose the most source-native thermodynamic table available."""

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


def write_thermodynamics(path: Path, model: EosModel) -> None:
    """Write native/source thermodynamics without rounding away precision."""

    names, columns = thermodynamic_columns(model)
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


def write_sequence_table(path: Path, result: SequenceResult) -> None:
    """Write every requested central pressure, including failed attempts."""

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
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for attempt in result.attempts:
            star = attempt.star
            writer.writerow(
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
                    "boundary_status": "" if star is None else star.boundary_status,
                }
            )


__all__ = ["write_sequence_table", "write_thermodynamics"]
