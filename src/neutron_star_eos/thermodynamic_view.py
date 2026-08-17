"""Build uniform, source-aware thermodynamic views for plotting and study."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from neutron_star_eos.compose import ComposeEos
from neutron_star_eos.compose.thermodynamics import (
    COMPOSE_NATIVE_INTERPOLATION_POLICY,
)
from neutron_star_eos.eos import EosInputError
from neutron_star_eos.tabulated import TabulatedEos
from neutron_star_eos.thermodynamics import ThermodynamicSeries, ThermodynamicView

if TYPE_CHECKING:
    from neutron_star_eos.model import EosModel


def build_thermodynamic_view(
    model: EosModel, *, curve_points: int = 513
) -> ThermodynamicView:
    """Expose source nodes and continuous curves without changing either.

    The different series remain explicit because a CompOSE native profile, a
    selected stellar barotrope, and an evaluated interpolant are scientifically
    different objects.
    """

    if isinstance(curve_points, bool) or not isinstance(curve_points, int):
        raise TypeError("curve_points must be an integer")
    if curve_points < 17:
        raise ValueError("curve_points must be at least 17")
    series: list[ThermodynamicSeries] = []
    profile = model.native_thermodynamics
    if profile is not None:
        series.append(
            ThermodynamicSeries(
                role="native_thermodynamics",
                label="CompOSE native thermodynamics",
                columns={name: profile.column(name) for name in profile.column_names},
                units=profile.units,
                descriptions=profile.descriptions,
                diagnostic_codes=tuple(item.code for item in profile.diagnostics),
                metadata={
                    "source_rows": profile.source_rows,
                    "interpolation_policy": COMPOSE_NATIVE_INTERPOLATION_POLICY,
                    "source_values_modified": False,
                    "provenance": json.loads(profile.provenance_json),
                },
            )
        )
    eos = model.barotrope
    if isinstance(eos, TabulatedEos):
        columns: dict[str, np.ndarray] = {
            "energy_density_mev_fm3": eos.energy_density_mev_fm3,
            "pressure_mev_fm3": eos.pressure_mev_fm3,
            "source_node_position": np.arange(
                len(eos.energy_density_mev_fm3), dtype=float
            ),
        }
        if eos.baryon_density_fm3 is not None:
            columns["baryon_density_fm3"] = eos.baryon_density_fm3
        series.append(
            ThermodynamicSeries(
                role="source_nodes",
                label="CSV source nodes",
                columns=columns,
                units={
                    "energy_density_mev_fm3": "MeV fm^-3",
                    "pressure_mev_fm3": "MeV fm^-3",
                    "source_node_position": "source-row index",
                    "baryon_density_fm3": "fm^-3",
                },
                descriptions={
                    "energy_density_mev_fm3": "Supplied total energy density",
                    "pressure_mev_fm3": "Supplied pressure",
                    "source_node_position": "Zero-based source-row position",
                    "baryon_density_fm3": "Supplied baryon number density",
                },
                metadata={
                    "source_values_modified": False,
                    "interpolation_policy": eos.provenance()["interpolation"],
                },
            )
        )
    elif isinstance(eos, ComposeEos):
        rows = len(eos.energy_density_mev_fm3)
        selection = eos.provenance()["selection"]
        retained_positions = np.asarray(
            selection["retained_source_positions"], dtype=float
        )
        if len(retained_positions) != rows:
            raise RuntimeError(
                "CompOSE retained source positions do not match barotrope rows"
            )
        series.append(
            ThermodynamicSeries(
                role="source_nodes",
                label="Selected CompOSE stellar-barotrope nodes",
                columns={
                    "baryon_density_fm3": eos.baryon_density_fm3,
                    "energy_density_mev_fm3": eos.energy_density_mev_fm3,
                    "pressure_mev_fm3": eos.pressure_mev_fm3,
                    "source_node_position": retained_positions,
                },
                units={
                    "baryon_density_fm3": "fm^-3",
                    "energy_density_mev_fm3": "MeV fm^-3",
                    "pressure_mev_fm3": "MeV fm^-3",
                    "source_node_position": "source-row index",
                },
                descriptions={
                    "baryon_density_fm3": "Selected source baryon number density",
                    "energy_density_mev_fm3": "Selected total energy density",
                    "pressure_mev_fm3": "Selected pressure",
                    "source_node_position": "Original selected source-row position",
                },
                diagnostic_codes=tuple(
                    item.code for item in eos.slice_report.diagnostics
                ),
                metadata={
                    "source_values_modified": False,
                    "compose": eos.compose_metadata,
                    "selection": selection,
                },
            )
        )
    if eos is not None:
        epsilon = np.geomspace(
            eos.energy_density_min_mev_fm3,
            eos.energy_density_max_mev_fm3,
            curve_points,
        )
        series.append(
            ThermodynamicSeries(
                role="continuous_barotrope",
                label="Evaluated continuous barotrope",
                columns={
                    "energy_density_mev_fm3": epsilon,
                    "pressure_mev_fm3": np.asarray(
                        eos.pressure_from_energy_density(epsilon), dtype=float
                    ),
                    "sound_speed_squared": np.asarray(
                        eos.sound_speed_squared_from_energy_density(epsilon),
                        dtype=float,
                    ),
                },
                units={
                    "energy_density_mev_fm3": "MeV fm^-3",
                    "pressure_mev_fm3": "MeV fm^-3",
                    "sound_speed_squared": "dimensionless",
                },
                descriptions={
                    "energy_density_mev_fm3": "Evaluation-grid total energy density",
                    "pressure_mev_fm3": "Evaluated continuous pressure",
                    "sound_speed_squared": "Evaluated dP/dE",
                },
                diagnostic_codes=tuple(issue.code for issue in eos.validate().issues),
                metadata={
                    "sampling": "geometric_energy_density_grid",
                    "points": curve_points,
                    "source_values_modified": False,
                    "extrapolation": "forbidden",
                },
            )
        )
    if not series:
        reason = model.report().capability("thermodynamics").reason
        raise EosInputError(reason or "thermodynamic data are unavailable")
    return ThermodynamicView(model.model_name, model.kind, tuple(series))


__all__ = ["build_thermodynamic_view"]
