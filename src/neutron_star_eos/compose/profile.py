"""Immutable table returned by native CompOSE thermodynamic interpolation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from neutron_star_eos.compose.diagnostics import ComposeProfileDiagnostic

COMPOSE_NATIVE_THERMODYNAMIC_SCHEMA_VERSION = "compose_native_thermodynamics_v1"
COMPOSE_NATIVE_INTERPOLATION_POLICY = "piecewise_linear_native_Q_in_nB_v1"

Q_FIELDS = (
    ("q1_pressure_per_baryon_mev", "MeV", "Q1 = P/nB"),
    ("q2_entropy_per_baryon", "dimensionless", "Q2 = entropy per baryon"),
    ("q3_mu_b_minus_mn_over_mn", "dimensionless", "Q3 = (muB-mn)/mn"),
    ("q4_mu_q_over_mn", "dimensionless", "Q4 = muQ/mn"),
    ("q5_mu_l_over_mn", "dimensionless", "Q5 = muL/mn"),
    ("q6_free_energy_per_baryon_over_mn_minus_1", "dimensionless", "Q6 = F/mn-1"),
    ("q7_internal_energy_per_baryon_over_mn_minus_1", "dimensionless", "Q7 = E/mn-1"),
)


def readonly_array(values: Any, *, dtype: Any = float) -> np.ndarray:
    """Copy an array and make it read-only for a public profile."""

    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def finite_range(values: np.ndarray) -> dict[str, float | None]:
    """Return finite extrema while keeping an all-missing column explicit."""

    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"minimum": None, "maximum": None}
    return {"minimum": float(np.min(finite)), "maximum": float(np.max(finite))}


@dataclass(frozen=True, slots=True)
class ComposeThermodynamicProfile:
    """Read-only native-Q interpolation and reconstructed thermodynamic table."""

    model_id: str
    source_url: str
    columns: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    descriptions: Mapping[str, str]
    diagnostics: tuple[ComposeProfileDiagnostic, ...]
    source_rows: int
    provenance_json: str
    interpolation_policy: str = COMPOSE_NATIVE_INTERPOLATION_POLICY

    @property
    def status(self) -> str:
        return "available_with_diagnostics" if self.diagnostics else "available"

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def column(self, name: str) -> np.ndarray:
        try:
            values = self.columns[name]
        except KeyError as exc:
            raise KeyError(f"unknown CompOSE profile column: {name}") from exc
        result = values.view()
        result.setflags(write=False)
        return result

    def summary(self) -> dict[str, Any]:
        provenance = json.loads(self.provenance_json)
        highlighted = (
            "pressure_mev_fm3",
            "energy_density_mev_fm3",
            "baryon_chemical_potential_mev",
            "sound_speed_squared_curve_derivative",
            "sound_speed_squared_compose_thermodynamic",
            "sound_speed_squared_cold_beta_mu_derivative",
            "euler_normalized_residual",
            "first_law_normalized_residual",
            "gibbs_duhem_normalized_residual",
        )
        return {
            "schema_version": COMPOSE_NATIVE_THERMODYNAMIC_SCHEMA_VERSION,
            "status": self.status,
            "model_id": self.model_id,
            "source_url": self.source_url,
            "source_rows": self.source_rows,
            "profile_points": len(self.column("baryon_density_fm3")),
            "columns": list(self.column_names),
            "interpolation": {
                "policy": self.interpolation_policy,
                "coordinate": "baryon_density_fm3",
                "native_fields": [name for name, _unit, _description in Q_FIELDS],
                "derivative_at_source_node": "right_interval_except_upper_endpoint",
                "extrapolation": "forbidden",
                "query_grid": provenance["query_grid"],
            },
            "official_cold_1d_quantities_not_applicable": {
                "10": "dp/dnB at fixed energy requires a temperature dimension",
                "11": "dp/dE at fixed nB requires a temperature dimension",
                "13": "cV is not available from a one-point T=0 axis",
                "14": "cP is not available from a one-point T=0 axis",
                "16": "thermal expansion is not available from a one-point T=0 axis",
                "17": "thermal pressure coefficient is not available from a one-point T=0 axis",
            },
            "ranges": {
                name: {**finite_range(self.columns[name]), "unit": self.units[name]}
                for name in highlighted
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "provenance": provenance,
        }


_Q_FIELDS = Q_FIELDS
_readonly = readonly_array
_finite_range = finite_range
