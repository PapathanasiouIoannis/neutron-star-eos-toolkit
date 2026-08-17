"""Native-field thermodynamic interpolation for cold CompOSE density paths.

This layer is deliberately independent of a continuous ``P(epsilon)``
reduction.  It remains available when a source contains a local pressure
reversal or a closure residual, so those features can be inspected rather
than hidden or silently repaired.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from neutron_star_eos.compose.dataset import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
    ComposeColdSlice,
)
from neutron_star_eos.eos import EosInputError

COMPOSE_NATIVE_THERMODYNAMIC_SCHEMA_VERSION = "compose_native_thermodynamics_v1"
COMPOSE_NATIVE_INTERPOLATION_POLICY = "piecewise_linear_native_Q_in_nB_v1"

_Q_FIELDS = (
    ("q1_pressure_per_baryon_mev", "MeV", "Q1 = P/nB"),
    ("q2_entropy_per_baryon", "dimensionless", "Q2 = entropy per baryon"),
    ("q3_mu_b_minus_mn_over_mn", "dimensionless", "Q3 = (muB-mn)/mn"),
    ("q4_mu_q_over_mn", "dimensionless", "Q4 = muQ/mn"),
    ("q5_mu_l_over_mn", "dimensionless", "Q5 = muL/mn"),
    ("q6_free_energy_per_baryon_over_mn_minus_1", "dimensionless", "Q6 = F/mn-1"),
    ("q7_internal_energy_per_baryon_over_mn_minus_1", "dimensionless", "Q7 = E/mn-1"),
)


def _readonly(values: Any, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _finite_range(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"minimum": None, "maximum": None}
    return {"minimum": float(np.min(finite)), "maximum": float(np.max(finite))}


def _normalized_residual(residual: np.ndarray, *terms: np.ndarray) -> np.ndarray:
    scale = np.maximum.reduce(tuple(np.abs(term) for term in terms))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale > 0.0, np.abs(residual) / scale, np.nan)


@dataclass(frozen=True, slots=True)
class ComposeProfileDiagnostic:
    """One visible sampled-profile finding; never an automatic repair."""

    code: str
    severity: str
    sampled_points: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "sampled_points": self.sampled_points,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ComposeThermodynamicProfile:
    """Immutable native-Q interpolation and reconstructed thermodynamic table."""

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
                "native_fields": [name for name, _unit, _description in _Q_FIELDS],
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
                name: {**_finite_range(self.columns[name]), "unit": self.units[name]}
                for name in highlighted
            },
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "provenance": provenance,
        }


def _query_grid(
    nodes: np.ndarray,
    values: Any | None,
    *,
    points: int,
    include_source_nodes: bool,
) -> np.ndarray:
    if values is None:
        if int(points) < 2:
            raise ValueError("CompOSE profile points must be at least two")
        query = np.geomspace(float(nodes[0]), float(nodes[-1]), int(points))
        if include_source_nodes:
            query = np.unique(np.concatenate((query, nodes)))
    else:
        query = np.asarray(values, dtype=float)
        if query.ndim != 1 or not len(query):
            raise ValueError(
                "baryon_density_fm3 must be a non-empty one-dimensional array"
            )
        if np.any(~np.isfinite(query)) or np.any(np.diff(query) <= 0.0):
            raise ValueError(
                "baryon_density_fm3 must be finite and strictly increasing"
            )
    if query[0] < nodes[0] or query[-1] > nodes[-1]:
        raise EosInputError("native CompOSE interpolation forbids extrapolation")
    return np.asarray(query, dtype=float)


def _interval_indices(nodes: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(nodes, query, side="right") - 1, 0, len(nodes) - 2)


def _linear_fields(
    nodes: np.ndarray,
    source_values: np.ndarray,
    query: np.ndarray,
    intervals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    widths = nodes[1:] - nodes[:-1]
    slopes = (source_values[1:] - source_values[:-1]) / widths[:, None]
    selected_slopes = slopes[intervals]
    values = (
        source_values[intervals] + (query - nodes[intervals])[:, None] * selected_slopes
    )
    return values, selected_slopes


def _parse_composition_payload(
    tokens: tuple[str, ...],
) -> tuple[dict[int, float], dict[int, tuple[float, float, float]]]:
    try:
        pair_count = int(tokens[0])
        if pair_count < 0:
            raise ValueError
        cursor = 1
        pairs: dict[int, float] = {}
        for _ in range(pair_count):
            pairs[int(tokens[cursor])] = float(tokens[cursor + 1])
            cursor += 2
        quadruple_count = int(tokens[cursor])
        cursor += 1
        if quadruple_count < 0:
            raise ValueError
        quadruples: dict[int, tuple[float, float, float]] = {}
        for _ in range(quadruple_count):
            code = int(tokens[cursor])
            quadruples[code] = (
                float(tokens[cursor + 1]),
                float(tokens[cursor + 2]),
                float(tokens[cursor + 3]),
            )
            cursor += 4
        if cursor != len(tokens):
            raise ValueError
    except (IndexError, ValueError) as exc:
        raise EosInputError("CompOSE composition payload is malformed") from exc
    return pairs, quadruples


def _parse_microphysics(
    cold_slice: ComposeColdSlice,
) -> dict[tuple[int, int, int], dict[int, float]]:
    if "eos.micro" not in cold_slice.dataset.available_files:
        return {}
    try:
        lines = (
            cold_slice.dataset.source_file_bytes("eos.micro")
            .decode("ascii")
            .splitlines()
        )
    except UnicodeDecodeError as exc:
        raise EosInputError("CompOSE eos.micro is not ASCII") from exc
    parsed: dict[tuple[int, int, int], dict[int, float]] = {}
    for line_number, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            key = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            count = int(tokens[3])
            if count < 0 or len(tokens) != 4 + 2 * count:
                raise ValueError
            values = {
                int(tokens[4 + 2 * index]): float(tokens[5 + 2 * index])
                for index in range(count)
            }
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError
        except (IndexError, ValueError) as exc:
            raise EosInputError(
                f"CompOSE eos.micro line {line_number} is malformed"
            ) from exc
        parsed[key] = values
    return parsed


def _optional_source_fields(
    cold_slice: ComposeColdSlice,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    rows = cold_slice.rows
    count = len(rows)
    values: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    descriptions: dict[str, str] = {}

    maximum_additional_width = max(
        (len(row.additional_values) for row in rows), default=0
    )
    for index in range(maximum_additional_width):
        name = f"additional_{index + 1}"
        values[name] = np.asarray(
            [
                row.additional_values[index]
                if index < len(row.additional_values)
                else np.nan
                for row in rows
            ]
        )
        units[name] = "source-defined"
        descriptions[name] = (
            f"CompOSE additional thermodynamic quantity {index + 1}; "
            "missing source-row values remain NaN"
        )

    composition_map = {row.key: row for row in cold_slice.dataset.composition_rows}
    parsed_composition = []
    for row in rows:
        source = composition_map.get(row.key)
        parsed_composition.append(
            ({}, {})
            if source is None
            else _parse_composition_payload(source.raw_payload_tokens)
        )
    pair_codes = sorted(
        {code for pairs, _quads in parsed_composition for code in pairs}
    )
    for code in pair_codes:
        name = f"composition_pair_{code}"
        values[name] = np.asarray(
            [pairs.get(code, np.nan) for pairs, _quads in parsed_composition]
        )
        units[name] = "dimensionless"
        descriptions[name] = (
            f"CompOSE composition pair quantity for particle code {code}"
        )
    quadruple_codes = sorted(
        {code for _pairs, quadruples in parsed_composition for code in quadruples}
    )
    for code in quadruple_codes:
        # eos.compo stores (Aav, Zav, Yav); the official output order is
        # (Yav, Aav, Zav, Nav=Aav-Zav).
        for component, label in enumerate(("Aav", "Zav", "Yav")):
            name = f"composition_quadruple_{code}_{label}"
            values[name] = np.asarray(
                [
                    quadruples.get(code, (np.nan, np.nan, np.nan))[component]
                    for _pairs, quadruples in parsed_composition
                ]
            )
            units[name] = "dimensionless"
            descriptions[name] = (
                f"CompOSE composition quadruple {label} for particle-set code {code}"
            )
        name = f"composition_quadruple_{code}_Nav"
        values[name] = np.asarray(
            [
                (
                    quadruples[code][0] - quadruples[code][1]
                    if code in quadruples
                    else np.nan
                )
                for _pairs, quadruples in parsed_composition
            ]
        )
        units[name] = "dimensionless"
        descriptions[name] = (
            f"Calculated neutron number Aav-Zav for particle-set code {code}"
        )

    micro_map = _parse_microphysics(cold_slice)
    micro_codes = sorted({code for row in rows for code in micro_map.get(row.key, {})})
    for code in micro_codes:
        name = f"micro_{code}"
        values[name] = np.asarray(
            [micro_map.get(row.key, {}).get(code, np.nan) for row in rows]
        )
        units[name] = "source-defined"
        descriptions[name] = f"CompOSE microscopic quantity code {code}"

    if cold_slice.phase_codes is not None:
        values["phase_code"] = np.asarray(
            [np.nan if item is None else float(item) for item in cold_slice.phase_codes]
        )
        units["phase_code"] = "model-specific code"
        descriptions["phase_code"] = (
            "Model-specific CompOSE phase code; not interpreted physically"
        )
    assert all(len(item) == count for item in values.values())
    return values, units, descriptions


def _sample_optional_field(
    source: np.ndarray,
    nodes: np.ndarray,
    query: np.ndarray,
    intervals: np.ndarray,
    *,
    piecewise_constant: bool,
) -> np.ndarray:
    if piecewise_constant:
        result = source[intervals].copy()
        result[query == nodes[-1]] = source[-1]
    else:
        left = source[intervals]
        right = source[intervals + 1]
        usable = np.isfinite(left) & np.isfinite(right)
        fraction = (query - nodes[intervals]) / (
            nodes[intervals + 1] - nodes[intervals]
        )
        result = np.where(usable, left + fraction * (right - left), np.nan)
    positions = np.searchsorted(query, nodes)
    valid = positions < len(query)
    exact = np.zeros(len(nodes), dtype=bool)
    exact[valid] = query[positions[valid]] == nodes[valid]
    result[positions[exact]] = source[exact]
    return result


def _diagnostic(
    code: str,
    severity: str,
    mask: np.ndarray,
    message: str,
) -> ComposeProfileDiagnostic | None:
    count = int(np.count_nonzero(mask))
    if not count:
        return None
    return ComposeProfileDiagnostic(code, severity, count, message)


def interpolate_compose_thermodynamics(
    cold_slice: ComposeColdSlice,
    baryon_density_fm3: Any | None = None,
    *,
    points: int = 2001,
    include_source_nodes: bool = True,
) -> ComposeThermodynamicProfile:
    """Interpolate native Q fields first, then reconstruct cold thermodynamics.

    The interpolation is the first-order CompOSE-style policy: each native Q
    field is piecewise linear in baryon density.  Derived identities and both
    sound-speed definitions are reported side by side.  No source row is
    removed, reordered, clipped, or repaired.
    """

    if not isinstance(cold_slice, ComposeColdSlice):
        raise TypeError("interpolate_compose_thermodynamics expects ComposeColdSlice")
    nodes = np.asarray(cold_slice.baryon_density_fm3, dtype=float)
    source_q = np.asarray(cold_slice.q_values, dtype=float)
    if len(nodes) < 2 or source_q.shape != (len(nodes), 7):
        raise EosInputError(
            "native CompOSE thermodynamics requires at least two seven-Q rows"
        )
    query = _query_grid(
        nodes,
        baryon_density_fm3,
        points=int(points),
        include_source_nodes=bool(include_source_nodes),
    )
    intervals = _interval_indices(nodes, query)
    q, dq_dn = _linear_fields(nodes, source_q, query, intervals)
    neutron_mass = float(cold_slice.dataset.neutron_mass_mev)

    pressure = query * q[:, 0]
    entropy = q[:, 1]
    mu_b = neutron_mass * (1.0 + q[:, 2])
    mu_q = neutron_mass * q[:, 3]
    mu_l = neutron_mass * q[:, 4]
    free_per_baryon = neutron_mass * (1.0 + q[:, 5])
    energy_per_baryon = neutron_mass * (1.0 + q[:, 6])
    enthalpy_per_baryon = energy_per_baryon + q[:, 0]
    gibbs_per_baryon = free_per_baryon + q[:, 0]
    free_density = query * free_per_baryon
    energy_density = query * energy_per_baryon
    enthalpy_density = energy_density + pressure

    dpressure_dn = q[:, 0] + query * dq_dn[:, 0]
    denergy_dn = neutron_mass * (1.0 + q[:, 6] + query * dq_dn[:, 6])
    dmu_b_dn = neutron_mass * dq_dn[:, 2]
    dfree_per_baryon_dn = neutron_mass * dq_dn[:, 5]
    pressure_from_free_derivative = query**2 * dfree_per_baryon_dn
    mu_b_from_free_derivative = free_per_baryon + query * dfree_per_baryon_dn
    with np.errstate(divide="ignore", invalid="ignore"):
        cs2_curve = dpressure_dn / denergy_dn
        cs2_compose = dpressure_dn / enthalpy_per_baryon
        cs2_mu_derivative = query * dmu_b_dn / mu_b
        gamma = query * dpressure_dn / pressure
        bulk_modulus = query * dpressure_dn
        compressibility = 1.0 / bulk_modulus

    euler_residual = pressure - (query * mu_b - energy_density)
    first_law_residual = denergy_dn - mu_b
    gibbs_duhem_residual = dpressure_dn - query * dmu_b_dn
    free_pressure_residual = pressure - pressure_from_free_derivative
    free_mu_residual = mu_b - mu_b_from_free_derivative
    euler_normalized = _normalized_residual(
        euler_residual, pressure, query * mu_b, energy_density
    )
    first_law_normalized = _normalized_residual(first_law_residual, denergy_dn, mu_b)
    gibbs_duhem_normalized = _normalized_residual(
        gibbs_duhem_residual, dpressure_dn, query * dmu_b_dn
    )
    free_pressure_normalized = _normalized_residual(
        free_pressure_residual, pressure, pressure_from_free_derivative
    )
    free_mu_normalized = _normalized_residual(
        free_mu_residual, mu_b, mu_b_from_free_derivative
    )

    columns: dict[str, np.ndarray] = {
        "baryon_density_fm3": query,
        "source_interval_left_position": intervals.astype(float),
        "pressure_mev_fm3": pressure,
        "energy_density_mev_fm3": energy_density,
        "free_energy_density_mev_fm3": free_density,
        "enthalpy_density_mev_fm3": enthalpy_density,
        "entropy_per_baryon": entropy,
        "baryon_chemical_potential_mev": mu_b,
        "baryon_chemical_potential_minus_neutron_mass_mev": mu_b - neutron_mass,
        "charge_chemical_potential_mev": mu_q,
        "lepton_chemical_potential_mev": mu_l,
        "free_energy_per_baryon_mev": free_per_baryon,
        "internal_energy_per_baryon_mev": energy_per_baryon,
        "enthalpy_per_baryon_mev": enthalpy_per_baryon,
        "gibbs_free_energy_per_baryon_mev": gibbs_per_baryon,
        "enthalpy_per_baryon_over_mn_minus_1": enthalpy_per_baryon / neutron_mass - 1.0,
        "gibbs_free_energy_per_baryon_over_mn_minus_1": gibbs_per_baryon / neutron_mass
        - 1.0,
        "dpressure_dnB_mev": dpressure_dn,
        "denergy_density_dnB_mev": denergy_dn,
        "dmuB_dnB_mev_fm3": dmu_b_dn,
        "sound_speed_squared_curve_derivative": cs2_curve,
        "sound_speed_squared_compose_thermodynamic": cs2_compose,
        "sound_speed_squared_cold_beta_mu_derivative": cs2_mu_derivative,
        "barotropic_adiabatic_index": gamma,
        "compose_heat_capacity_ratio_at_zero_temperature": np.ones_like(query),
        "bulk_modulus_mev_fm3": bulk_modulus,
        "compressibility_mev_inverse_fm3": compressibility,
        "isothermal_compressibility_mev_inverse_fm3": compressibility,
        "isentropic_compressibility_mev_inverse_fm3": compressibility,
        "pressure_from_free_energy_derivative_mev_fm3": pressure_from_free_derivative,
        "muB_from_free_energy_derivative_mev": mu_b_from_free_derivative,
        "euler_residual_mev_fm3": euler_residual,
        "euler_normalized_residual": euler_normalized,
        "first_law_residual_mev": first_law_residual,
        "first_law_normalized_residual": first_law_normalized,
        "gibbs_duhem_residual_mev": gibbs_duhem_residual,
        "gibbs_duhem_normalized_residual": gibbs_duhem_normalized,
        "free_energy_pressure_residual_mev_fm3": free_pressure_residual,
        "free_energy_pressure_normalized_residual": free_pressure_normalized,
        "free_energy_muB_residual_mev": free_mu_residual,
        "free_energy_muB_normalized_residual": free_mu_normalized,
        "q5_beta_equilibrium_residual": q[:, 4],
        "q6_minus_q7_zero_temperature_residual": q[:, 5] - q[:, 6],
        "sound_speed_squared_definition_difference": cs2_curve - cs2_compose,
        "sound_speed_squared_mu_minus_compose": cs2_mu_derivative - cs2_compose,
    }
    units: dict[str, str] = {
        "baryon_density_fm3": "fm^-3",
        "source_interval_left_position": "source-row index",
        "pressure_mev_fm3": "MeV fm^-3",
        "energy_density_mev_fm3": "MeV fm^-3",
        "free_energy_density_mev_fm3": "MeV fm^-3",
        "enthalpy_density_mev_fm3": "MeV fm^-3",
        "entropy_per_baryon": "dimensionless",
        "baryon_chemical_potential_mev": "MeV",
        "baryon_chemical_potential_minus_neutron_mass_mev": "MeV",
        "charge_chemical_potential_mev": "MeV",
        "lepton_chemical_potential_mev": "MeV",
        "free_energy_per_baryon_mev": "MeV",
        "internal_energy_per_baryon_mev": "MeV",
        "enthalpy_per_baryon_mev": "MeV",
        "gibbs_free_energy_per_baryon_mev": "MeV",
        "enthalpy_per_baryon_over_mn_minus_1": "dimensionless",
        "gibbs_free_energy_per_baryon_over_mn_minus_1": "dimensionless",
        "dpressure_dnB_mev": "MeV",
        "denergy_density_dnB_mev": "MeV",
        "dmuB_dnB_mev_fm3": "MeV fm^3",
        "sound_speed_squared_curve_derivative": "dimensionless",
        "sound_speed_squared_compose_thermodynamic": "dimensionless",
        "sound_speed_squared_cold_beta_mu_derivative": "dimensionless",
        "barotropic_adiabatic_index": "dimensionless",
        "compose_heat_capacity_ratio_at_zero_temperature": "dimensionless",
        "bulk_modulus_mev_fm3": "MeV fm^-3",
        "compressibility_mev_inverse_fm3": "MeV^-1 fm^3",
        "isothermal_compressibility_mev_inverse_fm3": "MeV^-1 fm^3",
        "isentropic_compressibility_mev_inverse_fm3": "MeV^-1 fm^3",
        "pressure_from_free_energy_derivative_mev_fm3": "MeV fm^-3",
        "muB_from_free_energy_derivative_mev": "MeV",
        "euler_residual_mev_fm3": "MeV fm^-3",
        "euler_normalized_residual": "dimensionless",
        "first_law_residual_mev": "MeV",
        "first_law_normalized_residual": "dimensionless",
        "gibbs_duhem_residual_mev": "MeV",
        "gibbs_duhem_normalized_residual": "dimensionless",
        "free_energy_pressure_residual_mev_fm3": "MeV fm^-3",
        "free_energy_pressure_normalized_residual": "dimensionless",
        "free_energy_muB_residual_mev": "MeV",
        "free_energy_muB_normalized_residual": "dimensionless",
        "q5_beta_equilibrium_residual": "dimensionless",
        "q6_minus_q7_zero_temperature_residual": "dimensionless",
        "sound_speed_squared_definition_difference": "dimensionless",
        "sound_speed_squared_mu_minus_compose": "dimensionless",
    }
    descriptions: dict[str, str] = {name: name.replace("_", " ") for name in columns}
    descriptions["compose_heat_capacity_ratio_at_zero_temperature"] = (
        "Official CompOSE cold-branch Gamma=cP/cV convention fixed to one; "
        "cP and cV are not independently evaluated on a one-point T=0 axis"
    )
    descriptions["barotropic_adiabatic_index"] = "nB/P times dP/dnB"
    for index, (name, unit, description) in enumerate(_Q_FIELDS):
        columns[name] = q[:, index]
        columns[f"d{name}_dnB"] = dq_dn[:, index]
        units[name] = unit
        units[f"d{name}_dnB"] = "MeV fm^3" if index == 0 else "fm^3"
        descriptions[name] = description
        descriptions[f"d{name}_dnB"] = f"Native-density derivative of {description}"

    optional, optional_units, optional_descriptions = _optional_source_fields(
        cold_slice
    )
    for name, source_values in optional.items():
        columns[name] = _sample_optional_field(
            source_values,
            nodes,
            query,
            intervals,
            piecewise_constant=name == "phase_code",
        )
        units[name] = optional_units[name]
        descriptions[name] = optional_descriptions[name]
        availability_name = f"{name}_available"
        columns[availability_name] = np.isfinite(columns[name]).astype(float)
        units[availability_name] = "boolean 0/1"
        descriptions[availability_name] = (
            f"Availability mask for {optional_descriptions[name]}"
        )

    node_indices = np.full(len(query), -1.0)
    query_positions = np.searchsorted(query, nodes)
    valid_positions = query_positions < len(query)
    exact = np.zeros(len(nodes), dtype=bool)
    exact[valid_positions] = (
        query[query_positions[valid_positions]] == nodes[valid_positions]
    )
    node_indices[query_positions[exact]] = np.flatnonzero(exact)
    columns["source_node_position"] = node_indices
    units["source_node_position"] = "source-row index; -1 means interpolated"
    descriptions["source_node_position"] = "Exact source-node position or -1"

    diagnostics: list[ComposeProfileDiagnostic] = []
    source_report = cold_slice.report()
    diagnostics.extend(
        ComposeProfileDiagnostic(item.code, item.severity, 0, item.message)
        for item in source_report.diagnostics
    )
    sampled_findings = (
        _diagnostic(
            "sampled_pressure_nonpositive",
            "warning",
            pressure <= 0.0,
            "Reconstructed pressure is nonpositive at sampled profile points",
        ),
        _diagnostic(
            "sampled_pressure_gradient_nonpositive",
            "warning",
            dpressure_dn <= 0.0,
            "dP/dnB is nonpositive; the affected region remains visible",
        ),
        _diagnostic(
            "sampled_energy_gradient_nonpositive",
            "warning",
            denergy_dn <= 0.0,
            "d epsilon/dnB is nonpositive; the affected region remains visible",
        ),
        _diagnostic(
            "sampled_curve_sound_speed_nonpositive",
            "warning",
            cs2_curve <= 0.0,
            "The derivative of the reconstructed P(epsilon) curve is nonpositive",
        ),
        _diagnostic(
            "sampled_curve_sound_speed_acausal",
            "warning",
            cs2_curve > 1.0,
            "The derivative of the reconstructed P(epsilon) curve exceeds one",
        ),
        _diagnostic(
            "sampled_compose_sound_speed_nonpositive",
            "warning",
            cs2_compose <= 0.0,
            "The CompOSE thermodynamic sound-speed definition is nonpositive",
        ),
        _diagnostic(
            "sampled_compose_sound_speed_acausal",
            "warning",
            cs2_compose > 1.0,
            "The CompOSE thermodynamic sound-speed definition exceeds one",
        ),
        _diagnostic(
            "sampled_cold_beta_mu_sound_speed_nonpositive",
            "diagnostic",
            cs2_mu_derivative <= 0.0,
            "The cold-beta nB/muB dmuB/dnB sound-speed definition is nonpositive",
        ),
        _diagnostic(
            "sampled_cold_beta_mu_sound_speed_acausal",
            "diagnostic",
            cs2_mu_derivative > 1.0,
            "The cold-beta nB/muB dmuB/dnB sound-speed definition exceeds one",
        ),
        _diagnostic(
            "sampled_euler_residual_above_diagnostic_threshold",
            "diagnostic",
            euler_normalized > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            "Euler closure residual exceeds the declared diagnostic threshold",
        ),
        _diagnostic(
            "sampled_first_law_residual_above_diagnostic_threshold",
            "diagnostic",
            first_law_normalized > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            "d epsilon/dnB differs from the interpolated baryon chemical potential",
        ),
        _diagnostic(
            "sampled_gibbs_duhem_residual_above_diagnostic_threshold",
            "diagnostic",
            gibbs_duhem_normalized > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            "dP/dnB differs from nB dmuB/dnB",
        ),
        _diagnostic(
            "sampled_free_energy_pressure_residual_above_diagnostic_threshold",
            "diagnostic",
            free_pressure_normalized > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            "P differs from nB^2 d(F/A)/dnB under the native interpolation",
        ),
        _diagnostic(
            "sampled_free_energy_mu_residual_above_diagnostic_threshold",
            "diagnostic",
            free_mu_normalized > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            "muB differs from F/A+nB d(F/A)/dnB under the native interpolation",
        ),
        _diagnostic(
            "sampled_q5_above_diagnostic_threshold",
            "diagnostic",
            np.abs(q[:, 4]) > COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
            "|Q5| exceeds the declared beta-equilibrium diagnostic threshold",
        ),
        _diagnostic(
            "sampled_q6_minus_q7_above_diagnostic_threshold",
            "diagnostic",
            np.abs(q[:, 5] - q[:, 6]) > COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
            "|Q6-Q7| exceeds the declared zero-temperature diagnostic threshold",
        ),
    )
    diagnostics.extend(item for item in sampled_findings if item is not None)

    immutable_columns = MappingProxyType(
        {name: _readonly(value) for name, value in columns.items()}
    )
    return ComposeThermodynamicProfile(
        model_id=cold_slice.dataset.model_id,
        source_url=cold_slice.dataset.source_url,
        columns=immutable_columns,
        units=MappingProxyType(dict(units)),
        descriptions=MappingProxyType(dict(descriptions)),
        diagnostics=tuple(diagnostics),
        source_rows=len(nodes),
        provenance_json=json.dumps(
            {
                "dataset": cold_slice.dataset.provenance(),
                "matter_declaration": cold_slice.matter_declaration,
                "source_positions": list(cold_slice.source_positions),
                "source_baryon_density_min_fm3": float(nodes[0]),
                "source_baryon_density_max_fm3": float(nodes[-1]),
                "query_grid": {
                    "source": (
                        "caller_supplied"
                        if baryon_density_fm3 is not None
                        else "geometric_with_optional_source_node_union"
                    ),
                    "requested_geometric_points": (
                        None if baryon_density_fm3 is not None else int(points)
                    ),
                    "include_source_nodes_requested": bool(include_source_nodes),
                    "include_source_nodes_effective": bool(
                        include_source_nodes and baryon_density_fm3 is None
                    ),
                    "final_points": len(query),
                    "minimum_fm3": float(query[0]),
                    "maximum_fm3": float(query[-1]),
                    "float64_little_endian_sha256": hashlib.sha256(
                        np.asarray(query, dtype="<f8").tobytes(order="C")
                    ).hexdigest(),
                },
                "interpolation_policy": COMPOSE_NATIVE_INTERPOLATION_POLICY,
                "source_values_modified": False,
                "source_rows_removed": False,
                "optional_quantity_missing_value_policy": (
                    "NaN unless both interval endpoints are present; exact source-node "
                    "values are retained. This intentionally differs from the official "
                    "zero-fill convention."
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )


__all__ = [
    "COMPOSE_NATIVE_INTERPOLATION_POLICY",
    "COMPOSE_NATIVE_THERMODYNAMIC_SCHEMA_VERSION",
    "ComposeProfileDiagnostic",
    "ComposeThermodynamicProfile",
    "interpolate_compose_thermodynamics",
]
