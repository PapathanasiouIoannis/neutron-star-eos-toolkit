"""Construction of continuous cold barotropes from parsed CompOSE data."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from neutron_star_eos.compose_dataset import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_DATASET_SCHEMA_VERSION,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
    COMPOSE_FORMAT_AUTHORITY,
    ComposeColdSlice,
    ComposeDataset,
    ComposeSliceReport,
    load_compose_dataset,
)
from neutron_star_eos.eos import (
    CANONICAL_ENERGY_DENSITY_CONVENTION,
    CANONICAL_UNITS,
    EOS_INPUT_SCHEMA_VERSION,
    EosDomainError,
    EosInputError,
    EosValidationIssue,
    EosValidationReport,
    _domain_values,
    _scalar_or_array,
)


COMPOSE_BAROTROPE_SCHEMA_VERSION = "compose_native_density_barotrope_v2"
COMPOSE_INTERPOLATION_POLICY = "separate_log_pchip_in_native_baryon_density_v1"
COMPOSE_ORDERING_POLICIES = (
    "strict",
    "diagnostic_monotone_subsequence",
    "diagnostic_keep_later_monotone_subsequence",
)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = values.view()
    result.setflags(write=False)
    return result


class ComposeEos:
    """Continuous CompOSE barotrope constructed in native baryon density.

    Pressure and total energy density are interpolated separately as functions
    of baryon density. The sound speed follows from their derivative ratio.
    This is a declared toolkit policy, not a source-supplied analytical form.
    """

    surface_boundary_kind = "finite_source_pressure_not_vacuum"
    tidal_capability_status = "unavailable_positive_pressure_source_boundary"
    preferred_stellar_integration_variable = "log_pressure"

    def __init__(
        self,
        *,
        cold_slice: ComposeColdSlice,
        source_slice_report: ComposeSliceReport,
        selection: dict[str, Any],
    ) -> None:
        report = cold_slice.report()
        if not report.continuous_barotrope_available:
            raise EosInputError(
                "selected CompOSE rows do not form one continuous invertible barotrope; "
                "inspect the dataset report and choose an explicit monotone source block "
                "or provide a future transition policy"
            )
        density = np.asarray(cold_slice.baryon_density_fm3, dtype=float)
        pressure = np.asarray(cold_slice.pressure_mev_fm3, dtype=float)
        epsilon = np.asarray(cold_slice.energy_density_mev_fm3, dtype=float)
        chemical_potential = np.asarray(
            cold_slice.baryon_chemical_potential_mev, dtype=float
        )
        if len(density) < 4:
            raise EosInputError("continuous CompOSE barotrope requires at least four rows")
        if (
            np.any(density <= 0.0)
            or np.any(pressure <= 0.0)
            or np.any(epsilon <= 0.0)
            or np.any(np.diff(density) <= 0.0)
            or np.any(np.diff(pressure) <= 0.0)
            or np.any(np.diff(epsilon) <= 0.0)
        ):
            raise EosInputError(
                "continuous CompOSE barotrope requires positive strictly increasing "
                "nB, pressure, and total energy density"
            )
        self.model_name = cold_slice.dataset.model_id
        self.source = cold_slice.dataset.source_url
        self.energy_density_convention = CANONICAL_ENERGY_DENSITY_CONVENTION
        self._cold_slice = cold_slice
        self._source_slice_report = source_slice_report
        self._slice_report = report
        self._selection = copy.deepcopy(selection)
        self._density = density.copy()
        self._pressure = pressure.copy()
        self._epsilon = epsilon.copy()
        self._chemical_potential = chemical_potential.copy()
        for array in (
            self._density,
            self._pressure,
            self._epsilon,
            self._chemical_potential,
        ):
            array.setflags(write=False)
        self.energy_density_min_mev_fm3 = float(epsilon[0])
        self.energy_density_max_mev_fm3 = float(epsilon[-1])
        self.pressure_min_mev_fm3 = float(pressure[0])
        self.pressure_max_mev_fm3 = float(pressure[-1])
        self._log_density = np.log(density)
        self._log_pressure = np.log(pressure)
        self._log_epsilon = np.log(epsilon)
        self._pressure_from_density = PchipInterpolator(
            self._log_density, self._log_pressure, extrapolate=False
        )
        self._epsilon_from_density = PchipInterpolator(
            self._log_density, self._log_epsilon, extrapolate=False
        )
        self._dlog_pressure = self._pressure_from_density.derivative()
        self._dlog_epsilon = self._epsilon_from_density.derivative()

    @property
    def baryon_density_fm3(self) -> np.ndarray:
        return _readonly(self._density)

    @property
    def pressure_mev_fm3(self) -> np.ndarray:
        return _readonly(self._pressure)

    @property
    def energy_density_mev_fm3(self) -> np.ndarray:
        return _readonly(self._epsilon)

    @property
    def baryon_chemical_potential_mev(self) -> np.ndarray:
        return _readonly(self._chemical_potential)

    @property
    def phase_codes(self) -> tuple[int | None, ...] | None:
        return self._cold_slice.phase_codes

    @property
    def compose_metadata(self) -> dict[str, Any]:
        return self.provenance()["compose"]

    @property
    def slice_report(self) -> ComposeSliceReport:
        return self._slice_report

    def _density_log_from_value(
        self,
        values: np.ndarray,
        *,
        nodes: np.ndarray,
        interpolant: PchipInterpolator,
        lower_value: float,
        upper_value: float,
    ) -> np.ndarray:
        result = np.empty_like(values, dtype=float)
        flattened = result.reshape(-1)
        for output_index, value in enumerate(values.reshape(-1)):
            candidate = float(value)
            if candidate == lower_value:
                flattened[output_index] = self._log_density[0]
                continue
            if candidate == upper_value:
                flattened[output_index] = self._log_density[-1]
                continue
            interval = int(np.searchsorted(nodes, candidate, side="right") - 1)
            interval = min(max(interval, 0), len(nodes) - 2)
            target = math.log(candidate)
            flattened[output_index] = brentq(
                lambda log_density: float(interpolant(log_density) - target),
                float(self._log_density[interval]),
                float(self._log_density[interval + 1]),
                xtol=5.0e-15,
                rtol=4.0 * np.finfo(float).eps,
            )
        return result

    def _log_density_from_pressure(self, pressure: np.ndarray) -> np.ndarray:
        return self._density_log_from_value(
            pressure,
            nodes=self._pressure,
            interpolant=self._pressure_from_density,
            lower_value=self.pressure_min_mev_fm3,
            upper_value=self.pressure_max_mev_fm3,
        )

    def _log_density_from_epsilon(self, epsilon: np.ndarray) -> np.ndarray:
        return self._density_log_from_value(
            epsilon,
            nodes=self._epsilon,
            interpolant=self._epsilon_from_density,
            lower_value=self.energy_density_min_mev_fm3,
            upper_value=self.energy_density_max_mev_fm3,
        )

    def _cs2_from_log_density(self, log_density: np.ndarray) -> np.ndarray:
        pressure = np.exp(self._pressure_from_density(log_density))
        epsilon = np.exp(self._epsilon_from_density(log_density))
        dlog_pressure = np.asarray(self._dlog_pressure(log_density), dtype=float)
        dlog_epsilon = np.asarray(self._dlog_epsilon(log_density), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return pressure * dlog_pressure / (epsilon * dlog_epsilon)

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> Any:
        values, scalar = _domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        log_density = self._log_density_from_epsilon(values)
        result = np.exp(self._pressure_from_density(log_density))
        result = np.where(values == self.energy_density_min_mev_fm3, self.pressure_min_mev_fm3, result)
        result = np.where(values == self.energy_density_max_mev_fm3, self.pressure_max_mev_fm3, result)
        if not np.all(np.isfinite(result)):
            raise EosDomainError("CompOSE interpolation left the selected density domain")
        return _scalar_or_array(result, scalar)

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> Any:
        values, scalar = _domain_values(
            pressure_mev_fm3,
            name="pressure_mev_fm3",
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
        )
        log_density = self._log_density_from_pressure(values)
        result = np.exp(self._epsilon_from_density(log_density))
        result = np.where(values == self.pressure_min_mev_fm3, self.energy_density_min_mev_fm3, result)
        result = np.where(values == self.pressure_max_mev_fm3, self.energy_density_max_mev_fm3, result)
        if not np.all(np.isfinite(result)):
            raise EosDomainError("CompOSE interpolation left the selected pressure domain")
        return _scalar_or_array(result, scalar)

    def sound_speed_squared_from_energy_density(self, energy_density_mev_fm3: Any) -> Any:
        values, scalar = _domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        result = self._cs2_from_log_density(self._log_density_from_epsilon(values))
        if not np.all(np.isfinite(result)):
            raise EosInputError("CompOSE sound-speed derivative is nonfinite")
        return _scalar_or_array(result, scalar)

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        values, _scalar = _domain_values(
            pressure_mev_fm3,
            name="pressure_mev_fm3",
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
        )
        log_density = self._log_density_from_pressure(values)
        epsilon = float(np.exp(self._epsilon_from_density(log_density)))
        cs2 = float(self._cs2_from_log_density(log_density))
        return epsilon, cs2

    def _validation_log_density(self, points: int) -> np.ndarray:
        intervals = len(self._log_density) - 1
        points_per_interval = max(9, int(np.ceil((int(points) - 1) / intervals)))
        candidates: list[float] = []
        pressure_coefficients = self._pressure_from_density.c
        epsilon_coefficients = self._epsilon_from_density.c
        for index in range(intervals):
            left = float(self._log_density[index])
            right = float(self._log_density[index + 1])
            width = right - left
            candidates.extend(
                float(value)
                for value in np.linspace(left, right, points_per_interval, endpoint=False)
            )
            fp_cubic, fp_quadratic, fp_linear, fp_constant = pressure_coefficients[:, index]
            ep_cubic, ep_quadratic, ep_linear, ep_constant = epsilon_coefficients[:, index]
            log_pressure = Polynomial(
                (fp_constant, fp_linear, fp_quadratic, fp_cubic)
            )
            log_epsilon = Polynomial(
                (ep_constant, ep_linear, ep_quadratic, ep_cubic)
            )
            pressure_derivative = log_pressure.deriv()
            epsilon_derivative = log_epsilon.deriv()
            cs2_stationarity = (
                (pressure_derivative - epsilon_derivative)
                * pressure_derivative
                * epsilon_derivative
                + pressure_derivative.deriv() * epsilon_derivative
                - epsilon_derivative.deriv() * pressure_derivative
            )
            endpoint_tolerance = 64.0 * np.finfo(float).eps * max(
                1.0, abs(left), abs(right), width
            )
            for polynomial in (
                pressure_derivative,
                epsilon_derivative,
                cs2_stationarity,
            ):
                for root in polynomial.roots():
                    if abs(float(np.imag(root))) > 1.0e-10:
                        continue
                    local = float(np.real(root))
                    if -endpoint_tolerance <= local <= width + endpoint_tolerance:
                        if abs(local) <= endpoint_tolerance:
                            candidates.append(left)
                        elif abs(local - width) <= endpoint_tolerance:
                            candidates.append(right)
                        else:
                            candidates.append(left + local)
        candidates.append(float(self._log_density[-1]))
        return np.unique(np.asarray(candidates, dtype=float))

    def validate(self, *, points: int = 2049) -> EosValidationReport:
        if int(points) < 17:
            raise ValueError("validation points must be at least 17")
        log_density = self._validation_log_density(int(points))
        pressure = np.exp(self._pressure_from_density(log_density))
        epsilon = np.exp(self._epsilon_from_density(log_density))
        cs2 = self._cs2_from_log_density(log_density)
        issues: list[EosValidationIssue] = []
        if np.any(~np.isfinite(pressure)) or np.any(~np.isfinite(epsilon)) or np.any(~np.isfinite(cs2)):
            issues.append(EosValidationIssue("nonfinite", "interpolated pressure, energy density, or cs2 is nonfinite"))
        if np.any(pressure <= 0.0) or np.any(epsilon <= 0.0):
            issues.append(EosValidationIssue("nonpositive_thermodynamics", "pressure and total energy density must remain positive"))
        if np.any(np.diff(pressure) <= 0.0) or np.any(np.diff(epsilon) <= 0.0):
            issues.append(EosValidationIssue("nonmonotone_native_interpolation", "native-density interpolation must remain strictly invertible"))
        if np.any(cs2 <= 0.0):
            issues.append(EosValidationIssue("mechanical_instability", "dP/dE must remain positive"))
        if np.any(cs2 > 1.0):
            issues.append(EosValidationIssue("acausal", "dP/dE must not exceed one"))
        return EosValidationReport(
            model_name=self.model_name,
            assessed_points=len(log_density),
            pressure_min_mev_fm3=float(np.min(pressure)),
            pressure_max_mev_fm3=float(np.max(pressure)),
            energy_density_min_mev_fm3=float(np.min(epsilon)),
            energy_density_max_mev_fm3=float(np.max(epsilon)),
            cs2_min=float(np.min(cs2)),
            cs2_max=float(np.max(cs2)),
            issues=tuple(issues),
        )

    def provenance(self) -> dict[str, Any]:
        dataset = self._cold_slice.dataset
        return {
            "schema_version": EOS_INPUT_SCHEMA_VERSION,
            "adapter": COMPOSE_BAROTROPE_SCHEMA_VERSION,
            "model_name": self.model_name,
            "source": self.source,
            "units": CANONICAL_UNITS,
            "energy_density_convention": self.energy_density_convention,
            "rows": len(self._density),
            "interpolation": COMPOSE_INTERPOLATION_POLICY,
            "interpolation_authority": "toolkit_declared_not_source_supplied",
            "extrapolation": "forbidden",
            "discontinuous_barotropes": "require_explicit_future_transition_policy",
            "selection": copy.deepcopy(self._selection),
            "domain": {
                "baryon_density_min_fm3": float(self._density[0]),
                "baryon_density_max_fm3": float(self._density[-1]),
                "energy_density_min_MeV_fm3": self.energy_density_min_mev_fm3,
                "energy_density_max_MeV_fm3": self.energy_density_max_mev_fm3,
                "pressure_min_MeV_fm3": self.pressure_min_mev_fm3,
                "pressure_max_MeV_fm3": self.pressure_max_mev_fm3,
            },
            "validation_policy": {
                "coordinate": "native_baryon_density",
                "interval_samples_minimum": 9,
                "bounded_cs2_extrema_search": True,
                "claim": "numerically_assessed_interpolant_not_analytical_proof",
            },
            "stellar_surface": {
                "kind": self.surface_boundary_kind,
                "pressure_MeV_fm3": self.pressure_min_mev_fm3,
                "tidal_capability": self.tidal_capability_status,
            },
            "compose": {
                "dataset": dataset.provenance(),
                "slice_report": self._slice_report.to_dict(),
                "source_diagnostics_are_acceptance_gates": False,
                "cold_euler_closure_relative_diagnostic_tolerance": COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
                "cold_condition_absolute_diagnostic_tolerance": COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
                "parser_version": COMPOSE_DATASET_SCHEMA_VERSION,
                "format_authority": COMPOSE_FORMAT_AUTHORITY,
                "model_id": dataset.model_id,
                "source_url": dataset.source_url,
                "matter_declaration": self._cold_slice.matter_declaration,
                "includes_leptons": True,
                "lepton_indicator_Il": dataset.lepton_indicator,
                "source_rows": len(dataset.baryon_density.values),
                "retained_rows": len(self._density),
                "thermodynamic_duplicate_indices_last_row_wins": dataset.thermodynamic_duplicate_indices,
                "phase_codes_available": self.phase_codes is not None,
                "phase_code_rows_missing": self._slice_report.missing_phase_codes,
                "phase_code_changes": self._slice_report.phase_code_changes,
                "phase_codes_interpreted_as_discontinuities": False,
                "cold_euler_closure_source_maximum_normalized_residual": self._source_slice_report.euler_maximum_normalized_residual,
                "cold_euler_closure_retained_maximum_normalized_residual": self._slice_report.euler_maximum_normalized_residual,
                "beta_equilibrium_Q5_source_maximum_absolute_residual": self._source_slice_report.q5_maximum_absolute_residual,
                "beta_equilibrium_Q5_retained_maximum_absolute_residual": self._slice_report.q5_maximum_absolute_residual,
                "zero_temperature_Q6_minus_Q7_source_maximum_absolute_residual": self._source_slice_report.q6_minus_q7_maximum_absolute_residual,
                "zero_temperature_Q6_minus_Q7_retained_maximum_absolute_residual": self._slice_report.q6_minus_q7_maximum_absolute_residual,
            },
        }


def build_compose_eos(
    dataset_or_slice: ComposeDataset | ComposeColdSlice,
    *,
    matter: str = "cold_beta_equilibrated",
    includes_leptons: bool = True,
    baryon_density_min_fm3: float | None = None,
    baryon_density_max_fm3: float | None = None,
    ordering_policy: str = "strict",
) -> ComposeEos:
    """Construct one explicitly selected continuous CompOSE barotrope.

    ``strict`` preserves the historical fail-closed behavior.  The explicit
    The diagnostic policies remove no ambiguity silently: they keep source
    values unchanged, choose the earlier or later member of a local ordering
    conflict, and record every omitted source position in provenance. Their
    separation is a sensitivity measure, not a physical seam/transition model.
    """

    if isinstance(dataset_or_slice, ComposeDataset):
        cold_slice = dataset_or_slice.cold_beta_equilibrium_slice(
            matter=matter,
            includes_leptons=includes_leptons,
        )
    elif isinstance(dataset_or_slice, ComposeColdSlice):
        cold_slice = dataset_or_slice
    else:
        raise TypeError("build_compose_eos expects ComposeDataset or ComposeColdSlice")
    if ordering_policy not in COMPOSE_ORDERING_POLICIES:
        raise EosInputError(
            f"ordering_policy must be one of {COMPOSE_ORDERING_POLICIES}"
        )
    source_slice_report = cold_slice.report()
    selected_source = cold_slice.selected_domain(
        baryon_density_min_fm3=baryon_density_min_fm3,
        baryon_density_max_fm3=baryon_density_max_fm3,
    )
    selected = selected_source
    if ordering_policy == "diagnostic_monotone_subsequence":
        selected = selected_source.diagnostic_monotone_subsequence(
            conflict_policy="keep_first"
        )
    elif ordering_policy == "diagnostic_keep_later_monotone_subsequence":
        selected = selected_source.diagnostic_monotone_subsequence(
            conflict_policy="keep_later"
        )
    retained_positions = set(selected.source_positions)
    omitted_positions = tuple(
        position
        for position in selected_source.source_positions
        if position not in retained_positions
    )
    return ComposeEos(
        cold_slice=selected,
        source_slice_report=source_slice_report,
        selection={
            "requested_baryon_density_min_fm3": baryon_density_min_fm3,
            "requested_baryon_density_max_fm3": baryon_density_max_fm3,
            "last_retained_source_node_fm3": float(selected.baryon_density_fm3[-1]),
            "ordering_policy": ordering_policy,
            "selected_source_rows_before_ordering_policy": len(selected_source.rows),
            "retained_source_positions": list(selected.source_positions),
            "omitted_source_positions": list(omitted_positions),
            "omitted_source_rows": len(omitted_positions),
            "diagnostic_reduction_is_physical_transition_policy": False,
        },
    )


def load_compose_eos(
    path_or_zip: str | Path,
    *,
    model_id: str,
    source_url: str,
    matter: str,
    includes_leptons: bool,
    baryon_density_min_fm3: float | None = None,
    baryon_density_max_fm3: float | None = None,
    ordering_policy: str = "strict",
) -> ComposeEos:
    """Compatibility wrapper: parse a dataset, then build a continuous barotrope."""

    if matter != "cold_beta_equilibrated":
        raise EosInputError(
            "v2 stellar CompOSE input must explicitly be cold beta-equilibrated matter"
        )
    if includes_leptons is not True:
        raise EosInputError("v2 stellar CompOSE input must explicitly include leptons")
    dataset = load_compose_dataset(path_or_zip, model_id=model_id, source_url=source_url)
    return build_compose_eos(
        dataset,
        matter=matter,
        includes_leptons=includes_leptons,
        baryon_density_min_fm3=baryon_density_min_fm3,
        baryon_density_max_fm3=baryon_density_max_fm3,
        ordering_policy=ordering_policy,
    )


__all__ = [
    "COMPOSE_BAROTROPE_SCHEMA_VERSION",
    "COMPOSE_INTERPOLATION_POLICY",
    "COMPOSE_ORDERING_POLICIES",
    "ComposeColdSlice",
    "ComposeDataset",
    "ComposeEos",
    "ComposeSliceReport",
    "build_compose_eos",
    "load_compose_dataset",
    "load_compose_eos",
]
