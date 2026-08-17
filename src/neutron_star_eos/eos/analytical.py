"""Adapter for a user-supplied analytical relation P(epsilon)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from neutron_star_eos.eos.core import (
    CANONICAL_ENERGY_DENSITY_CONVENTION,
    CANONICAL_UNITS,
    EOS_INPUT_SCHEMA_VERSION,
    EosDomainError,
    EosInputError,
    domain_values,
    evaluate_user_function,
    finite_float,
    scalar_or_array,
)
from neutron_star_eos.eos.validation import (
    EosValidationIssue,
    EosValidationReport,
    validate_eos_sampled,
)

ANALYTICAL_DERIVATIVE_RELATIVE_TOLERANCE = 2.0e-4
ANALYTICAL_DERIVATIVE_ABSOLUTE_TOLERANCE = 2.0e-6
ANALYTICAL_INVERSE_RELATIVE_TOLERANCE = 2.0e-8
ANALYTICAL_INVERSE_ABSOLUTE_TOLERANCE = 2.0e-10
ANALYTICAL_FINGERPRINT_GRID_POINTS = 2049
ANALYTICAL_INVERSE_FINGERPRINT_GRID_POINTS = 129


class AnalyticalEos:
    """A continuous cold barotrope defined by analytical functions.

    Supply pressure P(epsilon), its derivative dP/d(epsilon), a finite total
    energy-density domain in MeV fm^-3, and a source description.  If an
    analytical inverse is omitted, pressure is inverted only inside the
    declared domain with a bracketed root solve.
    """

    surface_boundary_kind = "finite_source_pressure_not_vacuum"
    tidal_capability_status = "unavailable_positive_pressure_source_boundary"

    def __init__(
        self,
        *,
        name: str,
        pressure_from_energy_density: Callable[[Any], Any],
        sound_speed_squared_from_energy_density: Callable[[Any], Any],
        energy_density_domain_mev_fm3: tuple[float, float],
        source: str,
        energy_density_from_pressure: Callable[[Any], Any] | None = None,
        energy_density_convention: str = CANONICAL_ENERGY_DENSITY_CONVENTION,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise EosInputError("analytical EoS name must be non-empty")
        if not isinstance(source, str) or not source.strip():
            raise EosInputError("analytical EoS source must be non-empty")
        if energy_density_convention != CANONICAL_ENERGY_DENSITY_CONVENTION:
            raise EosInputError(
                "energy density must be total energy density including rest mass"
            )
        try:
            raw_lower, raw_upper = energy_density_domain_mev_fm3
        except (TypeError, ValueError) as exc:
            raise EosInputError(
                "energy-density domain must contain exactly two values"
            ) from exc
        lower = finite_float("energy_density_domain lower", raw_lower)
        upper = finite_float("energy_density_domain upper", raw_upper)
        if lower <= 0.0 or upper <= lower:
            raise EosInputError("energy-density domain must satisfy 0 < lower < upper")

        self.model_name = name.strip()
        self.source = source.strip()
        self.energy_density_convention = energy_density_convention
        self.energy_density_min_mev_fm3 = lower
        self.energy_density_max_mev_fm3 = upper
        self._pressure = pressure_from_energy_density
        self._cs2 = sound_speed_squared_from_energy_density
        self._inverse = energy_density_from_pressure

        endpoint_pressure = evaluate_user_function(
            self._pressure, np.asarray([lower, upper]), name="pressure"
        )
        if endpoint_pressure.shape != (2,) or not np.all(
            np.isfinite(endpoint_pressure)
        ):
            raise EosInputError(
                "analytical pressure function must support a two-value array"
            )
        if endpoint_pressure[0] <= 0.0 or endpoint_pressure[1] <= endpoint_pressure[0]:
            raise EosInputError(
                "analytical endpoint pressure must be positive and increasing"
            )
        self.pressure_min_mev_fm3 = float(endpoint_pressure[0])
        self.pressure_max_mev_fm3 = float(endpoint_pressure[1])

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> Any:
        """Evaluate P(epsilon) inside the declared MeV fm^-3 domain."""

        values, scalar = domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        result = evaluate_user_function(self._pressure, values, name="pressure")
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise EosInputError("analytical pressure returned invalid values")
        return scalar_or_array(result, scalar)

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> Any:
        """Evaluate c_s^2=dP/d(epsilon) with c=1."""

        values, scalar = domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        result = evaluate_user_function(self._cs2, values, name="sound speed")
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise EosInputError("analytical sound speed returned invalid values")
        return scalar_or_array(result, scalar)

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> Any:
        """Invert P(epsilon) without extrapolating beyond its declared domain."""

        values, scalar = domain_values(
            pressure_mev_fm3,
            name="pressure_mev_fm3",
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
        )
        if self._inverse is not None:
            result = evaluate_user_function(
                self._inverse, values, name="energy-density inverse"
            )
        else:
            result = np.empty_like(values, dtype=float)
            flat = result.reshape(-1)
            for index, pressure in enumerate(values.reshape(-1)):
                if pressure == self.pressure_min_mev_fm3:
                    flat[index] = self.energy_density_min_mev_fm3
                elif pressure == self.pressure_max_mev_fm3:
                    flat[index] = self.energy_density_max_mev_fm3
                else:
                    flat[index] = brentq(
                        lambda epsilon, pressure=pressure: (
                            float(self._pressure(epsilon)) - float(pressure)
                        ),
                        self.energy_density_min_mev_fm3,
                        self.energy_density_max_mev_fm3,
                        xtol=5.0e-14,
                        rtol=4.0 * np.finfo(float).eps,
                    )
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise EosInputError("analytical pressure inversion returned invalid values")
        if np.any(result < self.energy_density_min_mev_fm3) or np.any(
            result > self.energy_density_max_mev_fm3
        ):
            raise EosDomainError(
                "analytical pressure inversion left the declared energy-density domain"
            )
        return scalar_or_array(result, scalar)

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        epsilon = float(self.energy_density_from_pressure(pressure_mev_fm3))
        cs2 = float(self.sound_speed_squared_from_energy_density(epsilon))
        return epsilon, cs2

    def validate(self, *, points: int = 2049) -> EosValidationReport:
        """Check stability, causality, derivative, and inverse consistency."""

        report = validate_eos_sampled(self, points=points)
        epsilon = np.geomspace(
            self.energy_density_min_mev_fm3,
            self.energy_density_max_mev_fm3,
            int(points),
        )
        pressure = np.asarray(self.pressure_from_energy_density(epsilon), dtype=float)
        supplied_cs2 = np.asarray(
            self.sound_speed_squared_from_energy_density(epsilon), dtype=float
        )
        numerical_cs2 = np.gradient(pressure, epsilon, edge_order=2)
        interior_difference = np.abs(supplied_cs2[1:-1] - numerical_cs2[1:-1])
        interior_scale = ANALYTICAL_DERIVATIVE_ABSOLUTE_TOLERANCE + (
            ANALYTICAL_DERIVATIVE_RELATIVE_TOLERANCE
            * np.maximum(np.abs(supplied_cs2[1:-1]), np.abs(numerical_cs2[1:-1]))
        )
        issues = list(report.issues)
        if np.any(interior_difference > interior_scale):
            issues.append(
                EosValidationIssue(
                    "inconsistent_sound_speed",
                    "supplied sound speed squared does not agree with dP/dE "
                    "on the declared analytical domain",
                )
            )

        probe_count = min(int(points), 129)
        probe_epsilon = np.geomspace(
            self.energy_density_min_mev_fm3,
            self.energy_density_max_mev_fm3,
            probe_count,
        )
        probe_pressure = np.asarray(
            self.pressure_from_energy_density(probe_epsilon), dtype=float
        )
        try:
            recovered_epsilon = np.asarray(
                self.energy_density_from_pressure(probe_pressure), dtype=float
            )
            recovered_pressure = np.asarray(
                self.pressure_from_energy_density(recovered_epsilon), dtype=float
            )
            epsilon_consistent = np.allclose(
                recovered_epsilon,
                probe_epsilon,
                rtol=ANALYTICAL_INVERSE_RELATIVE_TOLERANCE,
                atol=ANALYTICAL_INVERSE_ABSOLUTE_TOLERANCE,
            )
            pressure_consistent = np.allclose(
                recovered_pressure,
                probe_pressure,
                rtol=ANALYTICAL_INVERSE_RELATIVE_TOLERANCE,
                atol=ANALYTICAL_INVERSE_ABSOLUTE_TOLERANCE,
            )
        except EosInputError:
            epsilon_consistent = False
            pressure_consistent = False
        if not epsilon_consistent or not pressure_consistent:
            issues.append(
                EosValidationIssue(
                    "inconsistent_pressure_inverse",
                    "epsilon(P) does not round-trip the authoritative P(epsilon) "
                    "on the declared analytical domain",
                )
            )
        return replace(report, issues=tuple(issues))

    def provenance(self) -> dict[str, Any]:
        """Describe and fingerprint evaluated forward and inverse behaviour."""

        epsilon = np.geomspace(
            self.energy_density_min_mev_fm3,
            self.energy_density_max_mev_fm3,
            ANALYTICAL_FINGERPRINT_GRID_POINTS,
        )
        pressure = np.asarray(self.pressure_from_energy_density(epsilon), dtype=float)
        cs2 = np.asarray(
            self.sound_speed_squared_from_energy_density(epsilon), dtype=float
        )
        inverse_indices = np.linspace(
            0,
            len(epsilon) - 1,
            ANALYTICAL_INVERSE_FINGERPRINT_GRID_POINTS,
            dtype=int,
        )
        inverse_pressure = pressure[inverse_indices]
        recovered_epsilon = np.asarray(
            self.energy_density_from_pressure(inverse_pressure), dtype=float
        )
        fingerprint = hashlib.sha256()
        for name, values in (
            ("energy_density_mev_fm3", epsilon),
            ("pressure_mev_fm3", pressure),
            ("sound_speed_squared", cs2),
            ("inverse_pressure_mev_fm3", inverse_pressure),
            ("recovered_energy_density_mev_fm3", recovered_epsilon),
        ):
            fingerprint.update(name.encode("ascii") + b"\0")
            fingerprint.update(
                np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")
            )
        return {
            "schema_version": EOS_INPUT_SCHEMA_VERSION,
            "adapter": "analytical_eos_v1",
            "model_name": self.model_name,
            "source": self.source,
            "units": CANONICAL_UNITS,
            "energy_density_convention": self.energy_density_convention,
            "domain": {
                "energy_density_min_MeV_fm3": self.energy_density_min_mev_fm3,
                "energy_density_max_MeV_fm3": self.energy_density_max_mev_fm3,
                "pressure_min_MeV_fm3": self.pressure_min_mev_fm3,
                "pressure_max_MeV_fm3": self.pressure_max_mev_fm3,
            },
            "extrapolation": "forbidden",
            "callable_fingerprint": {
                "policy": "float64_le_sha256_forward_and_inverse_behavior_v2",
                "forward_grid_points": ANALYTICAL_FINGERPRINT_GRID_POINTS,
                "inverse_grid_points": ANALYTICAL_INVERSE_FINGERPRINT_GRID_POINTS,
                "inverse_pressure_grid": (
                    "forward-curve pressures at evenly spaced forward-grid indices"
                ),
                "sha256": fingerprint.hexdigest(),
                "scope": "evaluated callable behavior, not source-code identity",
            },
            "validation_policy": {
                "analytical_grid_points_default": 2049,
                "derivative_relative_tolerance": (
                    ANALYTICAL_DERIVATIVE_RELATIVE_TOLERANCE
                ),
                "derivative_absolute_tolerance": (
                    ANALYTICAL_DERIVATIVE_ABSOLUTE_TOLERANCE
                ),
                "inverse_relative_tolerance": ANALYTICAL_INVERSE_RELATIVE_TOLERANCE,
                "inverse_absolute_tolerance": ANALYTICAL_INVERSE_ABSOLUTE_TOLERANCE,
            },
            "stellar_surface": {
                "kind": self.surface_boundary_kind,
                "pressure_MeV_fm3": self.pressure_min_mev_fm3,
                "tidal_capability": self.tidal_capability_status,
            },
        }
