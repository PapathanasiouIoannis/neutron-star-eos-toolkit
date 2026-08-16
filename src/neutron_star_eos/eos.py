"""Small public contract for cold one-fluid equations of state.

All production-facing quantities use MeV/fm^3 and ``c = 1``.  Adapters in
this package are deliberately strict: they declare a finite domain, never
extrapolate, and must pass their physics report before a stellar helper will
use them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq

EOS_INPUT_SCHEMA_VERSION = "cold_barotrope_input_v1"
CANONICAL_ENERGY_DENSITY_CONVENTION = "total_including_rest_mass"
CANONICAL_UNITS = "MeV/fm^3"
ANALYTICAL_DERIVATIVE_RELATIVE_TOLERANCE = 2.0e-4
ANALYTICAL_DERIVATIVE_ABSOLUTE_TOLERANCE = 2.0e-6
ANALYTICAL_INVERSE_RELATIVE_TOLERANCE = 2.0e-8
ANALYTICAL_INVERSE_ABSOLUTE_TOLERANCE = 2.0e-10
ANALYTICAL_FINGERPRINT_GRID_POINTS = 2049
ANALYTICAL_INVERSE_FINGERPRINT_GRID_POINTS = 129


class EosInputError(ValueError):
    """Base error for invalid user-supplied EoS material."""


class EosDomainError(EosInputError):
    """Raised when an EoS is evaluated outside its declared domain."""


def _eos_provenance_sha256(eos: "ColdBarotrope") -> str:
    """Return one canonical identity for the exact declared EoS adapter."""

    encoded = json.dumps(
        eos.provenance(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EosValidationIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EosValidationReport:
    model_name: str
    assessed_points: int
    pressure_min_mev_fm3: float
    pressure_max_mev_fm3: float
    energy_density_min_mev_fm3: float
    energy_density_max_mev_fm3: float
    cs2_min: float
    cs2_max: float
    issues: tuple[EosValidationIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def require_pass(self) -> "EosValidationReport":
        if not self.passed:
            details = "; ".join(f"{item.code}: {item.message}" for item in self.issues)
            raise EosInputError(f"EoS validation failed: {details}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EOS_INPUT_SCHEMA_VERSION,
            "model_name": self.model_name,
            "status": "pass" if self.passed else "fail",
            "assessed_points": self.assessed_points,
            "domain": {
                "pressure_min_MeV_fm3": self.pressure_min_mev_fm3,
                "pressure_max_MeV_fm3": self.pressure_max_mev_fm3,
                "energy_density_min_MeV_fm3": self.energy_density_min_mev_fm3,
                "energy_density_max_MeV_fm3": self.energy_density_max_mev_fm3,
            },
            "sound_speed_squared": {"minimum": self.cs2_min, "maximum": self.cs2_max},
            "issues": [
                {"code": item.code, "message": item.message} for item in self.issues
            ],
        }


@runtime_checkable
class ColdBarotrope(Protocol):
    """Continuous cold-barotrope interface used by the public toolkit."""

    model_name: str
    pressure_min_mev_fm3: float
    pressure_max_mev_fm3: float
    energy_density_min_mev_fm3: float
    energy_density_max_mev_fm3: float
    surface_boundary_kind: str
    tidal_capability_status: str

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> Any: ...

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> Any: ...

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> Any: ...

    def validate(self, *, points: int = 2049) -> EosValidationReport: ...

    def provenance(self) -> dict[str, Any]: ...

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]: ...


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise EosInputError(f"{name} must be finite")
    return result


def _domain_values(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise EosInputError(f"{name} must contain only finite values")
    if np.any(array < lower) or np.any(array > upper):
        raise EosDomainError(
            f"{name} is outside the declared interval [{lower!r}, {upper!r}]"
        )
    return array, array.ndim == 0


def _scalar_or_array(value: np.ndarray, scalar: bool) -> float | np.ndarray:
    return float(np.asarray(value)) if scalar else np.asarray(value, dtype=float)


def _evaluate_user_function(
    function: Callable[[Any], Any], values: np.ndarray, *, name: str
) -> np.ndarray:
    """Accept ordinary scalar functions as well as NumPy-aware functions."""
    try:
        result = np.asarray(function(values), dtype=float)
        if result.shape == values.shape:
            return result
    except (TypeError, ValueError, ArithmeticError):
        pass
    try:
        result = np.asarray(
            [float(function(float(value))) for value in values.reshape(-1)], dtype=float
        ).reshape(values.shape)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise EosInputError(
            f"analytical {name} could not evaluate the declared domain"
        ) from exc
    return result


class AnalyticalEos:
    """Adapter for a continuous analytical cold barotrope.

    The user supplies the authoritative forward pressure and sound-speed
    functions.  If no analytical inverse is supplied, a bracketed root solve
    is used strictly inside the declared energy-density interval.
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
        lower, upper = (
            _finite_float("energy_density_domain lower", raw_lower),
            _finite_float("energy_density_domain upper", raw_upper),
        )
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
        endpoint_pressure = _evaluate_user_function(
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
        values, scalar = _domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        result = _evaluate_user_function(self._pressure, values, name="pressure")
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise EosInputError("analytical pressure returned invalid values")
        return _scalar_or_array(result, scalar)

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> Any:
        values, scalar = _domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        result = _evaluate_user_function(self._cs2, values, name="sound speed")
        if result.shape != values.shape or not np.all(np.isfinite(result)):
            raise EosInputError("analytical sound speed returned invalid values")
        return _scalar_or_array(result, scalar)

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> Any:
        values, scalar = _domain_values(
            pressure_mev_fm3,
            name="pressure_mev_fm3",
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
        )
        if self._inverse is not None:
            result = _evaluate_user_function(
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
        return _scalar_or_array(result, scalar)

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        epsilon = float(self.energy_density_from_pressure(pressure_mev_fm3))
        return epsilon, float(self.sound_speed_squared_from_energy_density(epsilon))

    def validate(self, *, points: int = 2049) -> EosValidationReport:
        report = _validate_eos_sampled(self, points=points)
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
                "derivative_relative_tolerance": ANALYTICAL_DERIVATIVE_RELATIVE_TOLERANCE,
                "derivative_absolute_tolerance": ANALYTICAL_DERIVATIVE_ABSOLUTE_TOLERANCE,
                "inverse_relative_tolerance": ANALYTICAL_INVERSE_RELATIVE_TOLERANCE,
                "inverse_absolute_tolerance": ANALYTICAL_INVERSE_ABSOLUTE_TOLERANCE,
            },
            "stellar_surface": {
                "kind": self.surface_boundary_kind,
                "pressure_MeV_fm3": self.pressure_min_mev_fm3,
                "tidal_capability": self.tidal_capability_status,
            },
        }


def _validate_eos_grid(
    eos: ColdBarotrope, energy_density_grid_mev_fm3: Any
) -> EosValidationReport:
    epsilon = np.asarray(energy_density_grid_mev_fm3, dtype=float)
    if epsilon.ndim != 1 or len(epsilon) < 17:
        raise ValueError("validation grid must contain at least 17 points")
    if not np.all(np.isfinite(epsilon)) or np.any(np.diff(epsilon) <= 0.0):
        raise ValueError("validation grid must be finite and strictly increasing")
    if epsilon[0] != float(eos.energy_density_min_mev_fm3) or epsilon[-1] != float(
        eos.energy_density_max_mev_fm3
    ):
        raise ValueError("validation grid must include the exact declared endpoints")
    pressure = np.asarray(eos.pressure_from_energy_density(epsilon), dtype=float)
    cs2 = np.asarray(eos.sound_speed_squared_from_energy_density(epsilon), dtype=float)
    issues: list[EosValidationIssue] = []
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(cs2)):
        issues.append(
            EosValidationIssue("nonfinite", "pressure or sound speed is nonfinite")
        )
    if np.any(pressure <= 0.0):
        issues.append(
            EosValidationIssue("nonpositive_pressure", "pressure must remain positive")
        )
    if np.any(np.diff(pressure) <= 0.0):
        issues.append(
            EosValidationIssue(
                "nonmonotone_pressure", "pressure must increase strictly"
            )
        )
    if np.any(cs2 <= 0.0):
        issues.append(
            EosValidationIssue("mechanical_instability", "dP/dE must remain positive")
        )
    if np.any(cs2 > 1.0):
        issues.append(EosValidationIssue("acausal", "dP/dE must not exceed one"))
    return EosValidationReport(
        model_name=str(eos.model_name),
        assessed_points=int(len(epsilon)),
        pressure_min_mev_fm3=float(np.min(pressure)),
        pressure_max_mev_fm3=float(np.max(pressure)),
        energy_density_min_mev_fm3=float(epsilon[0]),
        energy_density_max_mev_fm3=float(epsilon[-1]),
        cs2_min=float(np.min(cs2)),
        cs2_max=float(np.max(cs2)),
        issues=tuple(issues),
    )


def _validate_eos_sampled(
    eos: ColdBarotrope, *, points: int = 2049
) -> EosValidationReport:
    """Assess one continuous declared domain without repairing it."""
    if int(points) < 17:
        raise ValueError("validation points must be at least 17")
    epsilon = np.geomspace(
        float(eos.energy_density_min_mev_fm3),
        float(eos.energy_density_max_mev_fm3),
        int(points),
    )
    return _validate_eos_grid(eos, epsilon)


def validate_eos(eos: ColdBarotrope, *, points: int = 2049) -> EosValidationReport:
    """Run the adapter's strongest declared validation policy."""
    return eos.validate(points=points)


__all__ = [
    "AnalyticalEos",
    "CANONICAL_ENERGY_DENSITY_CONVENTION",
    "CANONICAL_UNITS",
    "ColdBarotrope",
    "EOS_INPUT_SCHEMA_VERSION",
    "EosDomainError",
    "EosInputError",
    "EosValidationIssue",
    "EosValidationReport",
    "validate_eos",
]
