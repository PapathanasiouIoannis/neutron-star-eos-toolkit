"""Continuous, background-only TOV calculations.

The public toolkit intentionally stops at the lower pressure supplied by the
EoS.  That boundary is not called a vacuum surface, and this module does not
calculate tidal observables, maximum masses, or discontinuous branches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, cast

import numpy as np
from scipy.integrate import solve_ivp

from neutron_star_eos.eos import (
    ColdBarotrope,
    EosDomainError,
    EosInputError,
    _eos_provenance_sha256,
)

STELLAR_CONSTANT_AUTHORITY = (
    "CompOSE Reference Manual v3.01 constants table; SI definitions"
)
STELLAR_CONSTANT_REFERENCE_URL = (
    "https://compose.obspm.fr/download/pdf/manual_v3.00.pdf"
)
SPEED_OF_LIGHT_M_S = 299_792_458.0
NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11
SOLAR_MASS_KG = 1.98841e30
MEV_J = 1.602176634e-13
FM3_M3 = 1.0e-45

# In dm/dr = A r^2 epsilon, r is km, epsilon is MeV fm^-3, and m is
# measured in solar masses.  The 1e9 factor converts r^2 dr from km^3 to m^3.
GRAVITY_CONVERSION = (
    4.0 * math.pi * 1.0e9 * (MEV_J / FM3_M3) / (SPEED_OF_LIGHT_M_S**2 * SOLAR_MASS_KG)
)
SOLAR_MASS_LENGTH_KM = (
    NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
    * SOLAR_MASS_KG
    / SPEED_OF_LIGHT_M_S**2
    / 1.0e3
)
STELLAR_VALIDATION_MODES = ("strict", "background_diagnostic")
BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES = frozenset({"acausal", "mechanical_instability"})


class StellarSolveError(RuntimeError):
    """A structured stellar-integration failure suitable for sequences."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class StellarConfig:
    """Numerical settings for one continuous background integration."""

    radius_start_km: float = 1.0e-4
    radius_max_km: float = 25.0
    center_expansion_limit_km: float = 1.0e-4
    ode_rtol: float = 1.0e-10
    ode_atol: float = 1.0e-12
    profile_points: int = 300


DEFAULT_STELLAR_CONFIG = StellarConfig()


def _config_to_dict(config: StellarConfig) -> dict[str, float | int]:
    return {
        "radius_start_km": config.radius_start_km,
        "radius_max_km": config.radius_max_km,
        "center_expansion_limit_km": config.center_expansion_limit_km,
        "ode_rtol": config.ode_rtol,
        "ode_atol": config.ode_atol,
        "profile_points": config.profile_points,
    }


def _constants_to_dict() -> dict[str, str | float]:
    return {
        "authority": STELLAR_CONSTANT_AUTHORITY,
        "authority_url": STELLAR_CONSTANT_REFERENCE_URL,
        "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
        "newtonian_gravitational_constant_m3_kg_s2": (
            NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
        ),
        "solar_mass_kg": SOLAR_MASS_KG,
        "MeV_J": MEV_J,
        "fm3_m3": FM3_M3,
        "gravity_conversion_Msun_per_km3_per_MeV_fm3": GRAVITY_CONVERSION,
        "solar_mass_length_km": SOLAR_MASS_LENGTH_KM,
    }


@dataclass(frozen=True, slots=True)
class StarResult:
    """One stellar background truncated at the EoS lower pressure."""

    central_pressure_mev_fm3: float
    central_energy_density_mev_fm3: float
    central_sound_speed_squared: float
    mass_msun: float
    radius_km: float
    boundary_pressure_mev_fm3: float
    boundary_energy_density_mev_fm3: float
    boundary_status: str
    integration_variable: str = "radius"
    radius_profile_km: tuple[float, ...] = ()
    mass_profile_msun: tuple[float, ...] = ()
    eos_validation_mode: str = "strict"
    eos_validation_status: str = "pass"
    eos_validation_issues: tuple[str, ...] = ()
    solver_config: StellarConfig = DEFAULT_STELLAR_CONFIG
    model_name: str = ""
    eos_provenance_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stellar-background-star-v1",
            "model": {
                "name": self.model_name,
                "eos_provenance_sha256": self.eos_provenance_sha256,
            },
            "central_pressure_MeV_fm3": self.central_pressure_mev_fm3,
            "central_energy_density_MeV_fm3": self.central_energy_density_mev_fm3,
            "central_sound_speed_squared": self.central_sound_speed_squared,
            "mass_Msun": self.mass_msun,
            "radius_km": self.radius_km,
            "integration_variable": self.integration_variable,
            "eos_validation": {
                "mode": self.eos_validation_mode,
                "status": self.eos_validation_status,
                "issues": list(self.eos_validation_issues),
            },
            "solver_config": _config_to_dict(self.solver_config),
            "physical_constants": _constants_to_dict(),
            "boundary": {
                "status": self.boundary_status,
                "pressure_MeV_fm3": self.boundary_pressure_mev_fm3,
                "energy_density_MeV_fm3": self.boundary_energy_density_mev_fm3,
                "is_vacuum_surface": False,
            },
            "profile": {
                "retained": bool(self.radius_profile_km),
                "radius_km": list(self.radius_profile_km),
                "mass_Msun": list(self.mass_profile_msun),
            },
        }


@dataclass(frozen=True, slots=True)
class SequenceAttempt:
    """One requested central pressure and its explicit outcome."""

    central_pressure_mev_fm3: float
    status: str
    star: StarResult | None
    reason: str | None
    reason_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "central_pressure_MeV_fm3": self.central_pressure_mev_fm3,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "star": None if self.star is None else self.star.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """Serial background sequence with failed attempts retained."""

    model_name: str
    attempts: tuple[SequenceAttempt, ...]
    status: str
    boundary_status: str
    eos_validation_mode: str = "strict"
    eos_validation_status: str = "pass"
    eos_validation_issues: tuple[str, ...] = ()
    solver_config: StellarConfig = DEFAULT_STELLAR_CONFIG
    eos_provenance_sha256: str = ""

    @property
    def stars(self) -> tuple[StarResult, ...]:
        return tuple(item.star for item in self.attempts if item.star is not None)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stellar-background-sequence-v1",
            "model_name": self.model_name,
            "eos_provenance_sha256": self.eos_provenance_sha256,
            "status": self.status,
            "boundary_status": self.boundary_status,
            "eos_validation": {
                "mode": self.eos_validation_mode,
                "status": self.eos_validation_status,
                "issues": list(self.eos_validation_issues),
            },
            "solver_config": _config_to_dict(self.solver_config),
            "physical_constants": _constants_to_dict(),
            "attempts": [item.to_dict() for item in self.attempts],
        }


def _finite(name: str, value: object) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise EosInputError(f"{name} must be finite")
    return result


def _checked_config(config: StellarConfig | None) -> StellarConfig:
    resolved = DEFAULT_STELLAR_CONFIG if config is None else config
    if not isinstance(resolved, StellarConfig):
        raise TypeError("config must be a StellarConfig")
    positive = {
        "radius_start_km": resolved.radius_start_km,
        "radius_max_km": resolved.radius_max_km,
        "center_expansion_limit_km": resolved.center_expansion_limit_km,
        "ode_rtol": resolved.ode_rtol,
        "ode_atol": resolved.ode_atol,
    }
    for name, value in positive.items():
        if _finite(name, value) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if resolved.radius_max_km <= resolved.radius_start_km:
        raise ValueError("radius_max_km must exceed radius_start_km")
    if isinstance(resolved.profile_points, bool) or not isinstance(
        resolved.profile_points, int
    ):
        raise TypeError("profile_points must be an integer")
    if resolved.profile_points < 2:
        raise ValueError("profile_points must be at least two")
    return resolved


def _validation_for_stellar_background(
    eos: ColdBarotrope,
    validation_mode: str,
) -> tuple[object, tuple[str, ...], str]:
    if validation_mode not in STELLAR_VALIDATION_MODES:
        raise ValueError(f"validation_mode must be one of {STELLAR_VALIDATION_MODES}")
    report = eos.validate()
    issue_codes = tuple(item.code for item in report.issues)
    if validation_mode == "strict":
        report.require_pass()
        return report, issue_codes, "pass"
    blockers = tuple(
        code for code in issue_codes if code not in BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES
    )
    if blockers:
        raise EosInputError(
            "background diagnostic cannot bypass EoS issue(s): " + ", ".join(blockers)
        )
    status = "pass" if not issue_codes else "diagnostic_with_issues"
    return report, issue_codes, status


def _annotate_validation(
    star: StarResult,
    *,
    validation_mode: str,
    validation_status: str,
    issue_codes: tuple[str, ...],
    model_name: str,
    eos_provenance_sha256: str,
) -> StarResult:
    return replace(
        star,
        eos_validation_mode=validation_mode,
        eos_validation_status=validation_status,
        eos_validation_issues=issue_codes,
        model_name=model_name,
        eos_provenance_sha256=eos_provenance_sha256,
    )


def _solve_validated_star_radius(
    eos: ColdBarotrope,
    central_pressure: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
    enforce_sound_speed_bounds: bool,
) -> StarResult:
    pressure_min = _finite("EoS lower pressure", eos.pressure_min_mev_fm3)
    pressure_max = _finite("EoS upper pressure", eos.pressure_max_mev_fm3)
    if not pressure_min < central_pressure <= pressure_max:
        raise EosDomainError(
            "central pressure must be above the EoS lower pressure and no greater "
            "than its declared upper pressure"
        )

    central_epsilon, central_cs2 = map(float, eos(central_pressure))
    if not math.isfinite(central_epsilon) or central_epsilon <= 0.0:
        raise EosInputError("central energy density must be finite and positive")
    if not math.isfinite(central_cs2):
        raise EosInputError("central sound speed squared must be finite")
    if enforce_sound_speed_bounds and not 0.0 < central_cs2 <= 1.0:
        raise EosInputError("central sound speed squared must satisfy 0 < cs2 <= 1")

    radius_start = config.radius_start_km
    initial_mass = radius_start**3 * central_epsilon * (GRAVITY_CONVERSION / 3.0)
    initial_state = np.asarray([initial_mass, central_pressure], dtype=float)

    def rhs(radius: float, state: np.ndarray) -> list[float]:
        mass, pressure = map(float, state)
        evaluation_pressure = max(pressure, pressure_min)
        epsilon, cs2 = map(float, eos(evaluation_pressure))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise EosInputError(
                "EoS returned invalid energy density during TOV integration"
            )
        if not math.isfinite(cs2):
            raise EosInputError(
                "EoS returned nonfinite sound speed during TOV integration"
            )
        if enforce_sound_speed_bounds and not 0.0 < cs2 <= 1.0:
            raise EosInputError(
                "EoS returned invalid sound speed during TOV integration"
            )
        pressure_safe = max(pressure, pressure_min)
        dm_dr = radius**2 * epsilon * GRAVITY_CONVERSION
        if radius <= config.center_expansion_limit_km:
            dpressure_dr = (
                -SOLAR_MASS_LENGTH_KM
                * GRAVITY_CONVERSION
                * (epsilon + pressure_safe)
                * (epsilon / 3.0 + pressure_safe)
                * radius
            )
        else:
            denominator = radius * (radius - 2.0 * mass * SOLAR_MASS_LENGTH_KM)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise EosInputError(
                    "TOV integration reached the Schwarzschild-radius boundary"
                )
            dpressure_dr = (
                -SOLAR_MASS_LENGTH_KM
                * (epsilon + pressure_safe)
                * (mass + radius**3 * pressure_safe * GRAVITY_CONVERSION)
                / denominator
            )
        if not math.isfinite(dm_dr) or not math.isfinite(dpressure_dr):
            raise EosInputError("TOV derivative became nonfinite")
        return [dm_dr, dpressure_dr]

    def boundary_event(_radius: float, state: np.ndarray) -> float:
        return float(state[1] - pressure_min)

    boundary_event_attributes = cast(Any, boundary_event)
    boundary_event_attributes.terminal = True
    boundary_event_attributes.direction = -1
    solution = solve_ivp(
        rhs,
        (radius_start, config.radius_max_km),
        initial_state,
        events=boundary_event,
        method="RK45",
        dense_output=retain_profile,
        rtol=config.ode_rtol,
        atol=config.ode_atol,
    )
    event_count = len(solution.t_events[0]) if solution.t_events else 0
    if solution.status != 1 or event_count != 1:
        if solution.status == 0 and event_count == 0:
            raise StellarSolveError(
                "radius_limit_reached",
                "the EoS lower-pressure boundary was not reached before "
                f"radius_max_km={config.radius_max_km:.12g}",
            )
        raise StellarSolveError(
            "boundary_event_failure",
            "the EoS lower-pressure boundary was not reached exactly once "
            f"(solver_status={solution.status}, event_count={event_count})",
        )
    event_state = np.asarray(solution.y_events[0][0], dtype=float)
    mass = float(event_state[0])
    radius = float(solution.t_events[0][0])
    if (
        not math.isfinite(mass)
        or not math.isfinite(radius)
        or mass <= 0.0
        or radius <= 0.0
    ):
        raise RuntimeError("TOV boundary mass or radius is invalid")

    radius_profile: tuple[float, ...] = ()
    mass_profile: tuple[float, ...] = ()
    if retain_profile:
        radii = np.linspace(radius_start, radius, int(config.profile_points))
        masses = np.asarray(solution.sol(radii)[0], dtype=float)
        if not np.all(np.isfinite(masses)) or np.any(np.diff(masses) < 0.0):
            raise RuntimeError("TOV mass profile is invalid")
        radius_profile = tuple(float(value) for value in radii)
        mass_profile = tuple(float(value) for value in masses)

    boundary_epsilon = float(eos(pressure_min)[0])
    return StarResult(
        central_pressure_mev_fm3=central_pressure,
        central_energy_density_mev_fm3=central_epsilon,
        central_sound_speed_squared=central_cs2,
        mass_msun=mass,
        radius_km=radius,
        boundary_pressure_mev_fm3=pressure_min,
        boundary_energy_density_mev_fm3=boundary_epsilon,
        boundary_status="truncated_at_eos_lower_pressure_not_vacuum",
        integration_variable="radius",
        radius_profile_km=radius_profile,
        mass_profile_msun=mass_profile,
        solver_config=config,
    )


def _solve_validated_star_log_pressure(
    eos: ColdBarotrope,
    central_pressure: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
    enforce_sound_speed_bounds: bool,
) -> StarResult:
    """Integrate a tabulated barotrope with ``log(P)`` as the coordinate.

    The independent variable reaches the exact positive source boundary, so
    the ODE tolerance is applied only to radius and mass rather than to a
    pressure state spanning many orders of magnitude.
    """

    pressure_min = _finite("EoS lower pressure", eos.pressure_min_mev_fm3)
    pressure_max = _finite("EoS upper pressure", eos.pressure_max_mev_fm3)
    if not pressure_min < central_pressure <= pressure_max:
        raise EosDomainError(
            "central pressure must be above the EoS lower pressure and no greater "
            "than its declared upper pressure"
        )
    central_epsilon, central_cs2 = map(float, eos(central_pressure))
    if not math.isfinite(central_epsilon) or central_epsilon <= 0.0:
        raise EosInputError("central energy density must be finite and positive")
    if not math.isfinite(central_cs2):
        raise EosInputError("central sound speed squared must be finite")
    if enforce_sound_speed_bounds and not 0.0 < central_cs2 <= 1.0:
        raise EosInputError("central sound speed squared must satisfy 0 < cs2 <= 1")

    radius_start = config.radius_start_km
    initial_mass = radius_start**3 * central_epsilon * (GRAVITY_CONVERSION / 3.0)
    initial_state = np.asarray([radius_start, initial_mass], dtype=float)

    def rhs(log_pressure: float, state: np.ndarray) -> list[float]:
        radius, mass = map(float, state)
        if not 0.0 < radius < config.radius_max_km:
            raise RuntimeError("tabulated TOV integration reached the radius limit")
        if log_pressure <= log_boundary:
            pressure = pressure_min
        elif log_pressure >= log_central:
            pressure = central_pressure
        else:
            pressure = math.exp(float(log_pressure))
        epsilon, cs2 = map(float, eos(pressure))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise EosInputError(
                "EoS returned invalid energy density during TOV integration"
            )
        if not math.isfinite(cs2):
            raise EosInputError(
                "EoS returned nonfinite sound speed during TOV integration"
            )
        if enforce_sound_speed_bounds and not 0.0 < cs2 <= 1.0:
            raise EosInputError(
                "EoS returned invalid sound speed during TOV integration"
            )
        dm_dr = radius**2 * epsilon * GRAVITY_CONVERSION
        if radius <= config.center_expansion_limit_km:
            dpressure_dr = (
                -SOLAR_MASS_LENGTH_KM
                * GRAVITY_CONVERSION
                * (epsilon + pressure)
                * (epsilon / 3.0 + pressure)
                * radius
            )
        else:
            denominator = radius * (radius - 2.0 * mass * SOLAR_MASS_LENGTH_KM)
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise EosInputError(
                    "TOV integration reached the Schwarzschild-radius boundary"
                )
            dpressure_dr = (
                -SOLAR_MASS_LENGTH_KM
                * (epsilon + pressure)
                * (mass + radius**3 * pressure * GRAVITY_CONVERSION)
                / denominator
            )
        if (
            not math.isfinite(dm_dr)
            or not math.isfinite(dpressure_dr)
            or dpressure_dr >= 0.0
        ):
            raise EosInputError("TOV derivative became invalid")
        dr_dlog_pressure = pressure / dpressure_dr
        dm_dlog_pressure = dm_dr * dr_dlog_pressure
        return [dr_dlog_pressure, dm_dlog_pressure]

    def radius_limit(_log_pressure: float, state: np.ndarray) -> float:
        return float(config.radius_max_km - state[0])

    radius_limit_attributes = cast(Any, radius_limit)
    radius_limit_attributes.terminal = True
    radius_limit_attributes.direction = -1
    log_central = math.log(central_pressure)
    log_boundary = math.log(pressure_min)
    solution = solve_ivp(
        rhs,
        (log_central, log_boundary),
        initial_state,
        events=radius_limit,
        method="RK45",
        dense_output=False,
        rtol=config.ode_rtol,
        atol=np.asarray([config.ode_atol, config.ode_atol], dtype=float),
    )
    if solution.status != 0 or not math.isclose(
        float(solution.t[-1]), log_boundary, rel_tol=0.0, abs_tol=1.0e-12
    ):
        if solution.status == 1:
            raise StellarSolveError(
                "radius_limit_reached",
                "the EoS lower-pressure boundary was not reached before "
                f"radius_max_km={config.radius_max_km:.12g}",
            )
        raise StellarSolveError(
            "boundary_integration_failure",
            "the tabulated integration did not reach the EoS lower-pressure boundary "
            f"(solver_status={solution.status})",
        )
    radius = float(solution.y[0, -1])
    mass = float(solution.y[1, -1])
    if (
        not math.isfinite(mass)
        or not math.isfinite(radius)
        or mass <= 0.0
        or radius <= 0.0
    ):
        raise RuntimeError("TOV boundary mass or radius is invalid")

    radius_profile: tuple[float, ...] = ()
    mass_profile: tuple[float, ...] = ()
    if retain_profile:
        profile_log_pressures = np.linspace(
            log_central, log_boundary, int(config.profile_points)
        )
        increasing_log_pressure = np.asarray(solution.t[::-1], dtype=float)
        radii = np.interp(
            profile_log_pressures,
            increasing_log_pressure,
            np.asarray(solution.y[0, ::-1], dtype=float),
        )
        masses = np.interp(
            profile_log_pressures,
            increasing_log_pressure,
            np.asarray(solution.y[1, ::-1], dtype=float),
        )
        if (
            not np.all(np.isfinite(radii))
            or not np.all(np.isfinite(masses))
            or np.any(np.diff(radii) <= 0.0)
            or np.any(np.diff(masses) < 0.0)
        ):
            raise RuntimeError("TOV profile is invalid")
        radius_profile = tuple(float(value) for value in radii)
        mass_profile = tuple(float(value) for value in masses)

    boundary_epsilon = float(eos(pressure_min)[0])
    return StarResult(
        central_pressure_mev_fm3=central_pressure,
        central_energy_density_mev_fm3=central_epsilon,
        central_sound_speed_squared=central_cs2,
        mass_msun=mass,
        radius_km=radius,
        boundary_pressure_mev_fm3=pressure_min,
        boundary_energy_density_mev_fm3=boundary_epsilon,
        boundary_status="truncated_at_eos_lower_pressure_not_vacuum",
        integration_variable="log_pressure",
        radius_profile_km=radius_profile,
        mass_profile_msun=mass_profile,
        solver_config=config,
    )


def _solve_validated_star(
    eos: ColdBarotrope,
    central_pressure: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
    validation_mode: str,
) -> StarResult:
    enforce_sound_speed_bounds = validation_mode == "strict"
    route = getattr(eos, "preferred_stellar_integration_variable", "radius")
    if route == "log_pressure":
        return _solve_validated_star_log_pressure(
            eos,
            central_pressure,
            config=config,
            retain_profile=retain_profile,
            enforce_sound_speed_bounds=enforce_sound_speed_bounds,
        )
    if route != "radius":
        raise EosInputError(f"unsupported stellar integration variable: {route!r}")
    return _solve_validated_star_radius(
        eos,
        central_pressure,
        config=config,
        retain_profile=retain_profile,
        enforce_sound_speed_bounds=enforce_sound_speed_bounds,
    )


def solve_star(
    eos: ColdBarotrope,
    central_pressure_mev_fm3: float,
    *,
    retain_profile: bool = False,
    config: StellarConfig | None = None,
    validation_mode: str = "strict",
) -> StarResult:
    """Validate an EoS and solve one continuous truncated background.

    ``background_diagnostic`` permits only causality or sound-speed-sign
    findings that do not make the positive, invertible ``epsilon(P)`` relation
    unusable by the background TOV equations.  It never upgrades those findings
    to a pass and records them on the returned result.
    """
    _report, issue_codes, validation_status = _validation_for_stellar_background(
        eos, validation_mode
    )
    resolved = _checked_config(config)
    model_name = str(eos.model_name)
    provenance_sha256 = _eos_provenance_sha256(eos)
    pressure = _finite("central_pressure_mev_fm3", central_pressure_mev_fm3)
    return _annotate_validation(
        _solve_validated_star(
            eos,
            pressure,
            config=resolved,
            retain_profile=bool(retain_profile),
            validation_mode=validation_mode,
        ),
        validation_mode=validation_mode,
        validation_status=validation_status,
        issue_codes=issue_codes,
        model_name=model_name,
        eos_provenance_sha256=provenance_sha256,
    )


def solve_sequence(
    eos: ColdBarotrope,
    central_pressures_mev_fm3: Iterable[float] | None = None,
    *,
    points: int = 50,
    config: StellarConfig | None = None,
    validation_mode: str = "strict",
) -> SequenceResult:
    """Solve a serial pressure sequence without inferring a stable branch or Mmax."""
    _report, issue_codes, validation_status = _validation_for_stellar_background(
        eos, validation_mode
    )
    resolved = _checked_config(config)
    model_name = str(eos.model_name)
    provenance_sha256 = _eos_provenance_sha256(eos)
    pressure_min = float(eos.pressure_min_mev_fm3)
    pressure_max = float(eos.pressure_max_mev_fm3)
    if central_pressures_mev_fm3 is None:
        if isinstance(points, bool) or not isinstance(points, int):
            raise TypeError("points must be an integer")
        if points < 9:
            raise ValueError("points must be at least nine")
        lower = float(np.nextafter(pressure_min, math.inf))
        if not lower < pressure_max:
            raise EosDomainError(
                "the EoS pressure domain is too narrow to form a central-pressure sequence"
            )
        pressures = np.geomspace(lower, pressure_max, points)
    else:
        pressures = np.asarray(tuple(central_pressures_mev_fm3), dtype=float)
        if pressures.ndim != 1 or len(pressures) == 0:
            raise ValueError(
                "central pressures must be a non-empty one-dimensional sequence"
            )
        if not np.all(np.isfinite(pressures)) or np.any(np.diff(pressures) <= 0.0):
            raise ValueError("central pressures must be finite and strictly increasing")
        if pressures[0] <= pressure_min or pressures[-1] > pressure_max:
            raise EosDomainError(
                "central-pressure sequence leaves the declared EoS domain"
            )

    attempts: list[SequenceAttempt] = []
    for pressure in pressures:
        candidate = float(pressure)
        try:
            star = _solve_validated_star(
                eos,
                candidate,
                config=resolved,
                retain_profile=False,
                validation_mode=validation_mode,
            )
        except (EosInputError, RuntimeError, ArithmeticError) as exc:
            attempts.append(
                SequenceAttempt(
                    candidate,
                    "unavailable",
                    None,
                    str(exc),
                    getattr(exc, "reason_code", "stellar_solve_failed"),
                )
            )
        else:
            attempts.append(
                SequenceAttempt(
                    candidate,
                    "solved",
                    _annotate_validation(
                        star,
                        validation_mode=validation_mode,
                        validation_status=validation_status,
                        issue_codes=issue_codes,
                        model_name=model_name,
                        eos_provenance_sha256=provenance_sha256,
                    ),
                    None,
                )
            )
    status = (
        "complete" if all(item.star is not None for item in attempts) else "partial"
    )
    return SequenceResult(
        model_name=model_name,
        attempts=tuple(attempts),
        status=status,
        boundary_status="truncated_at_eos_lower_pressure_not_vacuum",
        eos_validation_mode=validation_mode,
        eos_validation_status=validation_status,
        eos_validation_issues=issue_codes,
        solver_config=resolved,
        eos_provenance_sha256=provenance_sha256,
    )


__all__ = [
    "DEFAULT_STELLAR_CONFIG",
    "GRAVITY_CONVERSION",
    "MEV_J",
    "FM3_M3",
    "NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2",
    "SOLAR_MASS_KG",
    "SOLAR_MASS_LENGTH_KM",
    "SPEED_OF_LIGHT_M_S",
    "STELLAR_CONSTANT_AUTHORITY",
    "STELLAR_CONSTANT_REFERENCE_URL",
    "BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES",
    "SequenceAttempt",
    "SequenceResult",
    "StarResult",
    "StellarSolveError",
    "StellarConfig",
    "STELLAR_VALIDATION_MODES",
    "solve_sequence",
    "solve_star",
]
