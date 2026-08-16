"""Continuous, background-only TOV calculations.

The public toolkit intentionally stops at the lower pressure supplied by the
EoS.  That boundary is not called a vacuum surface, and this module does not
calculate tidal observables, maximum masses, or discontinuous branches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp

from neutron_star_eos.eos import ColdBarotrope, EosDomainError, EosInputError


GRAVITY_CONVERSION = 1.124e-5
SOLAR_MASS_LENGTH_KM = 1.4766


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
    radius_profile_km: tuple[float, ...] = ()
    mass_profile_msun: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "central_pressure_MeV_fm3": self.central_pressure_mev_fm3,
            "central_energy_density_MeV_fm3": self.central_energy_density_mev_fm3,
            "central_sound_speed_squared": self.central_sound_speed_squared,
            "mass_Msun": self.mass_msun,
            "radius_km": self.radius_km,
            "boundary": {
                "status": self.boundary_status,
                "pressure_MeV_fm3": self.boundary_pressure_mev_fm3,
                "energy_density_MeV_fm3": self.boundary_energy_density_mev_fm3,
                "is_vacuum_surface": False,
            },
        }


@dataclass(frozen=True, slots=True)
class SequenceAttempt:
    """One requested central pressure and its explicit outcome."""

    central_pressure_mev_fm3: float
    status: str
    star: StarResult | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """Serial background sequence with failed attempts retained."""

    model_name: str
    attempts: tuple[SequenceAttempt, ...]
    status: str
    boundary_status: str

    @property
    def stars(self) -> tuple[StarResult, ...]:
        return tuple(item.star for item in self.attempts if item.star is not None)


def _finite(name: str, value: object) -> float:
    try:
        result = float(value)
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
    if int(resolved.profile_points) < 2:
        raise ValueError("profile_points must be at least two")
    return resolved


def _solve_validated_star(
    eos: ColdBarotrope,
    central_pressure: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
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
    if not math.isfinite(central_cs2) or not 0.0 < central_cs2 <= 1.0:
        raise EosInputError("central sound speed squared must satisfy 0 < cs2 <= 1")

    radius_start = config.radius_start_km
    initial_mass = radius_start**3 * central_epsilon * (GRAVITY_CONVERSION / 3.0)
    initial_state = np.asarray([initial_mass, central_pressure], dtype=float)

    def rhs(radius: float, state: np.ndarray) -> list[float]:
        mass, pressure = map(float, state)
        evaluation_pressure = max(pressure, pressure_min)
        epsilon, cs2 = map(float, eos(evaluation_pressure))
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise EosInputError("EoS returned invalid energy density during TOV integration")
        if not math.isfinite(cs2) or not 0.0 < cs2 <= 1.0:
            raise EosInputError("EoS returned invalid sound speed during TOV integration")
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
            denominator = radius * (
                radius - 2.0 * mass * SOLAR_MASS_LENGTH_KM
            )
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

    boundary_event.terminal = True
    boundary_event.direction = -1
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
        raise RuntimeError(
            "the EoS lower-pressure boundary was not reached exactly once "
            f"(solver_status={solution.status}, event_count={event_count})"
        )
    event_state = np.asarray(solution.y_events[0][0], dtype=float)
    mass = float(event_state[0])
    radius = float(solution.t_events[0][0])
    if not math.isfinite(mass) or not math.isfinite(radius) or mass <= 0.0 or radius <= 0.0:
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
        radius_profile_km=radius_profile,
        mass_profile_msun=mass_profile,
    )


def solve_star(
    eos: ColdBarotrope,
    central_pressure_mev_fm3: float,
    *,
    retain_profile: bool = False,
    config: StellarConfig | None = None,
) -> StarResult:
    """Validate an EoS and solve one continuous truncated background."""
    eos.validate().require_pass()
    pressure = _finite("central_pressure_mev_fm3", central_pressure_mev_fm3)
    return _solve_validated_star(
        eos,
        pressure,
        config=_checked_config(config),
        retain_profile=bool(retain_profile),
    )


def solve_sequence(
    eos: ColdBarotrope,
    central_pressures_mev_fm3: Iterable[float] | None = None,
    *,
    points: int = 50,
    config: StellarConfig | None = None,
) -> SequenceResult:
    """Solve a serial pressure sequence without inferring a stable branch or Mmax."""
    eos.validate().require_pass()
    resolved = _checked_config(config)
    pressure_min = float(eos.pressure_min_mev_fm3)
    pressure_max = float(eos.pressure_max_mev_fm3)
    if central_pressures_mev_fm3 is None:
        if int(points) < 9:
            raise ValueError("points must be at least nine")
        lower = float(np.nextafter(pressure_min, math.inf))
        if not lower < pressure_max:
            raise EosDomainError(
                "the EoS pressure domain is too narrow to form a central-pressure sequence"
            )
        pressures = np.geomspace(lower, pressure_max, int(points))
    else:
        pressures = np.asarray(tuple(central_pressures_mev_fm3), dtype=float)
        if pressures.ndim != 1 or len(pressures) == 0:
            raise ValueError("central pressures must be a non-empty one-dimensional sequence")
        if not np.all(np.isfinite(pressures)) or np.any(np.diff(pressures) <= 0.0):
            raise ValueError("central pressures must be finite and strictly increasing")
        if pressures[0] <= pressure_min or pressures[-1] > pressure_max:
            raise EosDomainError("central-pressure sequence leaves the declared EoS domain")

    attempts: list[SequenceAttempt] = []
    for pressure in pressures:
        candidate = float(pressure)
        try:
            star = _solve_validated_star(
                eos,
                candidate,
                config=resolved,
                retain_profile=False,
            )
        except (EosInputError, RuntimeError, ArithmeticError) as exc:
            attempts.append(SequenceAttempt(candidate, "unavailable", None, str(exc)))
        else:
            attempts.append(SequenceAttempt(candidate, "solved", star, None))
    status = "complete" if all(item.star is not None for item in attempts) else "partial"
    return SequenceResult(
        model_name=str(eos.model_name),
        attempts=tuple(attempts),
        status=status,
        boundary_status="truncated_at_eos_lower_pressure_not_vacuum",
    )


__all__ = [
    "DEFAULT_STELLAR_CONFIG",
    "SequenceAttempt",
    "SequenceResult",
    "StarResult",
    "StellarConfig",
    "solve_sequence",
    "solve_star",
]
