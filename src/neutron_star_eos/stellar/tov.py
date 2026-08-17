"""Direct numerical integration of the TOV equations.

This module contains the actual stellar-background equations.  It supports
radius as the independent variable and, for tabulated EoSs spanning many
pressure decades, log(P) as the independent variable.  Both routes terminate
at the lowest positive pressure supplied by the EoS.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from scipy.integrate import solve_ivp

from neutron_star_eos.eos import ColdBarotrope, EosDomainError, EosInputError
from neutron_star_eos.stellar.configuration import StellarConfig, finite_float
from neutron_star_eos.stellar.constants import (
    GRAVITY_CONVERSION,
    SOLAR_MASS_LENGTH_KM,
)
from neutron_star_eos.stellar.results import StarResult, StellarSolveError


def solve_validated_star_radius(
    eos: ColdBarotrope,
    central_pressure_mev_fm3: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
    enforce_sound_speed_bounds: bool,
) -> StarResult:
    """Integrate mass and pressure outward using radius in km."""

    pressure_min = finite_float("EoS lower pressure", eos.pressure_min_mev_fm3)
    pressure_max = finite_float("EoS upper pressure", eos.pressure_max_mev_fm3)
    central_pressure = central_pressure_mev_fm3
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

    def tov_derivatives(radius: float, state: np.ndarray) -> list[float]:
        """Return dm/dr and dP/dr in Msun/km and MeV fm^-3/km."""

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

        # The regular central expansion avoids the formal 0/0 TOV expression.
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

    def reaches_source_boundary(_radius: float, state: np.ndarray) -> float:
        return float(state[1] - pressure_min)

    boundary_event = cast(Any, reaches_source_boundary)
    boundary_event.terminal = True
    boundary_event.direction = -1
    solution = solve_ivp(
        tov_derivatives,
        (radius_start, config.radius_max_km),
        initial_state,
        events=reaches_source_boundary,
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


def solve_validated_star_log_pressure(
    eos: ColdBarotrope,
    central_pressure_mev_fm3: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
    enforce_sound_speed_bounds: bool,
) -> StarResult:
    """Integrate radius and mass inward in log(P) coordinate.

    Log pressure reaches the exact positive source boundary.  ODE tolerances
    therefore act on radius and mass rather than on a pressure state spanning
    many orders of magnitude.
    """

    pressure_min = finite_float("EoS lower pressure", eos.pressure_min_mev_fm3)
    pressure_max = finite_float("EoS upper pressure", eos.pressure_max_mev_fm3)
    central_pressure = central_pressure_mev_fm3
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
    log_central = math.log(central_pressure)
    log_boundary = math.log(pressure_min)

    def tov_derivatives(log_pressure: float, state: np.ndarray) -> list[float]:
        """Return dr/dlog(P) and dm/dlog(P)."""

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

    radius_event = cast(Any, radius_limit)
    radius_event.terminal = True
    radius_event.direction = -1
    solution = solve_ivp(
        tov_derivatives,
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


def solve_validated_star(
    eos: ColdBarotrope,
    central_pressure_mev_fm3: float,
    *,
    config: StellarConfig,
    retain_profile: bool,
    validation_mode: str,
) -> StarResult:
    """Choose the declared integration coordinate for a validated EoS."""

    enforce_sound_speed_bounds = validation_mode == "strict"
    route = getattr(eos, "preferred_stellar_integration_variable", "radius")
    if route == "log_pressure":
        return solve_validated_star_log_pressure(
            eos,
            central_pressure_mev_fm3,
            config=config,
            retain_profile=retain_profile,
            enforce_sound_speed_bounds=enforce_sound_speed_bounds,
        )
    if route != "radius":
        raise EosInputError(f"unsupported stellar integration variable: {route!r}")
    return solve_validated_star_radius(
        eos,
        central_pressure_mev_fm3,
        config=config,
        retain_profile=retain_profile,
        enforce_sound_speed_bounds=enforce_sound_speed_bounds,
    )


_solve_validated_star_radius = solve_validated_star_radius
_solve_validated_star_log_pressure = solve_validated_star_log_pressure
_solve_validated_star = solve_validated_star
