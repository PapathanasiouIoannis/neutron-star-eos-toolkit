"""Validation of a continuous CompOSE barotrope interpolant.

The source table diagnostics live elsewhere.  This module checks the actual
continuous functions used by the stellar solver: pressure and total energy
density as functions of native baryon density, plus ``c_s^2 = dP/dE``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PchipInterpolator

from neutron_star_eos.eos import EosValidationIssue, EosValidationReport


def validation_log_density_grid(
    *,
    log_baryon_density: np.ndarray,
    log_pressure_interpolant: PchipInterpolator,
    log_energy_density_interpolant: PchipInterpolator,
    points: int,
) -> np.ndarray:
    """Return density coordinates that include possible derivative extrema.

    Regular samples are supplemented with roots of the piecewise-polynomial
    derivatives.  This makes the causality and monotonicity checks sensitive
    to extrema between the original CompOSE nodes.
    """

    intervals = len(log_baryon_density) - 1
    points_per_interval = max(9, int(np.ceil((int(points) - 1) / intervals)))
    candidates: list[float] = []
    pressure_coefficients = log_pressure_interpolant.c
    epsilon_coefficients = log_energy_density_interpolant.c
    for index in range(intervals):
        left = float(log_baryon_density[index])
        right = float(log_baryon_density[index + 1])
        width = right - left
        candidates.extend(
            float(value)
            for value in np.linspace(left, right, points_per_interval, endpoint=False)
        )
        fp_cubic, fp_quadratic, fp_linear, fp_constant = pressure_coefficients[:, index]
        ep_cubic, ep_quadratic, ep_linear, ep_constant = epsilon_coefficients[:, index]
        log_pressure = Polynomial((fp_constant, fp_linear, fp_quadratic, fp_cubic))
        log_epsilon = Polynomial((ep_constant, ep_linear, ep_quadratic, ep_cubic))
        pressure_derivative = log_pressure.deriv()
        epsilon_derivative = log_epsilon.deriv()
        cs2_stationarity = (
            (pressure_derivative - epsilon_derivative)
            * pressure_derivative
            * epsilon_derivative
            + pressure_derivative.deriv() * epsilon_derivative
            - epsilon_derivative.deriv() * pressure_derivative
        )
        endpoint_tolerance = (
            64.0 * np.finfo(float).eps * max(1.0, abs(left), abs(right), width)
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
    candidates.append(float(log_baryon_density[-1]))
    return np.unique(np.asarray(candidates, dtype=float))


def validate_compose_interpolant(
    *,
    model_name: str,
    log_baryon_density: np.ndarray,
    log_pressure_interpolant: PchipInterpolator,
    log_energy_density_interpolant: PchipInterpolator,
    sound_speed_squared: Callable[[np.ndarray], np.ndarray],
    points: int,
) -> EosValidationReport:
    """Assess positivity, invertibility, stability, and causality."""

    if int(points) < 17:
        raise ValueError("validation points must be at least 17")
    log_density = validation_log_density_grid(
        log_baryon_density=log_baryon_density,
        log_pressure_interpolant=log_pressure_interpolant,
        log_energy_density_interpolant=log_energy_density_interpolant,
        points=int(points),
    )
    pressure = np.exp(log_pressure_interpolant(log_density))
    epsilon = np.exp(log_energy_density_interpolant(log_density))
    cs2 = sound_speed_squared(log_density)
    issues: list[EosValidationIssue] = []
    if (
        np.any(~np.isfinite(pressure))
        or np.any(~np.isfinite(epsilon))
        or np.any(~np.isfinite(cs2))
    ):
        issues.append(
            EosValidationIssue(
                "nonfinite",
                "interpolated pressure, energy density, or cs2 is nonfinite",
            )
        )
    if np.any(pressure <= 0.0) or np.any(epsilon <= 0.0):
        issues.append(
            EosValidationIssue(
                "nonpositive_thermodynamics",
                "pressure and total energy density must remain positive",
            )
        )
    if np.any(np.diff(pressure) <= 0.0) or np.any(np.diff(epsilon) <= 0.0):
        issues.append(
            EosValidationIssue(
                "nonmonotone_native_interpolation",
                "native-density interpolation must remain strictly invertible",
            )
        )
    if np.any(cs2 <= 0.0):
        issues.append(
            EosValidationIssue("mechanical_instability", "dP/dE must remain positive")
        )
    if np.any(cs2 > 1.0):
        issues.append(EosValidationIssue("acausal", "dP/dE must not exceed one"))
    return EosValidationReport(
        model_name=model_name,
        assessed_points=len(log_density),
        pressure_min_mev_fm3=float(np.min(pressure)),
        pressure_max_mev_fm3=float(np.max(pressure)),
        energy_density_min_mev_fm3=float(np.min(epsilon)),
        energy_density_max_mev_fm3=float(np.max(epsilon)),
        cs2_min=float(np.min(cs2)),
        cs2_max=float(np.max(cs2)),
        issues=tuple(issues),
    )


__all__ = ["validate_compose_interpolant", "validation_log_density_grid"]
