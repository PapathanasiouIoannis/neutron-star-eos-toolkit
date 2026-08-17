"""Numerical configuration and input checks for stellar integrations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from neutron_star_eos.eos import EosInputError


@dataclass(frozen=True, slots=True)
class StellarConfig:
    """Numerical settings for one continuous stellar background.

    Radii are measured in km.  ``ode_rtol`` and ``ode_atol`` control SciPy's
    RK45 integration.  These are numerical settings, not EoS parameters.
    """

    radius_start_km: float = 1.0e-4
    radius_max_km: float = 25.0
    center_expansion_limit_km: float = 1.0e-4
    ode_rtol: float = 1.0e-10
    ode_atol: float = 1.0e-12
    profile_points: int = 300


DEFAULT_STELLAR_CONFIG = StellarConfig()


def config_to_dict(config: StellarConfig) -> dict[str, float | int]:
    """Serialize the numerical settings without changing their values."""

    return {
        "radius_start_km": config.radius_start_km,
        "radius_max_km": config.radius_max_km,
        "center_expansion_limit_km": config.center_expansion_limit_km,
        "ode_rtol": config.ode_rtol,
        "ode_atol": config.ode_atol,
        "profile_points": config.profile_points,
    }


def finite_float(name: str, value: object) -> float:
    """Return a finite float or raise a user-facing EoS input error."""

    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise EosInputError(f"{name} must be finite")
    return result


def checked_config(config: StellarConfig | None) -> StellarConfig:
    """Validate solver settings before an integration begins."""

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
        if finite_float(name, value) <= 0.0:
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


# Historical private names retained while modules are reorganized.
_config_to_dict = config_to_dict
_finite = finite_float
_checked_config = checked_config
