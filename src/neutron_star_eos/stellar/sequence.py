"""Calculate a serial mass-radius sequence over central pressure."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from neutron_star_eos.eos import (
    ColdBarotrope,
    EosDomainError,
    EosInputError,
    _eos_provenance_sha256,
)
from neutron_star_eos.stellar.configuration import StellarConfig, checked_config
from neutron_star_eos.stellar.results import SequenceAttempt, SequenceResult
from neutron_star_eos.stellar.tov import solve_validated_star
from neutron_star_eos.stellar.validation import (
    annotate_validation,
    validation_for_stellar_background,
)


def central_pressure_grid(
    eos: ColdBarotrope,
    central_pressures_mev_fm3: Iterable[float] | None,
    *,
    points: int,
) -> np.ndarray:
    """Return a checked, increasing pressure grid in MeV fm^-3."""

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
                "the EoS pressure domain is too narrow to form a "
                "central-pressure sequence"
            )
        return np.geomspace(lower, pressure_max, points)

    pressures = np.asarray(tuple(central_pressures_mev_fm3), dtype=float)
    if pressures.ndim != 1 or len(pressures) == 0:
        raise ValueError(
            "central pressures must be a non-empty one-dimensional sequence"
        )
    if not np.all(np.isfinite(pressures)) or np.any(np.diff(pressures) <= 0.0):
        raise ValueError("central pressures must be finite and strictly increasing")
    if pressures[0] <= pressure_min or pressures[-1] > pressure_max:
        raise EosDomainError("central-pressure sequence leaves the declared EoS domain")
    return pressures


def solve_sequence(
    eos: ColdBarotrope,
    central_pressures_mev_fm3: Iterable[float] | None = None,
    *,
    points: int = 50,
    config: StellarConfig | None = None,
    validation_mode: str = "strict",
) -> SequenceResult:
    """Calculate a pressure sequence without inferring stability or Mmax."""

    _report, issue_codes, validation_status = validation_for_stellar_background(
        eos, validation_mode
    )
    resolved_config = checked_config(config)
    model_name = str(eos.model_name)
    provenance_sha256 = _eos_provenance_sha256(eos)
    pressures = central_pressure_grid(eos, central_pressures_mev_fm3, points=points)

    attempts: list[SequenceAttempt] = []
    for pressure in pressures:
        central_pressure = float(pressure)
        try:
            star = solve_validated_star(
                eos,
                central_pressure,
                config=resolved_config,
                retain_profile=False,
                validation_mode=validation_mode,
            )
        except (EosInputError, RuntimeError, ArithmeticError) as exc:
            attempts.append(
                SequenceAttempt(
                    central_pressure,
                    "unavailable",
                    None,
                    str(exc),
                    getattr(exc, "reason_code", "stellar_solve_failed"),
                )
            )
        else:
            attempts.append(
                SequenceAttempt(
                    central_pressure,
                    "solved",
                    annotate_validation(
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
        solver_config=resolved_config,
        eos_provenance_sha256=provenance_sha256,
    )
