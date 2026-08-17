"""Mechanical stability and causality checks for cold barotropes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neutron_star_eos.eos.core import (
    EOS_INPUT_SCHEMA_VERSION,
    ColdBarotrope,
    EosInputError,
)


@dataclass(frozen=True, slots=True)
class EosValidationIssue:
    """One stable validation code and its human-readable explanation."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class EosValidationReport:
    """Validation results over one declared EoS domain."""

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

    def require_pass(self) -> EosValidationReport:
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
            "sound_speed_squared": {
                "minimum": self.cs2_min,
                "maximum": self.cs2_max,
            },
            "issues": [
                {"code": item.code, "message": item.message} for item in self.issues
            ],
        }


def validate_eos_grid(
    eos: ColdBarotrope, energy_density_grid_mev_fm3: Any
) -> EosValidationReport:
    """Assess an explicit increasing grid including exact domain endpoints."""

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


def validate_eos_sampled(
    eos: ColdBarotrope, *, points: int = 2049
) -> EosValidationReport:
    """Assess one continuous domain on a logarithmic energy-density grid."""

    if int(points) < 17:
        raise ValueError("validation points must be at least 17")
    epsilon = np.geomspace(
        float(eos.energy_density_min_mev_fm3),
        float(eos.energy_density_max_mev_fm3),
        int(points),
    )
    return validate_eos_grid(eos, epsilon)


def validate_eos(eos: ColdBarotrope, *, points: int = 2049) -> EosValidationReport:
    """Run the adapter's strongest declared validation policy."""

    return eos.validate(points=points)


_validate_eos_grid = validate_eos_grid
_validate_eos_sampled = validate_eos_sampled
