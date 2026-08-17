"""Data returned by one-star and mass-radius-sequence calculations."""

from __future__ import annotations

from dataclasses import dataclass

from neutron_star_eos.stellar.configuration import (
    DEFAULT_STELLAR_CONFIG,
    StellarConfig,
    config_to_dict,
)
from neutron_star_eos.stellar.constants import constants_to_dict


class StellarSolveError(RuntimeError):
    """A stellar-integration failure with a stable machine-readable code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class StarResult:
    """One TOV background truncated at the EoS lower pressure.

    Mass is in solar masses.  Radius is in km.  The boundary is the lowest
    positive pressure supplied by the EoS and is not claimed to be a vacuum
    stellar surface.
    """

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
            "solver_config": config_to_dict(self.solver_config),
            "physical_constants": constants_to_dict(),
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
    """A central-pressure sequence with every failed attempt retained."""

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
            "solver_config": config_to_dict(self.solver_config),
            "physical_constants": constants_to_dict(),
            "attempts": [item.to_dict() for item in self.attempts],
        }
