"""Small immutable records used by CompOSE readers and cold slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COMPOSE_DATASET_SCHEMA_VERSION = "compose_dataset_v2"
COMPOSE_FORMAT_AUTHORITY = (
    "CompOSE Reference Manual v3.01, sections 4.2.1-4.2.9 and appendix A"
)
COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE = 1.0e-7
COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE = 1.0e-7
REQUIRED_FILES = ("eos.t", "eos.nb", "eos.yq", "eos.thermo")
OPTIONAL_FILES = ("eos.compo", "eos.micro", "eos.init", "eos.mr")


@dataclass(frozen=True, slots=True)
class ComposeAxis:
    """One indexed CompOSE coordinate axis."""

    minimum_index: int
    maximum_index: int
    values: tuple[float, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.minimum_index, self.maximum_index + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_index": self.minimum_index,
            "maximum_index": self.maximum_index,
            "points": len(self.values),
            "minimum": min(self.values),
            "maximum": max(self.values),
        }


@dataclass(frozen=True, slots=True)
class ComposeThermodynamicRow:
    """All thermodynamic fields from one ``eos.thermo`` record."""

    temperature_index: int
    baryon_density_index: int
    charge_fraction_index: int
    q_values: tuple[float, float, float, float, float, float, float]
    additional_values: tuple[float, ...]

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            self.temperature_index,
            self.baryon_density_index,
            self.charge_fraction_index,
        )


@dataclass(frozen=True, slots=True)
class ComposeCompositionRow:
    """One preserved ``eos.compo`` row with uninterpreted payload tokens."""

    temperature_index: int
    baryon_density_index: int
    charge_fraction_index: int
    phase_code: int
    raw_payload_tokens: tuple[str, ...]

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            self.temperature_index,
            self.baryon_density_index,
            self.charge_fraction_index,
        )


@dataclass(frozen=True, slots=True)
class ComposeDiagnostic:
    """One source or reduction diagnostic with explicit severity."""

    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass(frozen=True, slots=True)
class ComposeOrderingIssue:
    """One adjacent source-row ordering failure, preserved without repair."""

    left_position: int
    right_position: int
    left_baryon_density_fm3: float
    right_baryon_density_fm3: float
    left_value: float
    right_value: float

    @property
    def relative_change(self) -> float:
        return (self.right_value - self.left_value) / abs(self.left_value)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "left_position": self.left_position,
            "right_position": self.right_position,
            "left_baryon_density_fm3": self.left_baryon_density_fm3,
            "right_baryon_density_fm3": self.right_baryon_density_fm3,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "relative_change": self.relative_change,
        }


@dataclass(frozen=True, slots=True)
class ComposeSliceReport:
    """Diagnostics for one declared cold beta-equilibrium path."""

    rows: int
    euler_maximum_normalized_residual: float
    q5_maximum_absolute_residual: float
    q6_minus_q7_maximum_absolute_residual: float
    pressure_ordering_issues: tuple[ComposeOrderingIssue, ...]
    energy_density_ordering_issues: tuple[ComposeOrderingIssue, ...]
    phase_code_changes: int
    missing_phase_codes: int
    diagnostics: tuple[ComposeDiagnostic, ...]

    @property
    def continuous_barotrope_available(self) -> bool:
        return (
            not self.pressure_ordering_issues
            and not self.energy_density_ordering_issues
            and not any(
                item.severity == "barotrope_blocker" for item in self.diagnostics
            )
        )

    @property
    def status(self) -> str:
        if not self.continuous_barotrope_available:
            return "parsed_but_continuous_barotrope_unavailable"
        if any(item.severity == "warning" for item in self.diagnostics):
            return "continuous_barotrope_available_with_source_diagnostics"
        return "continuous_barotrope_available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPOSE_DATASET_SCHEMA_VERSION,
            "status": self.status,
            "rows": self.rows,
            "continuous_barotrope_available": self.continuous_barotrope_available,
            "cold_euler_closure": {
                "maximum_normalized_residual": self.euler_maximum_normalized_residual,
                "diagnostic_tolerance": COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            },
            "beta_equilibrium_Q5": {
                "maximum_absolute_residual": self.q5_maximum_absolute_residual,
                "diagnostic_tolerance": COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
            },
            "zero_temperature_Q6_minus_Q7": {
                "maximum_absolute_residual": self.q6_minus_q7_maximum_absolute_residual,
                "diagnostic_tolerance": COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
            },
            "pressure_ordering_issues": [
                item.to_dict() for item in self.pressure_ordering_issues
            ],
            "energy_density_ordering_issues": [
                item.to_dict() for item in self.energy_density_ordering_issues
            ],
            "phase_code_changes": self.phase_code_changes,
            "missing_phase_codes": self.missing_phase_codes,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }
