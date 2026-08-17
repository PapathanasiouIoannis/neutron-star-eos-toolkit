"""User-facing descriptions of the operations available for one EoS."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Mapping

from neutron_star_eos.eos import EosValidationReport

CAPABILITY_STATUSES = (
    "available",
    "available_with_diagnostics",
    "unavailable",
    "not_applicable",
)
CAPABILITY_NAMES = (
    "source",
    "thermodynamics",
    "continuous_barotrope",
    "stellar_background",
    "composition",
    "tidal",
)


def _json_copy(value: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    """Return a detached JSON-compatible copy of governed report data."""

    return json.loads(json.dumps(value, sort_keys=True))


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "uninstalled-source-tree"


def _software_provenance() -> dict[str, str]:
    return {
        "toolkit_version": _distribution_version("neutron-star-eos-toolkit"),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": _distribution_version("numpy"),
        "scipy_version": _distribution_version("scipy"),
    }


@dataclass(frozen=True, slots=True)
class Capability:
    """Availability of one operation, with an explicit reason and evidence."""

    name: str
    status: str
    reason: str | None = None
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in CAPABILITY_NAMES:
            raise ValueError(f"unknown capability {self.name!r}")
        if self.status not in CAPABILITY_STATUSES:
            raise ValueError(f"unknown capability status {self.status!r}")
        if self.status in {"unavailable", "not_applicable"} and not self.reason:
            raise ValueError(f"{self.name} status {self.status!r} requires a reason")

    @property
    def available(self) -> bool:
        return self.status in {"available", "available_with_diagnostics"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "diagnostic_codes": list(self.diagnostic_codes),
        }


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Serializable summary of what a loaded model can and cannot do."""

    model_name: str
    input_kind: str
    capabilities: tuple[Capability, ...]
    details: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def capability(self, name: str) -> Capability:
        for item in self.capabilities:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                code
                for capability in self.capabilities
                for code in capability.diagnostic_codes
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "eos-capability-report-v1",
            "model_name": self.model_name,
            "input_kind": self.input_kind,
            "capabilities": {item.name: item.to_dict() for item in self.capabilities},
            "diagnostic_codes": list(self.diagnostic_codes),
            "software": _software_provenance(),
            "details": _json_copy(dict(self.details)),
            "provenance": _json_copy(dict(self.provenance)),
        }

    def format_text(self) -> str:
        lines = [f"Model: {self.model_name}", f"Input: {self.input_kind}"]
        for item in self.capabilities:
            label = item.name.replace("_", " ").capitalize()
            lines.append(f"{label}: {item.status}")
            if item.reason:
                lines.append(f"  Reason: {item.reason}")
            if item.diagnostic_codes:
                lines.append("  Diagnostics: " + ", ".join(item.diagnostic_codes))
        lines.append("Extrapolation: forbidden")
        return "\n".join(lines)


def barotrope_capabilities(
    report: EosValidationReport,
    *,
    diagnostic_codes: tuple[str, ...] = (),
) -> tuple[Capability, ...]:
    """Translate physical validation results into user-visible operations."""

    validation_codes = tuple(issue.code for issue in report.issues)
    codes = tuple(dict.fromkeys((*diagnostic_codes, *validation_codes)))
    thermodynamic_status = "available" if not codes else "available_with_diagnostics"
    if report.passed:
        continuous = Capability(
            "continuous_barotrope",
            "available" if not codes else "available_with_diagnostics",
            diagnostic_codes=codes,
        )
        stellar = Capability(
            "stellar_background",
            "available" if not codes else "available_with_diagnostics",
            diagnostic_codes=codes,
        )
    else:
        reason = "continuous barotrope failed its mechanical/causal validation"
        continuous = Capability("continuous_barotrope", "unavailable", reason, codes)
        stellar = Capability("stellar_background", "unavailable", reason, codes)
    return (
        Capability("source", "available"),
        Capability("thermodynamics", thermodynamic_status, diagnostic_codes=codes),
        continuous,
        stellar,
        Capability(
            "composition",
            "not_applicable",
            "this input does not declare microscopic composition",
        ),
        Capability(
            "tidal",
            "unavailable",
            "tidal observables are not implemented for the positive-pressure source boundary",
        ),
    )


__all__ = [
    "CAPABILITY_NAMES",
    "CAPABILITY_STATUSES",
    "Capability",
    "CapabilityReport",
    "barotrope_capabilities",
]
