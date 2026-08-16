"""One small, user-facing workflow over the distinct EoS input layers.

The facade in this module deliberately orchestrates existing scientific
objects.  It does not make analytical, ordinary tabulated, and CompOSE inputs
share an interpolation policy.  In particular, a CompOSE model can retain
usable native thermodynamics while its optional continuous stellar barotrope
is unavailable.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from neutron_star_eos.compose import (
    COMPOSE_ORDERING_POLICIES,
    ComposeColdSlice,
    ComposeDataset,
    ComposeEos,
    build_compose_eos,
    load_compose_dataset,
)
from neutron_star_eos.compose_thermodynamics import (
    ComposeThermodynamicProfile,
    interpolate_compose_thermodynamics,
)
from neutron_star_eos.eos import (
    AnalyticalEos,
    ColdBarotrope,
    EosInputError,
    EosValidationReport,
    _eos_provenance_sha256,
)
from neutron_star_eos.stellar import (
    STELLAR_VALIDATION_MODES,
    SequenceResult,
    StarResult,
    StellarConfig,
    solve_sequence,
    solve_star,
)
from neutron_star_eos.tabulated import TabulatedEos, load_csv_eos


MODEL_KINDS = ("analytical", "csv", "compose")
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
            "capabilities": {
                item.name: item.to_dict() for item in self.capabilities
            },
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


@dataclass(frozen=True, slots=True)
class EosModel:
    """High-level handle for one analytical, CSV, or CompOSE input."""

    kind: str
    model_name: str
    _report: CapabilityReport
    barotrope: ColdBarotrope | None = None
    dataset: ComposeDataset | None = None
    cold_slice: ComposeColdSlice | None = None
    native_thermodynamics: ComposeThermodynamicProfile | None = None

    def report(self) -> CapabilityReport:
        return self._report

    def summary(self) -> str:
        return self._report.format_text()

    def require_barotrope(self, *, validation_mode: str = "strict") -> ColdBarotrope:
        if validation_mode not in STELLAR_VALIDATION_MODES:
            raise ValueError(
                f"validation_mode must be one of {STELLAR_VALIDATION_MODES}"
            )
        capability = self._report.capability("continuous_barotrope")
        if self.barotrope is None:
            reason = capability.reason or "continuous stellar barotrope is unavailable"
            raise EosInputError(reason)
        if not capability.available and validation_mode != "background_diagnostic":
            reason = capability.reason or "continuous stellar barotrope is unavailable"
            raise EosInputError(reason)
        return self.barotrope

    def _require_matching_result(self, model_name: str, provenance_sha256: str) -> None:
        eos = self.barotrope
        if eos is None:
            raise EosInputError("cannot write a stellar result without a barotrope")
        expected = _eos_provenance_sha256(eos)
        if model_name != str(eos.model_name) or provenance_sha256 != expected:
            raise EosInputError(
                "stellar result does not belong to this model and EoS provenance"
            )

    def solve_star(
        self,
        central_pressure_mev_fm3: float,
        *,
        config: StellarConfig | None = None,
        retain_profile: bool = False,
        validation_mode: str = "strict",
    ) -> StarResult:
        return solve_star(
            self.require_barotrope(validation_mode=validation_mode),
            central_pressure_mev_fm3,
            config=config,
            retain_profile=retain_profile,
            validation_mode=validation_mode,
        )

    def solve_sequence(
        self,
        central_pressures_mev_fm3: Iterable[float] | None = None,
        *,
        points: int = 50,
        config: StellarConfig | None = None,
        validation_mode: str = "strict",
    ) -> SequenceResult:
        return solve_sequence(
            self.require_barotrope(validation_mode=validation_mode),
            central_pressures_mev_fm3,
            points=points,
            config=config,
            validation_mode=validation_mode,
        )

    def write_inspection(self, output_directory: str | Path) -> Path:
        """Write a deterministic inspection bundle to a new directory."""

        return _write_new_bundle(
            output_directory,
            lambda temporary: _write_inspection_files(temporary, self),
        )

    def write_star(self, output_directory: str | Path, result: StarResult) -> Path:
        """Write one already-computed star and its model report."""

        self._require_matching_result(result.model_name, result.eos_provenance_sha256)
        return _write_new_bundle(
            output_directory,
            lambda temporary: _write_star_files(temporary, self, result),
        )

    def write_sequence(
        self, output_directory: str | Path, result: SequenceResult
    ) -> Path:
        """Write every requested sequence attempt, including failures."""

        self._require_matching_result(
            result.model_name, result.eos_provenance_sha256
        )
        return _write_new_bundle(
            output_directory,
            lambda temporary: _write_sequence_files(temporary, self, result),
        )

    @classmethod
    def from_analytical(
        cls,
        *,
        name: str,
        pressure_from_energy_density: Callable[[Any], Any],
        sound_speed_squared_from_energy_density: Callable[[Any], Any],
        energy_density_domain_mev_fm3: tuple[float, float],
        source: str,
        energy_density_from_pressure: Callable[[Any], Any] | None = None,
    ) -> "EosModel":
        eos = AnalyticalEos(
            name=name,
            pressure_from_energy_density=pressure_from_energy_density,
            sound_speed_squared_from_energy_density=(
                sound_speed_squared_from_energy_density
            ),
            energy_density_domain_mev_fm3=energy_density_domain_mev_fm3,
            source=source,
            energy_density_from_pressure=energy_density_from_pressure,
        )
        return _model_from_barotrope("analytical", eos)


def _barotrope_capabilities(
    report: EosValidationReport,
    *,
    diagnostic_codes: tuple[str, ...] = (),
) -> tuple[Capability, ...]:
    validation_codes = tuple(issue.code for issue in report.issues)
    codes = tuple(dict.fromkeys((*diagnostic_codes, *validation_codes)))
    thermodynamic_status = (
        "available" if not codes else "available_with_diagnostics"
    )
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
        continuous = Capability(
            "continuous_barotrope", "unavailable", reason, codes
        )
        stellar = Capability(
            "stellar_background", "unavailable", reason, codes
        )
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


def _model_from_barotrope(kind: str, eos: ColdBarotrope) -> EosModel:
    validation = eos.validate()
    capabilities = _barotrope_capabilities(validation)
    report = CapabilityReport(
        model_name=eos.model_name,
        input_kind=kind,
        capabilities=capabilities,
        details={
            "validation": validation.to_dict(),
            "barotrope_provenance_sha256": _eos_provenance_sha256(eos),
        },
        provenance=eos.provenance(),
    )
    return EosModel(
        kind=kind,
        model_name=eos.model_name,
        _report=report,
        barotrope=eos,
    )


def _unavailable_compose_model(
    dataset: ComposeDataset,
    *,
    details: dict[str, Any],
    stage: str,
    reason: str,
    cold_slice: ComposeColdSlice | None = None,
) -> EosModel:
    source_diagnostics = tuple(
        item["code"]
        for item in details.get("cold_slice", {}).get("diagnostics", [])
    )
    capabilities = (
        Capability("source", "available"),
        Capability("thermodynamics", "unavailable", reason, source_diagnostics),
        Capability("continuous_barotrope", "unavailable", reason),
        Capability("stellar_background", "unavailable", reason),
        Capability("composition", "unavailable", f"{stage} did not complete"),
        Capability(
            "tidal",
            "unavailable",
            "tidal observables are outside the current continuous-background release",
        ),
    )
    report = CapabilityReport(
        model_name=dataset.model_id,
        input_kind="compose",
        capabilities=capabilities,
        details=details,
        provenance=dataset.provenance(),
    )
    return EosModel(
        kind="compose",
        model_name=dataset.model_id,
        _report=report,
        dataset=dataset,
        cold_slice=cold_slice,
    )


def _open_compose(
    path: str | Path,
    *,
    model_id: str | None,
    source_url: str | None,
    matter: str,
    includes_leptons: bool,
    baryon_density_min_fm3: float | None,
    baryon_density_max_fm3: float | None,
    native_points: int,
    ordering_policy: str,
) -> EosModel:
    if model_id is None or source_url is None:
        raise EosInputError("CompOSE inputs require model_id and source_url")
    if ordering_policy not in COMPOSE_ORDERING_POLICIES:
        raise EosInputError(
            f"ordering_policy must be one of {COMPOSE_ORDERING_POLICIES}"
        )
    dataset = load_compose_dataset(path, model_id=model_id, source_url=source_url)
    details: dict[str, Any] = {
        "dataset": {
            "status": "parsed",
            "model_id": dataset.model_id,
            "provenance": dataset.provenance(),
        }
    }
    try:
        cold = dataset.cold_beta_equilibrium_slice(
            matter=matter,
            includes_leptons=includes_leptons,
        )
    except EosInputError as exc:
        details["cold_slice"] = {"status": "unavailable", "reason": str(exc)}
        return _unavailable_compose_model(
            dataset,
            details=details,
            stage="cold slice",
            reason=str(exc),
        )
    cold_report = cold.report().to_dict()
    details["cold_slice"] = cold_report
    try:
        selected = cold.selected_domain(
            baryon_density_min_fm3=baryon_density_min_fm3,
            baryon_density_max_fm3=baryon_density_max_fm3,
            minimum_rows=2,
        )
        details["density_selection"] = {
            "status": "selected",
            "requested_baryon_density_min_fm3": baryon_density_min_fm3,
            "requested_baryon_density_max_fm3": baryon_density_max_fm3,
            "source_rows": len(selected.rows),
            "source_positions": list(selected.source_positions),
            "baryon_density_min_fm3": float(selected.baryon_density_fm3[0]),
            "baryon_density_max_fm3": float(selected.baryon_density_fm3[-1]),
        }
        profile = interpolate_compose_thermodynamics(
            selected,
            points=native_points,
        )
    except (EosInputError, ValueError) as exc:
        details["native_thermodynamics"] = {
            "status": "unavailable",
            "reason": str(exc),
        }
        return _unavailable_compose_model(
            dataset,
            details=details,
            stage="native thermodynamics",
            reason=str(exc),
            cold_slice=cold,
        )
    native_summary = profile.summary()
    details["native_thermodynamics"] = native_summary
    source_codes = tuple(
        dict.fromkeys(
            item["code"]
            for collection in (cold_report, native_summary)
            for item in collection.get("diagnostics", [])
        )
    )
    barotrope: ComposeEos | None = None
    validation: EosValidationReport | None = None
    barotrope_reason: str | None = None
    try:
        barotrope = build_compose_eos(
            cold,
            baryon_density_min_fm3=baryon_density_min_fm3,
            baryon_density_max_fm3=baryon_density_max_fm3,
            ordering_policy=ordering_policy,
        )
        validation = barotrope.validate()
    except EosInputError as exc:
        barotrope_reason = str(exc)
    if barotrope is None or validation is None:
        details["barotrope"] = {
            "status": "unavailable",
            "reason": barotrope_reason,
        }
        continuous = Capability(
            "continuous_barotrope",
            "unavailable",
            barotrope_reason or "continuous barotrope could not be constructed",
            source_codes,
        )
        stellar = Capability(
            "stellar_background",
            "unavailable",
            barotrope_reason or "continuous barotrope could not be constructed",
            source_codes,
        )
    else:
        validation_codes = tuple(item.code for item in validation.issues)
        combined_codes = tuple(dict.fromkeys((*source_codes, *validation_codes)))
        details["barotrope"] = {
            "status": (
                "available" if validation.passed else "available_but_physics_gate_failed"
            ),
            "validation": validation.to_dict(),
            "provenance": barotrope.provenance(),
            "provenance_sha256": _eos_provenance_sha256(barotrope),
        }
        if validation.passed:
            status = "available" if not combined_codes else "available_with_diagnostics"
            continuous = Capability(
                "continuous_barotrope", status, diagnostic_codes=combined_codes
            )
            stellar = Capability(
                "stellar_background", status, diagnostic_codes=combined_codes
            )
        else:
            reason = "continuous barotrope failed its mechanical/causal validation"
            continuous = Capability(
                "continuous_barotrope", "unavailable", reason, combined_codes
            )
            stellar = Capability(
                "stellar_background", "unavailable", reason, combined_codes
            )
    native_status = (
        "available" if not source_codes else "available_with_diagnostics"
    )
    composition_columns = tuple(
        name
        for name in profile.column_names
        if name.startswith("composition_") and not name.endswith("_available")
    )
    composition_masks = tuple(
        profile.column(f"{name}_available")
        for name in composition_columns
        if f"{name}_available" in profile.column_names
    )
    composition_present = bool(composition_masks) and any(
        np.any(mask > 0.5) for mask in composition_masks
    )
    composition_partial = composition_present and any(
        np.any(mask < 0.5) for mask in composition_masks
    )
    composition_codes = tuple(
        dict.fromkeys(
            (
                *source_codes,
                *(("composition_partial_coverage",) if composition_partial else ()),
            )
        )
    )
    composition = (
        Capability(
            "composition",
            "available" if not composition_codes else "available_with_diagnostics",
            (
                "composition fields cover only part of the selected density path"
                if composition_partial
                else None
            ),
            diagnostic_codes=composition_codes,
        )
        if composition_present
        else Capability(
            "composition",
            "unavailable",
            "the selected CompOSE source does not provide composition rows",
        )
    )
    capabilities = (
        Capability("source", "available"),
        Capability(
            "thermodynamics", native_status, diagnostic_codes=source_codes
        ),
        continuous,
        stellar,
        composition,
        Capability(
            "tidal",
            "unavailable",
            "tidal observables are outside the current continuous-background release",
        ),
    )
    report = CapabilityReport(
        model_name=dataset.model_id,
        input_kind="compose",
        capabilities=capabilities,
        details=details,
        provenance=dataset.provenance(),
    )
    return EosModel(
        kind="compose",
        model_name=dataset.model_id,
        _report=report,
        barotrope=barotrope,
        dataset=dataset,
        cold_slice=cold,
        native_thermodynamics=profile,
    )


def open_eos(
    source: str | Path | AnalyticalEos,
    *,
    kind: str | None = None,
    name: str | None = None,
    source_description: str | None = None,
    epsilon_column: str = "epsilon_mev_fm3",
    pressure_column: str = "pressure_mev_fm3",
    baryon_density_column: str | None = None,
    model_id: str | None = None,
    source_url: str | None = None,
    matter: str = "cold_beta_equilibrated",
    includes_leptons: bool = False,
    baryon_density_min_fm3: float | None = None,
    baryon_density_max_fm3: float | None = None,
    native_points: int = 2001,
    ordering_policy: str = "strict",
) -> EosModel:
    """Open one EoS through an explicit source-specific workflow.

    File kinds are never guessed.  Analytical inputs are existing
    :class:`AnalyticalEos` objects or are constructed with
    :meth:`EosModel.from_analytical`.
    """

    if isinstance(source, AnalyticalEos):
        if kind not in (None, "analytical"):
            raise EosInputError("an AnalyticalEos requires kind='analytical'")
        return _model_from_barotrope("analytical", source)
    if kind not in MODEL_KINDS:
        raise EosInputError(f"kind must be explicitly selected from {MODEL_KINDS}")
    if kind == "analytical":
        raise EosInputError(
            "construct analytical functions with EosModel.from_analytical or pass an AnalyticalEos"
        )
    if kind == "csv":
        eos = load_csv_eos(
            source,
            name=name,
            source=source_description,
            epsilon_column=epsilon_column,
            pressure_column=pressure_column,
            baryon_density_column=baryon_density_column,
        )
        return _model_from_barotrope("csv", eos)
    return _open_compose(
        source,
        model_id=model_id,
        source_url=source_url,
        matter=matter,
        includes_leptons=includes_leptons,
        baryon_density_min_fm3=baryon_density_min_fm3,
        baryon_density_max_fm3=baryon_density_max_fm3,
        native_points=native_points,
        ordering_policy=ordering_policy,
    )


def _write_new_bundle(
    destination: str | Path,
    writer: Callable[[Path], None],
) -> Path:
    resolved = Path(destination).expanduser().resolve()
    if resolved.exists():
        raise EosInputError(f"output directory already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{resolved.name}-", dir=resolved.parent)
    )
    try:
        writer(temporary)
        os.replace(temporary, resolved)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_model_files(directory: Path, model: EosModel) -> None:
    (directory / "summary.txt").write_text(
        model.summary() + "\n", encoding="utf-8", newline="\n"
    )
    _write_json(directory / "report.json", model.report().to_dict())


def _thermodynamic_columns(model: EosModel) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    if model.native_thermodynamics is not None:
        names = model.native_thermodynamics.column_names
        return names, tuple(model.native_thermodynamics.column(name) for name in names)
    eos = model.barotrope
    if isinstance(eos, TabulatedEos):
        names = ["energy_density_mev_fm3", "pressure_mev_fm3"]
        arrays = [eos.energy_density_mev_fm3, eos.pressure_mev_fm3]
        baryon_density = eos.baryon_density_fm3
        if baryon_density is not None:
            names.append("baryon_density_fm3")
            arrays.append(baryon_density)
        return tuple(names), tuple(np.asarray(value) for value in arrays)
    if isinstance(eos, AnalyticalEos):
        epsilon = np.geomspace(
            eos.energy_density_min_mev_fm3,
            eos.energy_density_max_mev_fm3,
            257,
        )
        return (
            (
                "energy_density_mev_fm3",
                "pressure_mev_fm3",
                "sound_speed_squared",
            ),
            (
                epsilon,
                np.asarray(eos.pressure_from_energy_density(epsilon)),
                np.asarray(eos.sound_speed_squared_from_energy_density(epsilon)),
            ),
        )
    raise EosInputError("thermodynamic table is unavailable")


def _write_thermodynamics(path: Path, model: EosModel) -> None:
    names, columns = _thermodynamic_columns(model)
    lengths = {len(column) for column in columns}
    if len(lengths) != 1:
        raise RuntimeError("thermodynamic columns have inconsistent lengths")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(names)
        for row in zip(*columns):
            writer.writerow(
                "" if not np.isfinite(value) else format(float(value), ".17g")
                for value in row
            )


def _write_inspection_files(directory: Path, model: EosModel) -> None:
    _write_model_files(directory, model)
    if model.report().capability("thermodynamics").available:
        _write_thermodynamics(directory / "thermodynamics.csv", model)


def _write_star_files(directory: Path, model: EosModel, result: StarResult) -> None:
    _write_model_files(directory, model)
    _write_json(directory / "star.json", result.to_dict())


def _write_sequence_files(
    directory: Path, model: EosModel, result: SequenceResult
) -> None:
    _write_model_files(directory, model)
    _write_json(directory / "sequence.json", result.to_dict())
    with (directory / "sequence.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = (
            "central_pressure_mev_fm3",
            "status",
            "reason",
            "mass_msun",
            "radius_km",
            "boundary_pressure_mev_fm3",
            "boundary_status",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for attempt in result.attempts:
            star = attempt.star
            writer.writerow(
                {
                    "central_pressure_mev_fm3": format(
                        attempt.central_pressure_mev_fm3, ".17g"
                    ),
                    "status": attempt.status,
                    "reason": attempt.reason or "",
                    "mass_msun": "" if star is None else format(star.mass_msun, ".17g"),
                    "radius_km": "" if star is None else format(star.radius_km, ".17g"),
                    "boundary_pressure_mev_fm3": (
                        ""
                        if star is None
                        else format(star.boundary_pressure_mev_fm3, ".17g")
                    ),
                    "boundary_status": "" if star is None else star.boundary_status,
                }
            )


__all__ = [
    "CAPABILITY_NAMES",
    "CAPABILITY_STATUSES",
    "MODEL_KINDS",
    "Capability",
    "CapabilityReport",
    "EosModel",
    "open_eos",
]
