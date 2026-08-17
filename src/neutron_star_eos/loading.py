"""Load analytical, CSV, and CompOSE inputs into the public model handle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from neutron_star_eos.analytical import AnalyticalEos
from neutron_star_eos.capabilities import (
    Capability,
    CapabilityReport,
    barotrope_capabilities,
)
from neutron_star_eos.compose import (
    COMPOSE_ORDERING_POLICIES,
    ComposeColdSlice,
    ComposeDataset,
    ComposeEos,
    build_compose_eos,
    load_compose_dataset,
)
from neutron_star_eos.compose.thermodynamics import interpolate_compose_thermodynamics
from neutron_star_eos.eos import (
    ColdBarotrope,
    EosInputError,
    EosValidationReport,
    _eos_provenance_sha256,
)
from neutron_star_eos.model import EosModel
from neutron_star_eos.tabulated import load_csv_eos

MODEL_KINDS = ("analytical", "csv", "compose")


def _model_from_barotrope(kind: str, eos: ColdBarotrope) -> EosModel:
    validation = eos.validate()
    capabilities = barotrope_capabilities(validation)
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
        item["code"] for item in details.get("cold_slice", {}).get("diagnostics", [])
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
                "available"
                if validation.passed
                else "available_but_physics_gate_failed"
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
    native_status = "available" if not source_codes else "available_with_diagnostics"
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
        Capability("thermodynamics", native_status, diagnostic_codes=source_codes),
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


__all__ = ["MODEL_KINDS", "open_eos"]
