"""Compare calculated mass-radius curves with optional CompOSE references."""

from __future__ import annotations

import hashlib
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from settings import (
    OPTIONAL_REFERENCE_RADIUS_MATERIALITY_THRESHOLD_KM,
    REFERENCE_FIXED_MASSES_MSUN,
    REFERENCE_PEAK_EXCLUSION_MARGIN_MSUN,
    BranchData,
)

from neutron_star_eos.compose import ComposeMassRadiusReference


def _reference_selected_peak_side(
    reference: ComposeMassRadiusReference,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    peak = int(np.argmax(reference.mass_msun))
    left = np.arange(0, peak + 1)
    right = np.arange(peak, reference.rows)
    left_span = float(np.ptp(reference.mass_msun[left]))
    right_span = float(np.ptp(reference.mass_msun[right]))
    use_left = left_span >= right_span
    indices = left if use_left else right
    order = np.argsort(reference.mass_msun[indices])
    indices = indices[order]
    unique_mass, unique_indices = np.unique(
        reference.mass_msun[indices], return_index=True
    )
    indices = indices[unique_indices]
    return (
        unique_mass,
        reference.radius_km[indices],
        {
            "algorithm": (
                "choose the source-order side of the global sampled peak with "
                "the larger mass span, then sort that side by mass"
            ),
            "selected_source_side": (
                "source_start_through_peak" if use_left else "peak_through_source_end"
            ),
            "left_mass_span_msun": left_span,
            "right_mass_span_msun": right_span,
            "physical_branch_or_stability_inferred": False,
        },
    )


def _reference_metrics(reference: ComposeMassRadiusReference) -> dict[str, Any]:
    peak = int(np.argmax(reference.mass_msun))
    mass, radius, selection = _reference_selected_peak_side(reference)
    at_1_4 = (
        float(np.interp(1.4, mass, radius))
        if float(mass[0]) <= 1.4 <= float(mass[-1])
        else None
    )
    return {
        "role": "independent_reference_not_solver_input",
        "rows": reference.rows,
        "sampled_peak_mass_msun": float(reference.mass_msun[peak]),
        "radius_at_sampled_peak_km": float(reference.radius_km[peak]),
        "radius_at_1_4_msun_km": at_1_4,
        "source_order_peak_index": peak,
        "source_order_peak_side_selection": selection,
        "provenance": reference.provenance(),
    }


def _comparison_to_reference(
    calculated: BranchData, reference: ComposeMassRadiusReference
) -> dict[str, Any]:
    calc_stop = calculated.peak_index + 1
    calc_mass = calculated.mass_msun[:calc_stop]
    calc_radius = calculated.radius_km[:calc_stop]
    ref_mass, ref_radius, selection = _reference_selected_peak_side(reference)
    lower = max(float(np.min(calc_mass)), float(ref_mass[0]))
    common_peak_mass = min(float(np.max(calc_mass)), float(ref_mass[-1]))
    safe_upper = common_peak_mass - REFERENCE_PEAK_EXCLUSION_MARGIN_MSUN
    calc_order = np.argsort(calc_mass)
    fixed_masses = np.asarray(
        [mass for mass in REFERENCE_FIXED_MASSES_MSUN if lower <= mass <= safe_upper],
        dtype=float,
    )
    residual = np.asarray(
        [
            np.interp(mass, calc_mass[calc_order], calc_radius[calc_order])
            - np.interp(mass, ref_mass, ref_radius)
            for mass in fixed_masses
        ],
        dtype=float,
    )
    reference_peak = int(np.argmax(reference.mass_msun))
    fixed_rows = [
        {
            "mass_msun": float(mass),
            "calculated_radius_km": float(
                np.interp(mass, calc_mass[calc_order], calc_radius[calc_order])
            ),
            "reference_radius_km": float(np.interp(mass, ref_mass, ref_radius)),
            "calculated_minus_reference_radius_km": float(delta),
        }
        for mass, delta in zip(fixed_masses, residual)
    ]
    residual_metrics = (
        None
        if not len(residual)
        else {
            "mean": float(np.mean(residual)),
            "rms": float(np.sqrt(np.mean(residual**2))),
            "maximum_absolute": float(np.max(np.abs(residual))),
        }
    )
    return {
        "method": (
            "fixed masses on the calculated pre-peak segment and selected eos.mr "
            "source-order peak side, excluding a margin below the common peak"
        ),
        "physical_stability_inferred_for_eos_mr": False,
        "overlap_mass_msun": [lower, common_peak_mass],
        "turning_point_exclusion_margin_msun": REFERENCE_PEAK_EXCLUSION_MARGIN_MSUN,
        "safe_fixed_mass_ceiling_msun": safe_upper,
        "candidate_fixed_masses_msun": list(REFERENCE_FIXED_MASSES_MSUN),
        "fixed_mass_comparisons": fixed_rows,
        "comparison_points": len(fixed_rows),
        "reference_source_order_peak_side_selection": selection,
        "radius_residual_calculated_minus_reference_km": residual_metrics,
        "sampled_peak_coordinate_comparison": {
            "calculated": {
                "mass_msun": float(calculated.mass_msun[calculated.peak_index]),
                "radius_km": float(calculated.radius_km[calculated.peak_index]),
            },
            "reference": {
                "mass_msun": float(reference.mass_msun[reference_peak]),
                "radius_km": float(reference.radius_km[reference_peak]),
            },
            "calculated_minus_reference": {
                "mass_msun": float(calculated.mass_msun[calculated.peak_index])
                - float(reference.mass_msun[reference_peak]),
                "radius_km": float(calculated.radius_km[calculated.peak_index])
                - float(reference.radius_km[reference_peak]),
            },
        },
    }


def _eos_mr_source_consistency(
    catalogue: Mapping[str, Any],
    reference_metrics: Mapping[str, Any] | None,
    reference_comparison: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Classify an optional ``eos.mr`` comparison without making it a gate."""

    threshold = OPTIONAL_REFERENCE_RADIUS_MATERIALITY_THRESHOLD_KM
    result: dict[str, Any] = {
        "classification": "unavailable",
        "classification_complete": True,
        "material": False,
        "acceptance_gate": False,
        "basis": (
            "triangle comparison among the independent toolkit TOV calculation, "
            "the current CompOSE catalogue benchmark, and optional eos.mr"
        ),
        "thresholds": {
            "radius_km": threshold,
            "role": "diagnostic_materiality_threshold_not_uncertainty",
            "source_attribution_rule": (
                "both absolute catalogue-minus-reference and exact "
                "calculated-minus-reference R1.4 differences must be strictly "
                "greater than the threshold after the catalogue gate passes"
            ),
        },
        "catalogue_gate_passed": bool(catalogue["passed"]),
        "r1_4_triangle_complete": False,
        "source_attribution_condition_met": False,
        "catalogue_minus_reference_radius_at_1_4_msun_km": None,
        "calculated_minus_reference_radius_at_1_4_msun_km": None,
        "calculated_minus_reference_r1_4_is_exact": False,
        "maximum_absolute_fixed_mass_radius_residual_km": None,
        "rms_fixed_mass_radius_residual_km": None,
        "threshold_exceeded_by": [],
        "cause": None,
        "interpretation": "optional eos.mr is not available for this source archive",
    }
    if reference_metrics is None:
        return result

    has_comparison_overlap = bool(
        reference_comparison is not None
        and int(reference_comparison.get("comparison_points", 0)) > 0
    )

    catalogue_radius = catalogue["benchmark"].get("radius_at_1_4_msun_km")
    reference_radius = reference_metrics.get("radius_at_1_4_msun_km")
    catalogue_minus_reference = (
        None
        if catalogue_radius is None or reference_radius is None
        else float(catalogue_radius) - float(reference_radius)
    )
    calculated_minus_catalogue = catalogue.get("calculated_minus_benchmark", {}).get(
        "radius_at_1_4_msun_km"
    )
    exact_calculated_minus_reference = (
        None
        if catalogue_minus_reference is None or calculated_minus_catalogue is None
        else catalogue_minus_reference + float(calculated_minus_catalogue)
    )
    calculated_minus_reference = exact_calculated_minus_reference
    if calculated_minus_reference is None:
        fixed_rows = (
            ()
            if reference_comparison is None
            else reference_comparison.get("fixed_mass_comparisons", ())
        )
        for row in fixed_rows:
            if math.isclose(float(row["mass_msun"]), 1.4, rel_tol=0.0, abs_tol=1.0e-12):
                calculated_minus_reference = float(
                    row["calculated_minus_reference_radius_km"]
                )
                break
    residuals = (
        None
        if reference_comparison is None
        else reference_comparison.get("radius_residual_calculated_minus_reference_km")
    )
    maximum_absolute = None if residuals is None else residuals.get("maximum_absolute")
    rms = None if residuals is None else residuals.get("rms")
    exceeded_by: list[str] = []

    def exceeds_threshold(value: Any) -> bool:
        return value is not None and abs(float(value)) > threshold

    if exceeds_threshold(catalogue_minus_reference):
        exceeded_by.append("catalogue_minus_reference_radius_at_1_4_msun_km")
    if exceeds_threshold(calculated_minus_reference):
        exceeded_by.append("calculated_minus_reference_radius_at_1_4_msun_km")
    if exceeds_threshold(maximum_absolute):
        exceeded_by.append("maximum_absolute_fixed_mass_radius_residual_km")
    if exceeds_threshold(rms):
        exceeded_by.append("rms_fixed_mass_radius_residual_km")

    triangle_complete = (
        catalogue_minus_reference is not None
        and exact_calculated_minus_reference is not None
    )
    attribution_condition_met = bool(
        catalogue["passed"]
        and triangle_complete
        and exceeds_threshold(catalogue_minus_reference)
        and exceeds_threshold(exact_calculated_minus_reference)
    )

    if not bool(catalogue["passed"]):
        classification = "indeterminate_calculation_catalogue_disagreement"
        material = bool(exceeded_by)
        cause = (
            "the calculation did not independently pass the catalogue gate, so no "
            "optional-reference source attribution is permitted"
        )
        interpretation = (
            "calculation-catalogue disagreement makes the optional-reference "
            "diagnostics indeterminate and prevents identifying any inconsistent "
            "source vertex"
        )
    elif attribution_condition_met:
        classification = "material_optional_reference_source_inconsistency"
        material = True
        cause = "undocumented upstream/source-internal cause; not solver failure"
        interpretation = (
            "the independent toolkit TOV calculation passes the current CompOSE "
            "catalogue gate while optional eos.mr disagrees beyond the diagnostic "
            "materiality threshold; eos.mr was never solver input and the finding "
            "does not fail calculation acceptance"
        )
    elif exceeded_by:
        classification = "material_optional_reference_discrepancy_unattributed"
        material = True
        cause = (
            "the complete corroborating exact R1.4 triangle required for source "
            "attribution is absent or non-corroborating"
        )
        interpretation = (
            "a material optional-reference diagnostic discrepancy is present, but "
            "no individual triangle vertex is identified as its cause; the finding "
            "is explicitly non-gating"
        )
    elif not has_comparison_overlap:
        classification = "no_overlap"
        material = False
        cause = None
        interpretation = (
            "optional eos.mr is present, but no predeclared fixed-mass comparison "
            "point lies in the common pre-peak mass domain"
        )
    else:
        classification = "consistent"
        material = False
        cause = None
        interpretation = (
            "optional eos.mr is consistent with the independently catalogue-checked "
            "toolkit calculation within the diagnostic materiality threshold"
        )
    result.update(
        {
            "classification": classification,
            "material": material,
            "r1_4_triangle_complete": triangle_complete,
            "source_attribution_condition_met": attribution_condition_met,
            "catalogue_minus_reference_radius_at_1_4_msun_km": (
                catalogue_minus_reference
            ),
            "calculated_minus_reference_radius_at_1_4_msun_km": (
                calculated_minus_reference
            ),
            "calculated_minus_reference_r1_4_is_exact": (
                exact_calculated_minus_reference is not None
            ),
            "maximum_absolute_fixed_mass_radius_residual_km": maximum_absolute,
            "rms_fixed_mass_radius_residual_km": rms,
            "threshold_exceeded_by": exceeded_by,
            "cause": cause,
            "interpretation": interpretation,
        }
    )
    return result


def _eos_mr_comparison_coverage(
    reference_comparison: Mapping[str, Any] | None,
) -> bool:
    """Return whether an absent or present optional reference has usable coverage."""

    return (
        reference_comparison is None
        or int(reference_comparison.get("comparison_points", 0)) > 0
    )


def _archive_metadata_findings(
    spec: Mapping[str, Any], archive: Path
) -> list[dict[str, Any]]:
    """Verify and materialize registry-declared archive metadata findings."""

    declarations = spec.get("archive_metadata_findings", [])
    findings: list[dict[str, Any]] = []
    if not declarations:
        return findings
    with zipfile.ZipFile(archive) as source:
        for declaration in declarations:
            member = str(declaration["source_member"])
            try:
                info = source.getinfo(member)
            except KeyError as exc:
                raise RuntimeError(
                    f"declared archive metadata member missing for {spec['slug']}: "
                    f"{member}"
                ) from exc
            content = source.read(member)
            digest = hashlib.sha256(content).hexdigest()
            expected_bytes = int(declaration["source_member_bytes"])
            expected_digest = str(declaration["source_member_sha256"])
            if info.file_size != expected_bytes or digest != expected_digest:
                raise RuntimeError(
                    f"declared archive metadata evidence changed for {spec['slug']}: "
                    f"{member}"
                )
            findings.append(
                {
                    "classification": declaration["classification"],
                    "acceptance_gate": False,
                    "solver_input": False,
                    "source_member": {
                        "name": member,
                        "bytes": info.file_size,
                        "sha256": digest,
                        "identity_verified": True,
                    },
                    "declared_embedded_benchmark": dict(
                        declaration["declared_embedded_benchmark"]
                    ),
                    "authoritative_compose_benchmark": dict(spec["compose_benchmark"]),
                    "cause": declaration["cause"],
                    "interpretation": declaration["interpretation"],
                }
            )
    return findings
