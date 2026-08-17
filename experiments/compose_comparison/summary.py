"""Flatten per-model campaign summaries into the comparison CSV schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _summary_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summaries:
        peak = item["metrics"]["sampled_peak"]
        at_1_4 = item["metrics"]["at_1_4_msun"]
        causal = item["causal_endpoint"]
        catalogue = item["compose_catalogue_crosscheck"]
        literature = item["literature_crosscheck"]
        reference_consistency = item["eos_mr_source_consistency"]
        ordering_sensitivity = item["ordering"]["sensitivity"]
        ordering_span = (
            None
            if ordering_sensitivity is None
            else ordering_sensitivity["conditional_radius_span_at_1_4_msun_km"]
        )
        literature_checks = literature["numeric_checks"]
        closure_maxima = item["closure_residual_diagnostics"][
            "maximum_absolute_normalized_residual"
        ]
        finite_closure_maxima = [
            float(value) for value in closure_maxima.values() if value is not None
        ]
        rows.append(
            {
                "slug": item["slug"],
                "role": item["role"],
                "model_id": item["model_id"],
                "sampled_peak_mass_msun": peak["mass_msun"],
                "radius_at_sampled_peak_km": peak["radius_km"],
                "radius_at_1_4_msun_km": None
                if at_1_4 is None
                else at_1_4["radius_km"],
                "causal_endpoint_status": causal["status"],
                "causal_endpoint_mass_msun": causal.get("mass_msun"),
                "causal_endpoint_density_fm3": causal.get("central_baryon_density_fm3"),
                "catalogue_delta_peak_mass_msun": catalogue[
                    "calculated_minus_benchmark"
                ]["sampled_peak_mass_msun"],
                "catalogue_delta_peak_radius_km": catalogue[
                    "calculated_minus_benchmark"
                ]["radius_at_sampled_peak_km"],
                "catalogue_delta_r1_4_km": catalogue["calculated_minus_benchmark"][
                    "radius_at_1_4_msun_km"
                ],
                "catalogue_check_passed": catalogue["passed"],
                "eos_mr_source_consistency_classification": (
                    reference_consistency["classification"]
                ),
                "catalogue_minus_eos_mr_r1_4_km": reference_consistency[
                    "catalogue_minus_reference_radius_at_1_4_msun_km"
                ],
                "calculated_minus_eos_mr_r1_4_km": reference_consistency[
                    "calculated_minus_reference_radius_at_1_4_msun_km"
                ],
                "maximum_absolute_fixed_mass_eos_mr_radius_residual_km": (
                    reference_consistency[
                        "maximum_absolute_fixed_mass_radius_residual_km"
                    ]
                ),
                "rms_fixed_mass_eos_mr_radius_residual_km": reference_consistency[
                    "rms_fixed_mass_radius_residual_km"
                ],
                "eos_mr_source_consistency_acceptance_gate": reference_consistency[
                    "acceptance_gate"
                ],
                "eos_mr_material_discrepancy": reference_consistency["material"],
                "archive_metadata_provenance_finding_count": len(
                    item["non_gating_findings"]["archive_metadata_provenance"]
                ),
                "ordering_sensitivity_classification": (
                    None
                    if ordering_sensitivity is None
                    else ordering_sensitivity["classification"]
                ),
                "ordering_conditional_r1_4_span_km": (
                    None if ordering_span is None else ordering_span["span"]
                ),
                "ordering_analysis_acceptance_gate_passed": (
                    True
                    if ordering_sensitivity is None
                    else ordering_sensitivity["acceptance_gate_passed"]
                ),
                "literature_comparability": literature["comparability"],
                "literature_numeric_comparison_passed": literature[
                    "numeric_comparison_passed"
                ],
                "literature_delta_peak_mass_msun": literature_checks.get(
                    "maximum_mass_msun", {}
                ).get("calculated_minus_literature"),
                "literature_delta_peak_radius_km": literature_checks.get(
                    "radius_at_maximum_mass_km", {}
                ).get("calculated_minus_literature"),
                "literature_delta_r1_4_km": literature_checks.get(
                    "radius_at_1_4_msun_km", {}
                ).get("calculated_minus_literature"),
                "literature_acceptance_status": literature["acceptance_status"],
                "literature_acceptance_gate_passed": literature[
                    "acceptance_gate_passed"
                ],
                "maximum_closure_residual_diagnostic_only": (
                    None if not finite_closure_maxima else max(finite_closure_maxima)
                ),
                "remaining_sequence_failures": item["remaining_sequence_failures"],
                "required_plot_coverage_passed": item["plots"][
                    "required_coverage_passed"
                ],
                "overall_acceptance_passed": item["acceptance"]["passed"],
            }
        )
    return rows


SUMMARY_FIELDS = (
    "slug",
    "role",
    "model_id",
    "sampled_peak_mass_msun",
    "radius_at_sampled_peak_km",
    "radius_at_1_4_msun_km",
    "causal_endpoint_status",
    "causal_endpoint_mass_msun",
    "causal_endpoint_density_fm3",
    "catalogue_delta_peak_mass_msun",
    "catalogue_delta_peak_radius_km",
    "catalogue_delta_r1_4_km",
    "catalogue_check_passed",
    "eos_mr_source_consistency_classification",
    "catalogue_minus_eos_mr_r1_4_km",
    "calculated_minus_eos_mr_r1_4_km",
    "maximum_absolute_fixed_mass_eos_mr_radius_residual_km",
    "rms_fixed_mass_eos_mr_radius_residual_km",
    "eos_mr_source_consistency_acceptance_gate",
    "eos_mr_material_discrepancy",
    "archive_metadata_provenance_finding_count",
    "ordering_sensitivity_classification",
    "ordering_conditional_r1_4_span_km",
    "ordering_analysis_acceptance_gate_passed",
    "literature_comparability",
    "literature_numeric_comparison_passed",
    "literature_delta_peak_mass_msun",
    "literature_delta_peak_radius_km",
    "literature_delta_r1_4_km",
    "literature_acceptance_status",
    "literature_acceptance_gate_passed",
    "maximum_closure_residual_diagnostic_only",
    "remaining_sequence_failures",
    "required_plot_coverage_passed",
    "overall_acceptance_passed",
)
