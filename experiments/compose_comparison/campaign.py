"""Run the complete calculation and validation workflow for one EoS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from acceptance import _ordering_sensitivity
from acquire import verify_archive
from calculate import (
    SEQUENCE_FIELDS,
    _causal_endpoint,
    _convergence_rows,
    _positive_source_boundary_sensitivity,
    _save_thermodynamics,
    _sequence_rows,
)
from campaign_io import write_columns as _write_columns
from campaign_io import write_json as _write_json
from campaign_io import write_rows as _write_rows
from compare import (
    _catalogue_check,
    _closure_diagnostics,
    _literature_check,
    _profile_stars,
)
from figures import _save_model_plots
from reference import (
    _archive_metadata_findings,
    _comparison_to_reference,
    _eos_mr_comparison_coverage,
    _eos_mr_source_consistency,
    _reference_metrics,
)
from sampling import (
    _adaptive_sequence,
    _branch_metrics,
    _calculated_branch,
    _compose_eos,
    _refine_target_mass,
    _select_model,
    _validation_mode,
)
from settings import DERIVED_ROOT, FIGURE_ROOT, RUN_SCHEMA_VERSION

from neutron_star_eos import load_compose_mass_radius_reference
from neutron_star_eos.compose import ComposeMassRadiusReference


def _run_model(
    spec: Mapping[str, Any],
    *,
    raw_root: Path,
    quick: bool,
    required_members: Sequence[str],
) -> dict[str, Any]:
    slug = str(spec["slug"])
    archive = raw_root / slug / str(spec["archive"]["filename"])
    acquisition = verify_archive(archive, spec, required_members=required_members)
    acquisition["archive_filename"] = archive.name
    acquisition["local_path"] = str(archive.resolve())
    derived = DERIVED_ROOT / slug
    figures = FIGURE_ROOT / slug
    derived.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    strict, model, ordering_policy, sensitivity_policies = _select_model(spec, archive)
    eos = _compose_eos(model)
    validation_mode = _validation_mode(eos)
    _write_json(derived / "strict_source_report.json", strict.report().to_dict())
    _write_json(derived / "analysis_model_report.json", model.report().to_dict())
    view = _save_thermodynamics(model, derived)
    sequence, sampling, coarse = _adaptive_sequence(
        model, validation_mode=validation_mode, quick=quick
    )
    branch = _calculated_branch(sequence, eos)
    metrics = _branch_metrics(branch)
    metrics["at_1_4_msun"] = _refine_target_mass(
        model, branch, 1.4, validation_mode=validation_mode
    )
    coarse_metrics = _branch_metrics(_calculated_branch(coarse, eos))
    resolution = {
        "refined_minus_coarse": {
            "sampled_peak_mass_msun": float(metrics["sampled_peak"]["mass_msun"])
            - float(coarse_metrics["sampled_peak"]["mass_msun"]),
            "radius_at_sampled_peak_km": float(metrics["sampled_peak"]["radius_km"])
            - float(coarse_metrics["sampled_peak"]["radius_km"]),
            "radius_at_1_4_msun_km": (
                None
                if metrics["at_1_4_msun"] is None
                or coarse_metrics["at_1_4_msun"] is None
                else float(metrics["at_1_4_msun"]["radius_km"])
                - float(coarse_metrics["at_1_4_msun"]["radius_km"])
            ),
        }
    }
    _write_rows(
        derived / "sequence.csv",
        SEQUENCE_FIELDS,
        _sequence_rows(sequence, eos),
    )
    _write_json(derived / "sequence.json", sequence.to_dict())
    causal = _causal_endpoint(model, validation_mode=validation_mode)
    convergence_rows, convergence = _convergence_rows(
        model, metrics, validation_mode=validation_mode
    )
    _write_rows(
        derived / "convergence.csv",
        ("target", "configuration", "ode_rtol", "ode_atol", "mass_msun", "radius_km"),
        convergence_rows,
    )
    positive_boundary = _positive_source_boundary_sensitivity(
        spec, archive, ordering_policy, branch, metrics, eos
    )
    reference: ComposeMassRadiusReference | None = None
    reference_metrics: dict[str, Any] | None = None
    reference_comparison: dict[str, Any] | None = None
    if bool(spec["expected_optional_files"]["eos.mr"]):
        if model.dataset is None:
            raise RuntimeError("CompOSE model lost its parsed dataset")
        reference = load_compose_mass_radius_reference(model.dataset)
        reference_metrics = _reference_metrics(reference)
        reference_comparison = _comparison_to_reference(branch, reference)
        _write_columns(derived / "eos_mr_reference.csv", reference.columns)
        _write_json(derived / "eos_mr_reference.json", reference.to_dict())
    profiles = _profile_stars(model, metrics, validation_mode=validation_mode)
    plots = _save_model_plots(
        model,
        view,
        sequence,
        branch,
        metrics,
        causal,
        reference,
        figures,
        profiles,
    )
    remaining_sequence_failures = sum(item.star is None for item in sequence.attempts)
    ordering_sensitivity = _ordering_sensitivity(
        spec,
        archive,
        ordering_policy,
        sensitivity_policies,
        metrics,
        branch,
        remaining_sequence_failures,
        quick=quick,
        derived_directory=derived,
        figure_directory=figures,
    )
    if ordering_sensitivity is not None:
        ordering_plot = str(ordering_sensitivity["plot_filename"])
        plots["required"].append(ordering_plot)
        if (figures / ordering_plot).is_file():
            plots["created"].append(ordering_plot)
        plots["missing_required"] = sorted(
            set(plots["required"]) - set(plots["created"])
        )
        plots["required_coverage_passed"] = not plots["missing_required"]
    catalogue = _catalogue_check(spec, metrics)
    literature = _literature_check(spec, metrics)
    closure = _closure_diagnostics(view)
    reference_consistency = _eos_mr_source_consistency(
        catalogue, reference_metrics, reference_comparison
    )
    archive_metadata = _archive_metadata_findings(spec, archive)
    acceptance = {
        "catalogue": bool(catalogue["passed"]),
        "convention_classified_literature": bool(literature["acceptance_gate_passed"]),
        "ode_convergence": all(item["passed"] for item in convergence.values()),
        "ordering_analysis_complete_and_catalogue_consistent": (
            True
            if ordering_sensitivity is None
            else bool(ordering_sensitivity["acceptance_gate_passed"])
        ),
        "zero_sequence_failures": remaining_sequence_failures == 0,
        "target_1_4_msun_covered": metrics["at_1_4_msun"] is not None,
        "peak_bracketed": bool(metrics["peak_bracketed_by_sampled_central_densities"]),
        "zero_significant_pre_peak_mass_decreases": (
            int(metrics["pre_peak_mass_decrease_count"]) == 0
        ),
        "causal_endpoint_within_cs2_threshold_tolerance": bool(
            causal.get("sound_speed_within_threshold_tolerance", False)
        ),
        "required_plot_coverage": bool(plots["required_coverage_passed"]),
        "eos_mr_comparison_coverage": _eos_mr_comparison_coverage(reference_comparison),
    }
    acceptance["passed"] = all(acceptance.values())
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "slug": slug,
        "role": spec["role"],
        "model_id": spec["model_id"],
        "compose_eos_id": spec["compose_eos_id"],
        "compose_page_url": spec["compose_page_url"],
        "archive": acquisition,
        "ordering": {
            "strict_source_status": strict.report()
            .capability("continuous_barotrope")
            .status,
            "analysis_policy": ordering_policy,
            "analysis_policy_rationale": (
                None
                if spec.get("ordering_analysis") is None
                else spec["ordering_analysis"]["analysis_policy_rationale"]
            ),
            "acceptance_policy": (
                None
                if spec.get("ordering_analysis") is None
                else spec["ordering_analysis"]["acceptance_policy"]
            ),
            "sensitivity": ordering_sensitivity,
        },
        "validation_mode": validation_mode,
        "validation": eos.validate().to_dict(),
        "sampling": sampling,
        "metrics": metrics,
        "coarse_grid_metrics": coarse_metrics,
        "sequence_resolution": resolution,
        "remaining_sequence_failures": remaining_sequence_failures,
        "causal_endpoint": causal,
        "positive_source_boundary_sensitivity": positive_boundary,
        "convergence": convergence,
        "compose_catalogue_crosscheck": catalogue,
        "literature_crosscheck": literature,
        "closure_residual_diagnostics": closure,
        "eos_mr_reference": reference_metrics,
        "eos_mr_crosscheck": reference_comparison,
        "eos_mr_source_consistency": reference_consistency,
        "primary_citations": spec["primary_citations"],
        "analysis_notes": spec["analysis_notes"],
        "plots": plots,
        "non_gating_findings": {
            "material_source_seam_systematic": bool(
                ordering_sensitivity is not None
                and ordering_sensitivity["classification"]
                == "material_source_seam_systematic"
            ),
            "conditional_radius_span_at_1_4_msun_km": (
                None
                if ordering_sensitivity is None
                else ordering_sensitivity["conditional_radius_span_at_1_4_msun_km"]
            ),
            "eos_mr_source_consistency": reference_consistency,
            "material_optional_reference_source_inconsistency": bool(
                reference_consistency["classification"]
                == "material_optional_reference_source_inconsistency"
            ),
            "material_optional_reference_discrepancy": bool(
                reference_consistency["material"]
            ),
            "archive_metadata_provenance": archive_metadata,
        },
        "acceptance": acceptance,
        "interpretation": {
            "mass_radius_source": "toolkit TOV calculation",
            "eos_mr_role": "independent_reference_not_solver_input",
            "boundary": (
                "lowest retained positive source pressure, not P=0; the common "
                "positive-source-boundary check does not measure omitted surface layers"
            ),
            "sampled_peak_is_exact_maximum_claim": False,
            "pre_peak_segment_is_stability_claim": False,
            "closure_residuals_are_acceptance_gate": False,
        },
    }
    _write_json(derived / "summary.json", summary)
    return summary
