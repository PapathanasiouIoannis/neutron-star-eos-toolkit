"""Apply predeclared campaign gates and classify non-gating findings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from calculate import SEQUENCE_FIELDS, _sequence_rows
from campaign_io import write_json as _write_json
from campaign_io import write_rows as _write_rows
from compare import _catalogue_check
from figures import _plot_style, _save_ax
from sampling import (
    _adaptive_sequence,
    _branch_metrics,
    _calculated_branch,
    _compose_eos,
    _open_model,
    _refine_target_mass,
    _validation_mode,
)
from settings import (
    COLORS,
    NOMINAL_SEAM_MASS_DELTA_MSUN,
    NOMINAL_SEAM_RADIUS_DELTA_KM,
    ORDERING_ACCEPTANCE_POLICY,
    BranchData,
)


def _ordering_policy_label(policy: str) -> str:
    labels = {
        "diagnostic_monotone_subsequence": "keep-first diagnostic reduction",
        "diagnostic_keep_later_monotone_subsequence": (
            "keep-later diagnostic reduction"
        ),
    }
    try:
        return labels[policy]
    except KeyError as exc:
        raise ValueError(f"unsupported ordering policy label: {policy}") from exc


def _ordering_branch_complete(
    metrics: Mapping[str, Any], *, remaining_failures: int
) -> bool:
    return bool(
        remaining_failures == 0
        and metrics["at_1_4_msun"] is not None
        and metrics["peak_bracketed_by_sampled_central_densities"]
        and int(metrics["pre_peak_mass_decrease_count"]) == 0
    )


def _ordering_analysis_assessment(
    spec: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    alternative_metrics: Mapping[str, Any],
    *,
    baseline_remaining_failures: int,
    alternative_remaining_failures: int,
) -> dict[str, Any]:
    """Classify a declared seam without treating a small delta as acceptance."""

    ordering = spec["ordering_analysis"]
    if ordering["acceptance_policy"] != ORDERING_ACCEPTANCE_POLICY:
        raise RuntimeError(f"unsupported ordering acceptance policy for {spec['slug']}")
    baseline_peak = baseline_metrics["sampled_peak"]
    alternative_peak = alternative_metrics["sampled_peak"]
    baseline_1_4 = baseline_metrics["at_1_4_msun"]
    alternative_1_4 = alternative_metrics["at_1_4_msun"]
    delta_mass = float(alternative_peak["mass_msun"]) - float(
        baseline_peak["mass_msun"]
    )
    delta_r14 = (
        None
        if baseline_1_4 is None or alternative_1_4 is None
        else float(alternative_1_4["radius_km"]) - float(baseline_1_4["radius_km"])
    )
    baseline_complete = _ordering_branch_complete(
        baseline_metrics, remaining_failures=baseline_remaining_failures
    )
    alternative_complete = _ordering_branch_complete(
        alternative_metrics, remaining_failures=alternative_remaining_failures
    )
    baseline_catalogue = _catalogue_check(spec, baseline_metrics)
    alternative_catalogue = _catalogue_check(spec, alternative_metrics)
    numerical_completion_passed = baseline_complete and alternative_complete
    both_catalogue_consistent = bool(
        baseline_catalogue["passed"] and alternative_catalogue["passed"]
    )
    delta_within_nominal_tolerance = bool(
        abs(delta_mass) <= NOMINAL_SEAM_MASS_DELTA_MSUN
        and delta_r14 is not None
        and abs(delta_r14) <= NOMINAL_SEAM_RADIUS_DELTA_KM
    )
    if not numerical_completion_passed:
        classification = "incomplete_ordering_analysis"
    elif delta_within_nominal_tolerance:
        classification = "within_nominal_delta_tolerance"
    else:
        classification = "material_source_seam_systematic"
    conditional_span = None
    if baseline_1_4 is not None and alternative_1_4 is not None:
        if delta_r14 is None:
            raise RuntimeError("ordering R1.4 delta is unexpectedly unavailable")
        conditional_span = {
            "lower": min(
                float(baseline_1_4["radius_km"]),
                float(alternative_1_4["radius_km"]),
            ),
            "upper": max(
                float(baseline_1_4["radius_km"]),
                float(alternative_1_4["radius_km"]),
            ),
            "span": abs(delta_r14),
        }
    reversals = ordering["expected_pressure_issues"]
    maximum_reversal = max(abs(float(item["relative_change"])) for item in reversals)
    return {
        "acceptance_policy": ORDERING_ACCEPTANCE_POLICY,
        "analysis_policy_rationale": ordering["analysis_policy_rationale"],
        "numerical_completion": {
            "baseline": baseline_complete,
            "alternative": alternative_complete,
            "passed": numerical_completion_passed,
        },
        "catalogue_consistency": {
            "baseline": baseline_catalogue,
            "alternative": alternative_catalogue,
            "both_reductions_passed": both_catalogue_consistent,
        },
        "alternative_minus_baseline": {
            "sampled_peak_mass_msun": delta_mass,
            "radius_at_1_4_msun_km": delta_r14,
        },
        "nominal_delta_tolerances_not_acceptance_gates": {
            "sampled_peak_mass_msun": NOMINAL_SEAM_MASS_DELTA_MSUN,
            "radius_at_1_4_msun_km": NOMINAL_SEAM_RADIUS_DELTA_KM,
        },
        "delta_within_nominal_tolerance": delta_within_nominal_tolerance,
        "classification": classification,
        "conditional_radius_span_at_1_4_msun_km": conditional_span,
        "maximum_declared_source_pressure_reversal_fraction": maximum_reversal,
        "acceptance_gate_passed": bool(
            numerical_completion_passed and both_catalogue_consistent
        ),
        "physical_transition_resolved": False,
        "conditional_span_is_statistical_uncertainty": False,
        "interpretation": (
            "acceptance requires both declared diagnostic reductions to complete "
            "and independently reproduce the predeclared CompOSE catalogue "
            "benchmarks; their delta is a reported source-construction finding, "
            "not a tuned acceptance tolerance or a physical transition solution"
        ),
    }


def _ordering_sensitivity(
    spec: Mapping[str, Any],
    archive: Path,
    baseline_policy: str,
    policies: Sequence[str],
    baseline_metrics: Mapping[str, Any],
    baseline_branch: BranchData,
    baseline_remaining_failures: int,
    *,
    quick: bool,
    derived_directory: Path,
    figure_directory: Path,
) -> dict[str, Any] | None:
    alternatives = [policy for policy in policies if policy != baseline_policy]
    if not alternatives:
        return None
    policy = alternatives[0]
    alternative = _open_model(spec, archive, ordering_policy=policy, native_points=1501)
    mode = _validation_mode(_compose_eos(alternative))
    sequence, sampling, _coarse = _adaptive_sequence(
        alternative, validation_mode=mode, quick=quick
    )
    branch = _calculated_branch(sequence, _compose_eos(alternative))
    metrics = _branch_metrics(branch)
    metrics["at_1_4_msun"] = _refine_target_mass(
        alternative, branch, 1.4, validation_mode=mode
    )
    _write_rows(
        derived_directory / "sequence_ordering_alternative.csv",
        SEQUENCE_FIELDS,
        _sequence_rows(sequence, _compose_eos(alternative)),
    )
    _write_json(
        derived_directory / "sequence_ordering_alternative.json", sequence.to_dict()
    )
    remaining_failures = sum(item.star is None for item in sequence.attempts)
    assessment = _ordering_analysis_assessment(
        spec,
        baseline_metrics,
        metrics,
        baseline_remaining_failures=baseline_remaining_failures,
        alternative_remaining_failures=remaining_failures,
    )
    with plt.rc_context(_plot_style()):
        figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.plot(
            baseline_branch.radius_km[: baseline_branch.peak_index + 1],
            baseline_branch.mass_msun[: baseline_branch.peak_index + 1],
            color=COLORS[0],
            linewidth=2.2,
            label=_ordering_policy_label(baseline_policy),
        )
        ax.plot(
            branch.radius_km[: branch.peak_index + 1],
            branch.mass_msun[: branch.peak_index + 1],
            color=COLORS[1],
            linestyle="--",
            linewidth=2.0,
            label=_ordering_policy_label(policy),
        )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title(f"{spec['model_id']}: ordering-seam sensitivity")
        ax.legend(loc="best")
        _save_ax(ax, figure_directory / "ordering_policy_sensitivity_mass_radius.png")
    return {
        "baseline_policy": baseline_policy,
        "alternative_policy": policy,
        "alternative_sampling": sampling,
        "alternative_metrics": metrics,
        "remaining_sequence_failures": remaining_failures,
        "plot_filename": "ordering_policy_sensitivity_mass_radius.png",
        **assessment,
    }


def _campaign_interpretive_status(
    summaries: Sequence[Mapping[str, Any]], *, campaign_passed: bool
) -> str:
    if not campaign_passed:
        return "FAIL"
    has_material_upstream_reference_finding = any(
        bool(
            item["non_gating_findings"][
                "material_optional_reference_source_inconsistency"
            ]
        )
        or bool(item["non_gating_findings"]["archive_metadata_provenance"])
        for item in summaries
    )
    has_material_unattributed_diagnostic = any(
        bool(item["non_gating_findings"]["material_optional_reference_discrepancy"])
        and not bool(
            item["non_gating_findings"][
                "material_optional_reference_source_inconsistency"
            ]
        )
        for item in summaries
    )
    if has_material_upstream_reference_finding and has_material_unattributed_diagnostic:
        return "PASS_WITH_MATERIAL_UPSTREAM_AND_UNATTRIBUTED_DIAGNOSTIC_FINDINGS"
    if has_material_upstream_reference_finding:
        return "PASS_WITH_MATERIAL_UPSTREAM_REFERENCE_FINDINGS"
    if has_material_unattributed_diagnostic:
        return "PASS_WITH_MATERIAL_UNATTRIBUTED_DIAGNOSTIC_FINDINGS"
    return "PASS"


def _optional_reference_campaign_findings(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Collect all optional-reference states and their material subsets."""

    consistency = {
        str(item["slug"]): item["eos_mr_source_consistency"] for item in summaries
    }
    return {
        "eos_mr_source_consistency": consistency,
        "material_optional_reference_source_inconsistencies": {
            slug: finding
            for slug, finding in consistency.items()
            if finding["classification"]
            == "material_optional_reference_source_inconsistency"
        },
        "material_optional_reference_discrepancies": {
            slug: finding
            for slug, finding in consistency.items()
            if bool(finding["material"])
        },
    }
