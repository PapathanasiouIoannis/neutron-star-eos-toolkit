"""Write the human-readable scientific interpretation of a campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from acceptance import _ordering_policy_label
from settings import RESULTS_ROOT


def _ordering_systematic_report_line(item: Mapping[str, Any]) -> str | None:
    sensitivity = item["ordering"]["sensitivity"]
    if sensitivity is None:
        return None
    span = sensitivity["conditional_radius_span_at_1_4_msun_km"]
    baseline_1_4 = item["metrics"]["at_1_4_msun"]
    alternative_1_4 = sensitivity["alternative_metrics"]["at_1_4_msun"]
    if span is None or baseline_1_4 is None or alternative_1_4 is None:
        return (
            f"- **{item['model_id']}** has an incomplete declared ordering analysis; "
            "campaign acceptance must fail."
        )
    if sensitivity["classification"] == "incomplete_ordering_analysis":
        return (
            f"- **{item['model_id']}** did not complete both declared ordering "
            "reductions without numerical findings; campaign acceptance failed."
        )
    if not sensitivity["catalogue_consistency"]["both_reductions_passed"]:
        return (
            f"- **{item['model_id']}** has at least one diagnostic ordering "
            "reduction outside the predeclared CompOSE catalogue tolerances; "
            "campaign acceptance failed."
        )
    delta = sensitivity["alternative_minus_baseline"]["radius_at_1_4_msun_km"]
    baseline_label = _ordering_policy_label(sensitivity["baseline_policy"]).split()[0]
    alternative_label = _ordering_policy_label(
        sensitivity["alternative_policy"]
    ).split()[0]
    if sensitivity["classification"] == "material_source_seam_systematic":
        catalogue_tolerance = sensitivity["catalogue_consistency"]["baseline"][
            "tolerances"
        ]["radius_km"]
        pressure_reversal_percent = 100.0 * float(
            sensitivity["maximum_declared_source_pressure_reversal_fraction"]
        )
        return (
            f"- **{item['model_id']}** is conditionally reported, not seam-resolved. "
            f"{sensitivity['analysis_policy_rationale']} The pinned source reverses "
            f"pressure by {pressure_reversal_percent:.3f}% at its declared ordering "
            f"seam. The primary {baseline_label} convention gives "
            f"R1.4={float(baseline_1_4['radius_km']):.4f} km; {alternative_label} "
            f"gives {float(alternative_1_4['radius_km']):.4f} km, a "
            f"{float(span['span']):.4f} km conditional span. Both independently "
            f"satisfy the predeclared CompOSE catalogue radius tolerance of "
            f"{float(catalogue_tolerance):.2f} km. The span is a "
            "source-construction systematic--not TOV error, statistical "
            "uncertainty, or a Maxwell construction."
        )
    return (
        f"- **{item['model_id']}**: both declared diagnostic reductions complete "
        "and independently match the CompOSE catalogue; "
        f"alternative-minus-baseline R1.4={float(delta):+.6g} km is within the "
        "nominal delta classification threshold. Neither reduction resolves a "
        "physical transition."
    )


def _optional_reference_report_line(item: Mapping[str, Any]) -> str | None:
    finding = item["eos_mr_source_consistency"]
    classification = finding["classification"]
    if classification not in {
        "material_optional_reference_source_inconsistency",
        "material_optional_reference_discrepancy_unattributed",
    }:
        return None
    catalogue_delta = _report_number(
        finding["catalogue_minus_reference_radius_at_1_4_msun_km"]
    )
    calculated_delta = _report_number(
        finding["calculated_minus_reference_radius_at_1_4_msun_km"]
    )
    maximum = _report_magnitude(
        finding["maximum_absolute_fixed_mass_radius_residual_km"]
    )
    rms = _report_magnitude(finding["rms_fixed_mass_radius_residual_km"])
    diagnostics = (
        f"catalogue-minus-reference R1.4={catalogue_delta} km, "
        f"calculated-minus-reference R1.4={calculated_delta} km, "
        f"maximum fixed-mass |delta R|={maximum} km, and RMS fixed-mass "
        f"delta R={rms} km"
    )
    if classification == "material_optional_reference_discrepancy_unattributed":
        return (
            f"- **{item['model_id']}: MATERIAL OPTIONAL-REFERENCE DISCREPANCY "
            f"(UNATTRIBUTED).** The diagnostics report {diagnostics}. The complete "
            "corroborating exact R1.4 triangle required for source attribution is "
            "absent or non-corroborating, so no calculation, catalogue, or optional "
            "reference vertex is identified as the cause. The finding is "
            "explicitly non-gating."
        )
    return (
        f"- **{item['model_id']}: MATERIAL OPTIONAL-REFERENCE SOURCE "
        "INCONSISTENCY.** The toolkit TOV sequence independently passes the "
        "current CompOSE catalogue check, while the same entry's optional "
        f"`eos.mr` differs by {diagnostics}. `eos.mr` was never solver input and "
        "this finding "
        "does not fail calculation acceptance. CompOSE documents no convention "
        "that resolves the source-internal mismatch; cause undocumented, not "
        "solver failure."
    )


def _archive_metadata_report_line(
    item: Mapping[str, Any], finding: Mapping[str, Any]
) -> str:
    embedded = finding["declared_embedded_benchmark"]
    authoritative = finding["authoritative_compose_benchmark"]
    member = finding["source_member"]
    return (
        f"- **{item['model_id']}**: pinned `{member['name']}` "
        f"({member['bytes']} bytes; SHA-256 `{member['sha256']}`) declares "
        f"Mmax={float(embedded['maximum_mass_msun']):.2f} Msun, "
        f"R(Mmax)={float(embedded['radius_at_maximum_mass_km']):.2f} km, and "
        f"R1.4={float(embedded['radius_at_1_4_msun_km']):.2f} km, whereas the "
        f"registry-pinned current CompOSE page reports "
        f"{float(authoritative['maximum_mass_msun']):.2f} Msun, "
        f"{float(authoritative['radius_at_maximum_mass_km']):.2f} km, and "
        f"{float(authoritative['radius_at_1_4_msun_km']):.2f} km. "
        f"{finding['interpretation']} Cause: {finding['cause']}."
    )


def _report_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):+.6f}"


def _report_magnitude(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _causality_report_text(causal: Mapping[str, Any]) -> str:
    status = str(causal["status"])
    mass = causal.get("mass_msun")
    if status == "first_cs2_equals_one_threshold":
        return "threshold n/a" if mass is None else f"threshold {float(mass):.5f} Msun"
    if status == "entire_selected_barotrope_has_positive_cs2_at_or_below_one":
        return "full selected EoS causal"
    if status == "no_causal_prefix":
        return "no causal prefix"
    if status == "last_sample_before_nonpositive_sound_speed":
        return (
            "causal-prefix endpoint n/a"
            if mass is None
            else f"causal-prefix endpoint {float(mass):.5f} Msun"
        )
    return f"unrecognized status: {status}"


def _write_report(
    summaries: Sequence[Mapping[str, Any]],
    *,
    campaign_passed: bool,
    interpretive_status: str,
) -> None:
    lines = [
        "# Cold CompOSE comparison results",
        "",
        "All primary mass-radius curves below were calculated by the toolkit's TOV solver. Optional `eos.mr` files were used only after calculation as independent references.",
        "",
        f"Numerical campaign acceptance: **{'PASS' if campaign_passed else 'FAIL'}**. "
        f"Interpretive status: `{interpretive_status}`. Material interpretive "
        "findings remain explicitly non-gating.",
        "",
        "| Model | Sampled peak [Msun] | R(peak) [km] | R1.4 [km] | Causality check | Catalogue | Ordering seam | Literature convention |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for item in summaries:
        peak = item["metrics"]["sampled_peak"]
        at_1_4 = item["metrics"]["at_1_4_msun"]
        causal_text = _causality_report_text(item["causal_endpoint"])
        literature = item["literature_crosscheck"]
        sensitivity = item["ordering"]["sensitivity"]
        ordering_status = (
            "n/a"
            if sensitivity is None
            else (
                "MATERIAL conditional span"
                if sensitivity["classification"] == "material_source_seam_systematic"
                else sensitivity["classification"]
            )
        )
        radius_1_4_text = (
            "n/a" if at_1_4 is None else f"{float(at_1_4['radius_km']):.4f}"
        )
        lines.append(
            f"| {item['model_id']} | {peak['mass_msun']:.5f} | {peak['radius_km']:.4f} | "
            f"{radius_1_4_text} | "
            f"{causal_text} | "
            f"{'PASS' if item['compose_catalogue_crosscheck']['passed'] else 'FAIL'} | "
            f"{ordering_status} | "
            f"{literature['comparability']} ({literature['acceptance_status']}) |"
        )
    ordering_lines = [
        line
        for item in summaries
        if (line := _ordering_systematic_report_line(item)) is not None
    ]
    lines.extend(
        (
            "",
            "## Declared source-seam systematics",
            "",
            "Campaign acceptance here means reproducible calculation and external "
            "catalogue consistency under both declared reductions. It does not "
            "certify a unique crust-core radius or a physical seam construction.",
            "",
            *ordering_lines,
        )
    )
    optional_reference_lines = [
        line
        for item in summaries
        if (line := _optional_reference_report_line(item)) is not None
    ]
    lines.extend(
        (
            "",
            "## Optional eos.mr consistency diagnostics",
            "",
            "The 0.15 km threshold below classifies material diagnostic "
            "disagreement; it is not an uncertainty or an acceptance tolerance. "
            "A source inconsistency is assigned only when the independent toolkit "
            "calculation first passes the current CompOSE catalogue gate and both "
            "catalogue-minus-reference and exact calculated-minus-reference R1.4 "
            "differences strictly exceed that threshold. Other material diagnostics "
            "remain unattributed.",
            "",
            "| Model | Classification | Catalogue - eos.mr R1.4 [km] | TOV - eos.mr R1.4 [km] | Maximum fixed-mass abs(delta R) [km] | RMS fixed-mass delta R [km] |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for item in summaries:
        finding = item["eos_mr_source_consistency"]
        lines.append(
            f"| {item['model_id']} | {finding['classification']} | "
            f"{_report_number(finding['catalogue_minus_reference_radius_at_1_4_msun_km'])} | "
            f"{_report_number(finding['calculated_minus_reference_radius_at_1_4_msun_km'])} | "
            f"{_report_magnitude(finding['maximum_absolute_fixed_mass_radius_residual_km'])} | "
            f"{_report_magnitude(finding['rms_fixed_mass_radius_residual_km'])} |"
        )
    lines.extend(("", *optional_reference_lines))

    archive_metadata_lines = [
        _archive_metadata_report_line(item, finding)
        for item in summaries
        for finding in item["non_gating_findings"]["archive_metadata_provenance"]
    ]
    lines.extend(
        (
            "",
            "## Archive-metadata provenance findings",
            "",
            "These exact-member findings disclose stale upstream descriptive "
            "metadata. They are not solver inputs or calculation acceptance gates.",
            "",
            *(archive_metadata_lines or ["- None."]),
        )
    )
    lines.extend(
        (
            "",
            "## Closure residual diagnostics",
            "",
            "These maxima are source-table diagnostics only; they are not campaign acceptance gates or thermodynamic-closure certification.",
            "",
            "| Model | Maximum absolute normalized residual |",
            "|---|---:|",
        )
    )
    for item in summaries:
        maxima = item["closure_residual_diagnostics"][
            "maximum_absolute_normalized_residual"
        ]
        finite = [float(value) for value in maxima.values() if value is not None]
        value = "n/a" if not finite else f"{max(finite):.6g}"
        lines.append(f"| {item['model_id']} | {value} |")
    lines.extend(
        (
            "",
            "## Interpretation guardrails",
            "",
            "- 'Sampled peak' is the refined highest sampled background, not an analytical maximum-mass theorem.",
            "- The pre-peak central-density segment is a sampling description; no radial-stability result is inferred.",
            "- Radii stop at each selected table's lowest positive pressure and are not vacuum-surface radii.",
            "- The common positive-source-boundary check does not quantify the omitted P-to-zero surface layers.",
            "- BSk26 and APR hydrostatic peaks are reported separately from their numerically verified `c_s^2=1` thresholds.",
            "- Declared ordering reductions must both complete and independently match the predeclared catalogue tolerances; their mutual delta is classified and reported, not used as a tuned acceptance tolerance.",
            "- Literature values gate acceptance only when classified as like-for-like; contextual and provenance-only comparisons are still reported numerically when available.",
            "- `eos.mr` radius residuals use fixed masses below both sampled turning points; peak coordinates are reported separately.",
            "",
        )
    )
    (RESULTS_ROOT / "report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
