"""Command-line orchestration for the complete pinned campaign."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from acceptance import (
    _campaign_interpretive_status,
    _optional_reference_campaign_findings,
)
from acquire import (
    DEFAULT_CONFIG,
    DEFAULT_RAW_ROOT,
    AcquisitionError,
    load_config,
    selected_models,
)
from campaign import _run_model
from campaign_io import write_json as _write_json
from campaign_io import write_rows as _write_rows
from figures import _save_comparison_plots
from files import _preflight_raw_inputs, _prepare_selected_outputs
from provenance import _manifest
from report import _write_report
from settings import DERIVED_ROOT, FIGURE_ROOT, RESULTS_ROOT, RUN_SCHEMA_VERSION
from summary import SUMMARY_FIELDS, _summary_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate and validate the pinned cold-CompOSE TOV campaign."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Run one registry slug; repeat to preserve a chosen order.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced central-density grids for development checks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config_path = arguments.config.expanduser().resolve()
        config = load_config(config_path)
        models = selected_models(config, arguments.models)
        required = tuple(config["campaign"]["required_archive_members"])
        raw_root = arguments.raw_root.expanduser().resolve()
        _preflight_raw_inputs(models, raw_root=raw_root, required_members=required)
    except AcquisitionError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    _prepare_selected_outputs(models)
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for index, spec in enumerate(models, start=1):
        print(
            f"[{index}/{len(models)}] calculating {spec['model_id']} ({spec['slug']})",
            flush=True,
        )
        try:
            summary = _run_model(
                spec,
                raw_root=raw_root,
                quick=arguments.quick,
                required_members=required,
            )
        except Exception as exc:
            _write_json(
                RESULTS_ROOT / "failure.json",
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "slug": spec["slug"],
                    "exception_type": type(exc).__name__,
                    "reason": str(exc),
                    "completed_models": [item["slug"] for item in summaries],
                },
            )
            raise
        summaries.append(summary)
        peak = summary["metrics"]["sampled_peak"]
        print(
            f"    sampled peak={peak['mass_msun']:.6f} Msun, "
            f"R={peak['radius_km']:.5f} km, "
            f"acceptance={'PASS' if summary['acceptance']['passed'] else 'FAIL'}",
            flush=True,
        )
    rows = _summary_rows(summaries)
    _write_rows(RESULTS_ROOT / "all_models_summary.csv", SUMMARY_FIELDS, rows)
    comparison_plots = _save_comparison_plots(summaries)
    model_acceptance_passed = all(item["acceptance"]["passed"] for item in summaries)
    campaign_gates = {
        "all_models_passed": model_acceptance_passed,
        "comparison_plot_coverage": bool(comparison_plots["required_coverage_passed"]),
    }
    campaign_passed = all(campaign_gates.values())
    material_seam_findings = {
        str(item["slug"]): {
            "model_id": item["model_id"],
            "classification": item["ordering"]["sensitivity"]["classification"],
            "conditional_radius_span_at_1_4_msun_km": item["ordering"]["sensitivity"][
                "conditional_radius_span_at_1_4_msun_km"
            ],
            "delta_within_nominal_tolerance": item["ordering"]["sensitivity"][
                "delta_within_nominal_tolerance"
            ],
            "acceptance_gate_passed": item["ordering"]["sensitivity"][
                "acceptance_gate_passed"
            ],
            "physical_transition_resolved": item["ordering"]["sensitivity"][
                "physical_transition_resolved"
            ],
            "interpretation": item["ordering"]["sensitivity"]["interpretation"],
        }
        for item in summaries
        if item["ordering"]["sensitivity"] is not None
        and item["ordering"]["sensitivity"]["classification"]
        == "material_source_seam_systematic"
    }
    optional_reference_findings = _optional_reference_campaign_findings(summaries)
    archive_metadata_findings = {
        str(item["slug"]): item["non_gating_findings"]["archive_metadata_provenance"]
        for item in summaries
        if item["non_gating_findings"]["archive_metadata_provenance"]
    }
    interpretive_status = _campaign_interpretive_status(
        summaries, campaign_passed=campaign_passed
    )
    _write_json(
        RESULTS_ROOT / "acceptance.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "models": {item["slug"]: item["acceptance"] for item in summaries},
            "comparison_plots": comparison_plots,
            "campaign_gates": campaign_gates,
            "non_gating_findings": {
                "material_source_seam_systematics": material_seam_findings,
                **optional_reference_findings,
                "archive_metadata_provenance": archive_metadata_findings,
            },
            "interpretive_status": interpretive_status,
            "passed": campaign_passed,
        },
    )
    _write_report(
        summaries,
        campaign_passed=campaign_passed,
        interpretive_status=interpretive_status,
    )
    (RESULTS_ROOT / "failure.json").unlink(missing_ok=True)
    _manifest(summaries, config_path, raw_root=raw_root)
    passed = campaign_passed
    print(
        f"campaign {'PASS' if passed else 'FAIL'}: {len(summaries)} models; "
        f"results in {RESULTS_ROOT}",
        flush=True,
    )
    return 0 if passed else 1
