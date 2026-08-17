"""Cross-check calculated stars against catalogue and literature values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from settings import (
    BASE_CONFIG,
    CATALOGUE_MASS_ABSOLUTE_TOLERANCE_MSUN,
    CATALOGUE_MASS_RELATIVE_TOLERANCE,
    CATALOGUE_RADIUS_TOLERANCE_KM,
    CLOSURE_RESIDUAL_COLUMNS,
    SLY4_RADIUS_TOLERANCE_KM,
)

from neutron_star_eos import EosModel, StarResult


def _profile_stars(
    model: EosModel,
    metrics: Mapping[str, Any],
    *,
    validation_mode: str,
) -> dict[str, StarResult]:
    targets = {"sampled_peak": metrics["sampled_peak"]}
    if metrics["at_1_4_msun"] is not None:
        targets["1_4_msun"] = metrics["at_1_4_msun"]
    return {
        label: model.solve_star(
            float(values["central_pressure_mev_fm3"]),
            config=BASE_CONFIG,
            retain_profile=True,
            validation_mode=validation_mode,
        )
        for label, values in targets.items()
    }


def _catalogue_check(
    spec: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    benchmark = spec["compose_benchmark"]
    peak = metrics["sampled_peak"]
    at_1_4 = metrics["at_1_4_msun"]
    mass_tolerance = max(
        CATALOGUE_MASS_ABSOLUTE_TOLERANCE_MSUN,
        CATALOGUE_MASS_RELATIVE_TOLERANCE * float(benchmark["maximum_mass_msun"]),
    )
    radius_tolerance = (
        SLY4_RADIUS_TOLERANCE_KM
        if spec["slug"] == "sly4"
        else CATALOGUE_RADIUS_TOLERANCE_KM
    )
    delta_mass = float(peak["mass_msun"]) - float(benchmark["maximum_mass_msun"])
    delta_peak_radius = float(peak["radius_km"]) - float(
        benchmark["radius_at_maximum_mass_km"]
    )
    delta_1_4 = (
        None
        if at_1_4 is None
        else float(at_1_4["radius_km"]) - float(benchmark["radius_at_1_4_msun_km"])
    )
    passed = (
        abs(delta_mass) <= mass_tolerance
        and abs(delta_peak_radius) <= radius_tolerance
        and delta_1_4 is not None
        and abs(delta_1_4) <= radius_tolerance
    )
    return {
        "source": "current CompOSE catalogue page",
        "benchmark": dict(benchmark),
        "calculated_minus_benchmark": {
            "sampled_peak_mass_msun": delta_mass,
            "radius_at_sampled_peak_km": delta_peak_radius,
            "radius_at_1_4_msun_km": delta_1_4,
        },
        "tolerances": {
            "mass_msun": mass_tolerance,
            "radius_km": radius_tolerance,
        },
        "passed": bool(passed),
    }


def _literature_check(
    spec: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    assessment = spec["literature_assessment"]
    peak = metrics["sampled_peak"]
    at_1_4 = metrics["at_1_4_msun"]
    calculated = {
        "maximum_mass_msun": float(peak["mass_msun"]),
        "radius_at_maximum_mass_km": float(peak["radius_km"]),
        "radius_at_1_4_msun_km": (
            None if at_1_4 is None else float(at_1_4["radius_km"])
        ),
    }
    checks: dict[str, Any] = {}
    for name in (
        "maximum_mass_msun",
        "radius_at_maximum_mass_km",
        "radius_at_1_4_msun_km",
    ):
        literature_value = assessment.get(name)
        if literature_value is None:
            continue
        tolerance = (
            max(
                CATALOGUE_MASS_ABSOLUTE_TOLERANCE_MSUN,
                CATALOGUE_MASS_RELATIVE_TOLERANCE * float(literature_value),
            )
            if name == "maximum_mass_msun"
            else CATALOGUE_RADIUS_TOLERANCE_KM
        )
        calculated_value = calculated[name]
        delta = (
            None
            if calculated_value is None
            else calculated_value - float(literature_value)
        )
        checks[name] = {
            "calculated": calculated_value,
            "literature": float(literature_value),
            "calculated_minus_literature": delta,
            "absolute_tolerance": tolerance,
            "within_tolerance": delta is not None and abs(delta) <= tolerance,
        }
    comparability = str(assessment["comparability"])
    acceptance_required = comparability == "like_for_like"
    required_numeric_fields_available = all(
        assessment.get(name) is not None
        for name in (
            "maximum_mass_msun",
            "radius_at_maximum_mass_km",
            "radius_at_1_4_msun_km",
        )
    )
    numeric_passed = bool(checks) and all(
        bool(check["within_tolerance"]) for check in checks.values()
    )
    if acceptance_required:
        numeric_passed = numeric_passed and required_numeric_fields_available
    acceptance_gate_passed = bool(not acceptance_required or numeric_passed)
    return {
        "comparability": comparability,
        "source_label": assessment["source_label"],
        "source_url": assessment["source_url"],
        "notes": assessment["notes"],
        "numeric_checks": checks,
        "numeric_comparison_passed": numeric_passed if checks else None,
        "required_numeric_fields_available": required_numeric_fields_available,
        "acceptance_required": acceptance_required,
        "acceptance_gate_passed": acceptance_gate_passed,
        "acceptance_status": (
            "not_gated_contextual"
            if not acceptance_required
            else (
                "passed_like_for_like"
                if acceptance_gate_passed
                else "failed_like_for_like"
            )
        ),
        "interpretation": (
            "like-for-like numeric acceptance gate"
            if acceptance_required
            else "contextual or provenance-only comparison; not an acceptance gate"
        ),
    }


def _closure_diagnostics(view: Any) -> dict[str, Any]:
    native = view.series_for("native_thermodynamics")
    maxima: dict[str, float | None] = {}
    finite_counts: dict[str, int] = {}
    for name in CLOSURE_RESIDUAL_COLUMNS:
        if name not in native.column_names:
            maxima[name] = None
            finite_counts[name] = 0
            continue
        values = np.asarray(native.column(name), dtype=float)
        finite = np.abs(values[np.isfinite(values)])
        maxima[name] = None if not len(finite) else float(np.max(finite))
        finite_counts[name] = int(len(finite))
    return {
        "role": "diagnostic_only",
        "used_as_acceptance_gate": False,
        "maximum_absolute_normalized_residual": maxima,
        "finite_sample_count": finite_counts,
        "interpretation": (
            "reported without repairing the source table; campaign acceptance does "
            "not claim thermodynamic-closure certification"
        ),
    }
