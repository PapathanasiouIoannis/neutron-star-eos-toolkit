"""Run the pinned cold-CompOSE comparison campaign.

Every mass-radius curve produced here is calculated with the toolkit's TOV
solver.  Optional ``eos.mr`` tables are loaded only after each calculation and
are retained as independent reference data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import brentq

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from acquire import (  # type: ignore[import-not-found]
    DEFAULT_CONFIG,
    DEFAULT_RAW_ROOT,
    AcquisitionError,
    load_config,
    selected_models,
    sha256_file,
    verify_archive,
)

from neutron_star_eos import (
    EosInputError,
    EosModel,
    SequenceAttempt,
    SequenceResult,
    StarResult,
    StellarConfig,
    load_compose_mass_radius_reference,
    open_eos,
)
from neutron_star_eos import (
    __version__ as toolkit_version,
)
from neutron_star_eos.compose import ComposeEos, ComposeMassRadiusReference
from neutron_star_eos.plotting import (
    plot_compose_closure_residuals,
    plot_compose_cold_residuals,
    plot_compose_free_energy_closure_residuals,
    plot_composition,
    plot_mass_profile,
    plot_phase_codes,
    plot_pressure_energy,
    plot_sequence_status,
    plot_sound_speed_squared,
)
from neutron_star_eos.stellar import (
    FM3_M3,
    GRAVITY_CONVERSION,
    MEV_J,
    NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2,
    SOLAR_MASS_KG,
    SOLAR_MASS_LENGTH_KM,
    SPEED_OF_LIGHT_M_S,
    STELLAR_CONSTANT_AUTHORITY,
    STELLAR_CONSTANT_REFERENCE_URL,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DERIVED_ROOT = EXPERIMENT_ROOT / "data" / "derived"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
MANIFEST_PATH = EXPERIMENT_ROOT / "manifest.json"
RUN_SCHEMA_VERSION = "compose-comparison-run-v1"

BASE_CONFIG = StellarConfig(
    radius_max_km=60.0,
    ode_rtol=1.0e-10,
    ode_atol=1.0e-12,
    profile_points=500,
)
RETRY_CONFIG = replace(BASE_CONFIG, radius_max_km=120.0)
COMMON_BOUNDARY_DENSITY_FM3 = 1.0e-7
CENTRAL_DENSITY_FLOOR_FM3 = 0.18
CATALOGUE_MASS_ABSOLUTE_TOLERANCE_MSUN = 0.01
CATALOGUE_MASS_RELATIVE_TOLERANCE = 0.005
CATALOGUE_RADIUS_TOLERANCE_KM = 0.15
SLY4_RADIUS_TOLERANCE_KM = 0.25
CONVERGENCE_MASS_TOLERANCE_MSUN = 1.0e-5
CONVERGENCE_RADIUS_TOLERANCE_KM = 1.0e-3
SEAM_MASS_TOLERANCE_MSUN = 1.0e-3
SEAM_RADIUS_TOLERANCE_KM = 0.01
PRE_PEAK_MASS_DECREASE_TOLERANCE_MSUN = 1.0e-8
CAUSALITY_THRESHOLD_TOLERANCE = 1.0e-10
REFERENCE_PEAK_EXCLUSION_MARGIN_MSUN = 0.05
REFERENCE_FIXED_MASSES_MSUN = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)

_CLOSURE_RESIDUAL_COLUMNS = (
    "euler_normalized_residual",
    "first_law_normalized_residual",
    "gibbs_duhem_normalized_residual",
    "free_energy_pressure_normalized_residual",
    "free_energy_muB_normalized_residual",
)

COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7A5195",
    "#444444",
    "#882255",
)


@dataclass(frozen=True, slots=True)
class BranchData:
    pressure_mev_fm3: np.ndarray
    baryon_density_fm3: np.ndarray
    mass_msun: np.ndarray
    radius_km: np.ndarray
    peak_index: int
    pre_peak_mass_decrease_count: int


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        candidate = float(value)
        return candidate if math.isfinite(candidate) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    name: "" if row.get(name) is None else row.get(name)
                    for name in fieldnames
                }
            )
    temporary.replace(path)


def _write_columns(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    names = tuple(columns)
    lengths = {len(np.asarray(columns[name])) for name in names}
    if len(lengths) != 1:
        raise RuntimeError(f"unaligned columns for {path}")
    rows = (
        {
            name: (
                ""
                if not math.isfinite(float(np.asarray(columns[name])[index]))
                else f"{float(np.asarray(columns[name])[index]):.17g}"
            )
            for name in names
        }
        for index in range(next(iter(lengths)))
    )
    _write_rows(path, names, rows)


def _open_model(
    spec: Mapping[str, Any],
    archive: Path,
    *,
    ordering_policy: str,
    baryon_density_min_fm3: float | None = None,
    native_points: int = 3001,
) -> EosModel:
    return open_eos(
        archive,
        kind="compose",
        model_id=str(spec["model_id"]),
        source_url=str(spec["compose_page_url"]),
        matter=str(spec["matter"]),
        includes_leptons=bool(spec["includes_leptons"]),
        baryon_density_min_fm3=baryon_density_min_fm3,
        native_points=native_points,
        ordering_policy=ordering_policy,
    )


def _select_model(
    spec: Mapping[str, Any], archive: Path
) -> tuple[EosModel, EosModel, str, tuple[str, ...]]:
    strict = _open_model(spec, archive, ordering_policy="strict")
    if strict.barotrope is not None:
        if spec.get("ordering_analysis") is not None:
            raise EosInputError(
                f"{spec['model_id']} unexpectedly passed strict ordering despite a "
                "registry-declared seam analysis"
            )
        return strict, strict, "strict", ()
    ordering_value = spec.get("ordering_analysis")
    if not isinstance(ordering_value, Mapping):
        raise EosInputError(
            f"{spec['model_id']} failed strict ordering without a registry-declared "
            "seam analysis"
        )
    details = strict.report().to_dict()["details"]
    cold_slice = details["cold_slice"]
    actual_issues = cold_slice["pressure_ordering_issues"]
    expected_issues = ordering_value["expected_pressure_issues"]
    if len(actual_issues) != len(expected_issues):
        raise EosInputError(
            f"{spec['model_id']} pressure-issue count differs from the pinned registry"
        )
    for actual, expected in zip(actual_issues, expected_issues):
        for name in ("left_position", "right_position"):
            if int(actual[name]) != int(expected[name]):
                raise EosInputError(
                    f"{spec['model_id']} ordering issue {name} differs from registry"
                )
        for name in (
            "left_baryon_density_fm3",
            "right_baryon_density_fm3",
            "relative_change",
        ):
            if not math.isclose(
                float(actual[name]),
                float(expected[name]),
                rel_tol=2.0e-12,
                abs_tol=1.0e-15,
            ):
                raise EosInputError(
                    f"{spec['model_id']} ordering issue {name} differs from registry"
                )
    if cold_slice["energy_density_ordering_issues"]:
        raise EosInputError(
            f"{spec['model_id']} has an unexpected energy-density ordering issue"
        )
    policies = tuple(str(item) for item in ordering_value["sensitivity_policies"])
    analysis_policy = str(ordering_value["analysis_policy"])
    candidates = tuple(
        _open_model(spec, archive, ordering_policy=policy) for policy in policies
    )
    if any(candidate.barotrope is None for candidate in candidates):
        raise EosInputError(
            f"{spec['model_id']} remains unavailable under both explicit "
            "diagnostic ordering policies"
        )
    selected = candidates[policies.index(analysis_policy)]
    return strict, selected, analysis_policy, policies


def _compose_eos(model: EosModel) -> ComposeEos:
    if not isinstance(model.barotrope, ComposeEos):
        raise RuntimeError(f"{model.model_name} has no CompOSE stellar barotrope")
    return model.barotrope


def _validation_mode(eos: ComposeEos) -> str:
    return "strict" if eos.validate().passed else "background_diagnostic"


def _cs2_within_causal_threshold(value: float) -> bool:
    return bool(
        math.isfinite(value)
        and value > 0.0
        and value <= 1.0 + CAUSALITY_THRESHOLD_TOLERANCE
    )


def _failure_needs_radius_retry(attempt: SequenceAttempt) -> bool:
    return attempt.star is None and (
        attempt.reason_code == "radius_limit_reached"
        or (attempt.reason is not None and "radius limit" in attempt.reason.lower())
    )


def _merge_sequences(
    template: SequenceResult, sequences: Sequence[SequenceResult]
) -> SequenceResult:
    attempts_by_pressure: dict[float, SequenceAttempt] = {}
    for sequence in sequences:
        for attempt in sequence.attempts:
            previous = attempts_by_pressure.get(attempt.central_pressure_mev_fm3)
            if previous is None or (previous.star is None and attempt.star is not None):
                attempts_by_pressure[attempt.central_pressure_mev_fm3] = attempt
    attempts = tuple(attempts_by_pressure[key] for key in sorted(attempts_by_pressure))
    return replace(
        template,
        attempts=attempts,
        status="complete"
        if all(item.star is not None for item in attempts)
        else "partial",
    )


def _solve_pressures(
    model: EosModel,
    pressures: np.ndarray,
    *,
    validation_mode: str,
    config: StellarConfig = BASE_CONFIG,
) -> tuple[SequenceResult, dict[str, int]]:
    sequence = model.solve_sequence(
        pressures,
        config=config,
        validation_mode=validation_mode,
    )
    retry_candidates = tuple(
        item.central_pressure_mev_fm3
        for item in sequence.attempts
        if _failure_needs_radius_retry(item)
    )
    retry_solved = 0
    if retry_candidates:
        retry = model.solve_sequence(
            retry_candidates,
            config=RETRY_CONFIG,
            validation_mode=validation_mode,
        )
        retry_solved = len(retry.stars)
        sequence = _merge_sequences(sequence, (sequence, retry))
    return sequence, {
        "requested": len(pressures),
        "initial_failures": sum(item.star is None for item in sequence.attempts)
        + retry_solved,
        "radius_retry_candidates": len(retry_candidates),
        "radius_retry_solved": retry_solved,
        "remaining_failures": sum(item.star is None for item in sequence.attempts),
    }


def _adaptive_sequence(
    model: EosModel,
    *,
    validation_mode: str,
    quick: bool,
) -> tuple[SequenceResult, dict[str, Any], SequenceResult]:
    eos = _compose_eos(model)
    lower = max(
        CENTRAL_DENSITY_FLOOR_FM3,
        float(np.nextafter(eos.baryon_density_min_fm3, math.inf)),
    )
    upper = eos.baryon_density_max_fm3
    if not lower < upper:
        raise RuntimeError("selected central-density domain is empty")
    coarse_points = 51 if quick else 121
    refinement_points = 61 if quick else 161
    coarse_density = np.geomspace(lower, upper, coarse_points)
    coarse_pressure = np.asarray(
        eos.pressure_from_baryon_density(coarse_density), dtype=float
    )
    coarse, coarse_retry = _solve_pressures(
        model,
        coarse_pressure,
        validation_mode=validation_mode,
    )
    masses = np.asarray(
        [
            np.nan if attempt.star is None else attempt.star.mass_msun
            for attempt in coarse.attempts
        ],
        dtype=float,
    )
    if not np.any(np.isfinite(masses)):
        raise RuntimeError("coarse sequence produced no stellar backgrounds")
    peak = int(np.nanargmax(masses))
    left = max(0, peak - 3)
    right = min(len(coarse_density) - 1, peak + 3)
    refine_density = np.geomspace(
        coarse_density[left], coarse_density[right], refinement_points
    )
    refine_pressure = np.asarray(
        eos.pressure_from_baryon_density(refine_density), dtype=float
    )
    refined, refinement_retry = _solve_pressures(
        model,
        refine_pressure,
        validation_mode=validation_mode,
    )
    combined = _merge_sequences(coarse, (coarse, refined))
    return (
        combined,
        {
            "central_density_floor_fm3": lower,
            "central_density_ceiling_fm3": upper,
            "coarse_points": coarse_points,
            "refinement_points": refinement_points,
            "refinement_density_bracket_fm3": [
                float(coarse_density[left]),
                float(coarse_density[right]),
            ],
            "coarse_retry": coarse_retry,
            "refinement_retry": refinement_retry,
            "combined_attempts": len(combined.attempts),
            "combined_solved": len(combined.stars),
        },
        coarse,
    )


def _calculated_branch(sequence: SequenceResult, eos: ComposeEos) -> BranchData:
    solved = tuple(item for item in sequence.attempts if item.star is not None)
    if not solved:
        raise RuntimeError("sequence has no calculated backgrounds")
    pressure = np.asarray([item.central_pressure_mev_fm3 for item in solved])
    density = np.asarray(eos.baryon_density_from_pressure(pressure), dtype=float)
    mass = np.asarray([item.star.mass_msun for item in solved if item.star is not None])
    radius = np.asarray(
        [item.star.radius_km for item in solved if item.star is not None]
    )
    peak = int(np.argmax(mass))
    decreases = int(
        np.count_nonzero(
            np.diff(mass[: peak + 1]) < -PRE_PEAK_MASS_DECREASE_TOLERANCE_MSUN
        )
    )
    return BranchData(pressure, density, mass, radius, peak, decreases)


def _interpolate_branch(
    branch: BranchData, target_mass: float
) -> dict[str, float] | None:
    stop = branch.peak_index + 1
    mass = branch.mass_msun[:stop]
    if not float(np.min(mass)) <= target_mass <= float(np.max(mass)):
        return None
    order = np.argsort(mass)
    unique_mass, unique_indices = np.unique(mass[order], return_index=True)
    selected = order[unique_indices]
    return {
        "mass_msun": target_mass,
        "radius_km": float(
            np.interp(target_mass, unique_mass, branch.radius_km[:stop][selected])
        ),
        "central_pressure_mev_fm3": float(
            np.interp(
                target_mass, unique_mass, branch.pressure_mev_fm3[:stop][selected]
            )
        ),
        "central_baryon_density_fm3": float(
            np.interp(
                target_mass, unique_mass, branch.baryon_density_fm3[:stop][selected]
            )
        ),
    }


def _branch_metrics(branch: BranchData) -> dict[str, Any]:
    peak = branch.peak_index
    return {
        "algorithm": "global sampled peak; pre-peak increasing-central-density segment",
        "sampled_peak": {
            "mass_msun": float(branch.mass_msun[peak]),
            "radius_km": float(branch.radius_km[peak]),
            "central_pressure_mev_fm3": float(branch.pressure_mev_fm3[peak]),
            "central_baryon_density_fm3": float(branch.baryon_density_fm3[peak]),
        },
        "at_1_4_msun": _interpolate_branch(branch, 1.4),
        "sampled_points": len(branch.mass_msun),
        "pre_peak_points": peak + 1,
        "post_peak_points": len(branch.mass_msun) - peak - 1,
        "peak_bracketed_by_sampled_central_densities": (
            0 < peak < len(branch.mass_msun) - 1
        ),
        "peak_censored_at_upper_density_boundary": peak == len(branch.mass_msun) - 1,
        "pre_peak_mass_decrease_count": branch.pre_peak_mass_decrease_count,
        "pre_peak_mass_decrease_tolerance_msun": (
            PRE_PEAK_MASS_DECREASE_TOLERANCE_MSUN
        ),
    }


def _refine_target_mass(
    model: EosModel,
    branch: BranchData,
    target_mass_msun: float,
    *,
    validation_mode: str,
) -> dict[str, float] | None:
    """Locate a target mass on the sampled pre-peak central-density segment."""

    stop = branch.peak_index + 1
    mass = branch.mass_msun[:stop]
    if not float(np.min(mass)) <= target_mass_msun <= float(np.max(mass)):
        return None
    upper_index = int(np.searchsorted(mass, target_mass_msun))
    upper_index = min(max(upper_index, 1), stop - 1)
    lower_pressure = float(branch.pressure_mev_fm3[upper_index - 1])
    upper_pressure = float(branch.pressure_mev_fm3[upper_index])
    cache: dict[float, StarResult] = {}

    def residual(pressure: float) -> float:
        star = model.solve_star(
            pressure,
            config=BASE_CONFIG,
            validation_mode=validation_mode,
        )
        cache[pressure] = star
        return star.mass_msun - target_mass_msun

    pressure = brentq(
        residual,
        lower_pressure,
        upper_pressure,
        xtol=max(1.0e-12, lower_pressure * 1.0e-11),
        rtol=1.0e-11,
    )
    star = cache.get(pressure)
    if star is None:
        star = model.solve_star(
            pressure,
            config=BASE_CONFIG,
            validation_mode=validation_mode,
        )
    eos = _compose_eos(model)
    return {
        "mass_msun": star.mass_msun,
        "radius_km": star.radius_km,
        "central_pressure_mev_fm3": pressure,
        "central_baryon_density_fm3": float(eos.baryon_density_from_pressure(pressure)),
    }


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


def _sequence_rows(sequence: SequenceResult, eos: ComposeEos) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in sequence.attempts:
        pressure = attempt.central_pressure_mev_fm3
        density = float(eos.baryon_density_from_pressure(pressure))
        star = attempt.star
        rows.append(
            {
                "central_pressure_mev_fm3": f"{pressure:.17g}",
                "central_baryon_density_fm3": f"{density:.17g}",
                "status": attempt.status,
                "reason_code": attempt.reason_code,
                "reason": attempt.reason,
                "central_energy_density_mev_fm3": None
                if star is None
                else f"{star.central_energy_density_mev_fm3:.17g}",
                "central_sound_speed_squared": None
                if star is None
                else f"{star.central_sound_speed_squared:.17g}",
                "mass_msun": None if star is None else f"{star.mass_msun:.17g}",
                "radius_km": None if star is None else f"{star.radius_km:.17g}",
                "boundary_pressure_mev_fm3": None
                if star is None
                else f"{star.boundary_pressure_mev_fm3:.17g}",
                "boundary_energy_density_mev_fm3": None
                if star is None
                else f"{star.boundary_energy_density_mev_fm3:.17g}",
                "validation_mode": sequence.eos_validation_mode,
                "validation_status": sequence.eos_validation_status,
            }
        )
    return rows


SEQUENCE_FIELDS = (
    "central_pressure_mev_fm3",
    "central_baryon_density_fm3",
    "status",
    "reason_code",
    "reason",
    "central_energy_density_mev_fm3",
    "central_sound_speed_squared",
    "mass_msun",
    "radius_km",
    "boundary_pressure_mev_fm3",
    "boundary_energy_density_mev_fm3",
    "validation_mode",
    "validation_status",
)


def _save_thermodynamics(model: EosModel, directory: Path) -> Any:
    view = model.thermodynamics(curve_points=3001)
    filenames = {
        "native_thermodynamics": "native_thermodynamics.csv",
        "source_nodes": "barotrope_nodes.csv",
        "continuous_barotrope": "continuous_barotrope.csv",
    }
    metadata: dict[str, Any] = {"model": model.model_name, "series": {}}
    for series in view.series:
        _write_columns(directory / filenames[series.role], series.columns)
        metadata["series"][series.role] = {
            "label": series.label,
            "rows": series.rows,
            "columns": list(series.column_names),
            "units": dict(series.units),
            "descriptions": dict(series.descriptions),
            "diagnostic_codes": list(series.diagnostic_codes),
            "metadata": dict(series.metadata or {}),
        }
    _write_json(directory / "thermodynamics_metadata.json", metadata)
    return view


def _causal_endpoint(
    model: EosModel,
    *,
    validation_mode: str,
) -> dict[str, Any]:
    eos = _compose_eos(model)
    density = np.geomspace(
        eos.baryon_density_min_fm3, eos.baryon_density_max_fm3, 20001
    )
    pressure = np.asarray(eos.pressure_from_baryon_density(density), dtype=float)
    epsilon = np.asarray(eos.energy_density_from_pressure(pressure), dtype=float)
    cs2 = np.asarray(eos.sound_speed_squared_from_energy_density(epsilon), dtype=float)
    valid = np.isfinite(cs2) & (cs2 > 0.0) & (cs2 <= 1.0)
    cumulative = np.logical_and.accumulate(valid)
    if bool(cumulative[-1]):
        endpoint_density = eos.baryon_density_max_fm3
        status = "entire_selected_barotrope_has_positive_cs2_at_or_below_one"
    else:
        first_invalid = int(np.flatnonzero(~cumulative)[0])
        if first_invalid == 0:
            return {
                "status": "no_causal_prefix",
                "sampled_cs2_min": float(np.min(cs2)),
                "sampled_cs2_max": float(np.max(cs2)),
                "sound_speed_threshold_value": 1.0,
                "sound_speed_threshold_tolerance": CAUSALITY_THRESHOLD_TOLERANCE,
                "sound_speed_within_threshold_tolerance": False,
            }
        left = float(density[first_invalid - 1])
        right = float(density[first_invalid])

        def condition(log_density: float) -> float:
            n_b = math.exp(log_density)
            p = float(eos.pressure_from_baryon_density(n_b))
            e = float(eos.energy_density_from_pressure(p))
            return float(eos.sound_speed_squared_from_energy_density(e)) - 1.0

        if cs2[first_invalid] > 1.0 and cs2[first_invalid - 1] <= 1.0:
            endpoint_density = math.exp(
                brentq(condition, math.log(left), math.log(right))
            )
            status = "first_cs2_equals_one_threshold"
        else:
            endpoint_density = left
            status = "last_sample_before_nonpositive_sound_speed"
    endpoint_pressure = float(eos.pressure_from_baryon_density(endpoint_density))
    endpoint_energy_density = float(eos.energy_density_from_pressure(endpoint_pressure))
    endpoint_cs2 = float(
        eos.sound_speed_squared_from_energy_density(endpoint_energy_density)
    )
    within_threshold = _cs2_within_causal_threshold(endpoint_cs2)
    star = model.solve_star(
        endpoint_pressure,
        config=BASE_CONFIG,
        validation_mode=validation_mode,
    )
    return {
        "status": status,
        "central_baryon_density_fm3": endpoint_density,
        "central_pressure_mev_fm3": endpoint_pressure,
        "mass_msun": star.mass_msun,
        "radius_km": star.radius_km,
        "central_sound_speed_squared": endpoint_cs2,
        "solver_reported_central_sound_speed_squared": (
            star.central_sound_speed_squared
        ),
        "sound_speed_threshold_value": 1.0,
        "sound_speed_threshold_tolerance": CAUSALITY_THRESHOLD_TOLERANCE,
        "sound_speed_within_threshold_tolerance": within_threshold,
        "sampled_cs2_min": float(np.min(cs2)),
        "sampled_cs2_max": float(np.max(cs2)),
    }


def _convergence_rows(
    model: EosModel,
    metrics: Mapping[str, Any],
    *,
    validation_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    peak = metrics["sampled_peak"]
    at_1_4 = metrics["at_1_4_msun"]
    targets = {"sampled_peak": float(peak["central_pressure_mev_fm3"])}
    if at_1_4 is not None:
        targets["interpolated_1_4_msun"] = float(at_1_4["central_pressure_mev_fm3"])
    configurations = {
        "loose": replace(BASE_CONFIG, ode_rtol=1.0e-8, ode_atol=1.0e-10),
        "production": BASE_CONFIG,
        "tight": replace(BASE_CONFIG, ode_rtol=3.0e-11, ode_atol=3.0e-13),
    }
    rows: list[dict[str, Any]] = []
    results: dict[tuple[str, str], StarResult] = {}
    pressures = np.asarray(sorted(set(targets.values())), dtype=float)
    target_for_pressure = {value: name for name, value in targets.items()}
    for label, config in configurations.items():
        sequence = model.solve_sequence(
            pressures,
            config=config,
            validation_mode=validation_mode,
        )
        for attempt in sequence.attempts:
            if attempt.star is None:
                raise RuntimeError(
                    f"convergence solve failed: {model.model_name} {label} {attempt.reason}"
                )
            target = target_for_pressure[attempt.central_pressure_mev_fm3]
            results[(target, label)] = attempt.star
            rows.append(
                {
                    "target": target,
                    "configuration": label,
                    "ode_rtol": f"{config.ode_rtol:.17g}",
                    "ode_atol": f"{config.ode_atol:.17g}",
                    "mass_msun": f"{attempt.star.mass_msun:.17g}",
                    "radius_km": f"{attempt.star.radius_km:.17g}",
                }
            )
    checks: dict[str, Any] = {}
    for target in targets:
        production = results[(target, "production")]
        tight = results[(target, "tight")]
        delta_mass = abs(production.mass_msun - tight.mass_msun)
        delta_radius = abs(production.radius_km - tight.radius_km)
        checks[target] = {
            "production_minus_tight_absolute_mass_msun": delta_mass,
            "production_minus_tight_absolute_radius_km": delta_radius,
            "mass_tolerance_msun": CONVERGENCE_MASS_TOLERANCE_MSUN,
            "radius_tolerance_km": CONVERGENCE_RADIUS_TOLERANCE_KM,
            "passed": delta_mass <= CONVERGENCE_MASS_TOLERANCE_MSUN
            and delta_radius <= CONVERGENCE_RADIUS_TOLERANCE_KM,
        }
    return rows, checks


def _positive_source_boundary_sensitivity(
    spec: Mapping[str, Any],
    archive: Path,
    ordering_policy: str,
    branch: BranchData,
    full_metrics: Mapping[str, Any],
    full_eos: ComposeEos,
) -> dict[str, Any]:
    at_1_4 = full_metrics["at_1_4_msun"]
    if at_1_4 is None:
        return {"status": "not_available_no_1_4_msun_bracket"}
    common = _open_model(
        spec,
        archive,
        ordering_policy=ordering_policy,
        baryon_density_min_fm3=COMMON_BOUNDARY_DENSITY_FM3,
        native_points=1001,
    )
    eos = _compose_eos(common)
    if (
        eos.baryon_density_min_fm3 == full_eos.baryon_density_min_fm3
        and eos.pressure_min_mev_fm3 == full_eos.pressure_min_mev_fm3
    ):
        return {
            "status": "identical_positive_source_boundary",
            "requested_minimum_baryon_density_fm3": COMMON_BOUNDARY_DENSITY_FM3,
            "actual_minimum_baryon_density_fm3": eos.baryon_density_min_fm3,
            "actual_boundary_pressure_mev_fm3": eos.pressure_min_mev_fm3,
            "radius_at_1_4_msun_km": float(at_1_4["radius_km"]),
            "mass_at_interpolated_target_msun": float(at_1_4["mass_msun"]),
            "central_pressure_mev_fm3": float(at_1_4["central_pressure_mev_fm3"]),
            "full_source_minus_common_positive_source_boundary_radius_km": 0.0,
            "boundary_is_vacuum_surface": False,
            "interpretation": (
                "positive-source-boundary sensitivity only; omitted P-to-zero "
                "surface layers are not measured"
            ),
        }
    mode = _validation_mode(eos)
    full_stop = branch.peak_index + 1
    masses = branch.mass_msun[:full_stop]
    index = int(np.searchsorted(masses, 1.4))
    index = min(max(index, 1), full_stop - 1)
    lower_p = float(branch.pressure_mev_fm3[index - 1])
    upper_p = float(branch.pressure_mev_fm3[index])
    cache: dict[float, StarResult] = {}

    def residual(pressure: float) -> float:
        star = common.solve_star(
            pressure,
            config=BASE_CONFIG,
            validation_mode=mode,
        )
        cache[pressure] = star
        return star.mass_msun - 1.4

    target_pressure = brentq(
        residual,
        lower_p,
        upper_p,
        xtol=max(1.0e-12, lower_p * 1.0e-11),
        rtol=1.0e-11,
    )
    star = cache.get(target_pressure)
    if star is None:
        star = common.solve_star(
            target_pressure,
            config=BASE_CONFIG,
            validation_mode=mode,
        )
    return {
        "status": "calculated",
        "requested_minimum_baryon_density_fm3": COMMON_BOUNDARY_DENSITY_FM3,
        "actual_minimum_baryon_density_fm3": eos.baryon_density_min_fm3,
        "actual_boundary_pressure_mev_fm3": eos.pressure_min_mev_fm3,
        "radius_at_1_4_msun_km": star.radius_km,
        "mass_at_interpolated_target_msun": star.mass_msun,
        "central_pressure_mev_fm3": target_pressure,
        "full_source_minus_common_positive_source_boundary_radius_km": (
            float(at_1_4["radius_km"]) - star.radius_km
        ),
        "boundary_is_vacuum_surface": False,
        "interpretation": (
            "positive-source-boundary sensitivity only; omitted P-to-zero "
            "surface layers are not measured"
        ),
    }


def _clear_generated_tree(
    directory: Path,
    *,
    root: Path,
    allowed_suffixes: frozenset[str],
) -> None:
    """Remove one generated subtree only after validating its exact scope."""

    if not directory.exists():
        return
    resolved_root = root.resolve()
    resolved_directory = directory.resolve()
    if resolved_directory == resolved_root or not resolved_directory.is_relative_to(
        resolved_root
    ):
        raise RuntimeError(f"refusing unsafe generated-output cleanup: {directory}")

    def expected_generated_file(path: Path) -> bool:
        if path.suffix.lower() in allowed_suffixes:
            return True
        return (
            path.name.startswith(".")
            and path.name.endswith(".tmp")
            and Path(path.name[:-4]).suffix.lower() in allowed_suffixes
        )

    unexpected = [
        path
        for path in directory.rglob("*")
        if path.is_file() and not expected_generated_file(path)
    ]
    if unexpected:
        listed = ", ".join(str(path) for path in unexpected[:3])
        raise RuntimeError(
            f"refusing to clean {directory}; unexpected file type(s): {listed}"
        )
    shutil.rmtree(resolved_directory)


def _prepare_selected_outputs(models: Sequence[Mapping[str, Any]]) -> None:
    """Clear only selected generated outputs so interrupted runs cannot leak stale data."""

    for spec in models:
        slug = str(spec["slug"])
        _clear_generated_tree(
            DERIVED_ROOT / slug,
            root=DERIVED_ROOT,
            allowed_suffixes=frozenset({".csv", ".json"}),
        )
        _clear_generated_tree(
            FIGURE_ROOT / slug,
            root=FIGURE_ROOT,
            allowed_suffixes=frozenset({".png"}),
        )
    _clear_generated_tree(
        FIGURE_ROOT / "comparison",
        root=FIGURE_ROOT,
        allowed_suffixes=frozenset({".png"}),
    )
    for path in (
        RESULTS_ROOT / "all_models_summary.csv",
        RESULTS_ROOT / "acceptance.json",
        RESULTS_ROOT / "report.md",
        RESULTS_ROOT / "failure.json",
        MANIFEST_PATH,
    ):
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"refusing to replace non-file output: {path}")
            path.unlink()


def _preflight_raw_inputs(
    models: Sequence[Mapping[str, Any]],
    *,
    raw_root: Path,
    required_members: Sequence[str],
) -> None:
    """Verify all selected archives and sidecars before replacing prior outputs."""

    for spec in models:
        slug = str(spec["slug"])
        archive = raw_root / slug / str(spec["archive"]["filename"])
        verification = verify_archive(archive, spec, required_members=required_members)
        sidecar = archive.parent / "download.json"
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_hash = payload["verification"]["sha256"]
            sidecar_bytes = payload["verification"]["bytes"]
            sidecar_filename = payload["archive"]["local_filename"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AcquisitionError(
                f"{slug} acquisition sidecar is missing or malformed; run acquire.py"
            ) from exc
        if (
            sidecar_hash != verification["sha256"]
            or sidecar_bytes != verification["bytes"]
            or sidecar_filename != archive.name
        ):
            raise AcquisitionError(
                f"{slug} acquisition sidecar does not describe the pinned archive; "
                "rerun acquire.py"
            )


def _save_ax(ax: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        ax.figure.savefig(
            temporary,
            format="png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        temporary.replace(path)
    finally:
        plt.close(ax.figure)
        temporary.unlink(missing_ok=True)


def _plot_style() -> dict[str, Any]:
    return {
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titleweight": "bold",
    }


def _save_model_plots(
    model: EosModel,
    view: Any,
    sequence: SequenceResult,
    branch: BranchData,
    metrics: Mapping[str, Any],
    causal: Mapping[str, Any],
    reference: ComposeMassRadiusReference | None,
    figure_directory: Path,
    profile_stars: Mapping[str, StarResult],
) -> dict[str, Any]:
    created: list[str] = []
    required: list[str] = []
    skipped: dict[str, str] = {}

    def save(name: str, factory: Any) -> None:
        filename = f"{name}.png"
        required.append(filename)
        try:
            ax = factory()
        except (EosInputError, ValueError) as exc:
            skipped[name] = str(exc)
        else:
            _save_ax(ax, figure_directory / filename)
            created.append(filename)

    with plt.rc_context(_plot_style()):
        save(
            "pressure_energy",
            lambda: plot_pressure_energy(
                model, curve_points=3001, show_source_nodes=False
            ),
        )
        save(
            "pressure_energy_source_nodes",
            lambda: plot_pressure_energy(
                model, curve_points=3001, show_stellar_barotrope=False
            ),
        )
        save(
            "sound_speed_squared",
            lambda: plot_sound_speed_squared(
                model, curve_points=3001, include_stellar_barotrope=True
            ),
        )
        save(
            "closure_residuals",
            lambda: plot_compose_closure_residuals(
                model, curve_points=3001, include_free_energy=False
            ),
        )
        save(
            "free_energy_closure_residuals",
            lambda: plot_compose_free_energy_closure_residuals(
                model, curve_points=3001
            ),
        )
        save(
            "cold_condition_residuals",
            lambda: plot_compose_cold_residuals(model, curve_points=3001),
        )
        native = view.series_for("native_thermodynamics")
        abundance = [
            name
            for name in native.column_names
            if (
                name.startswith("composition_pair_")
                or (name.startswith("composition_quadruple_") and name.endswith("_Yav"))
            )
            and not name.endswith("_available")
        ]
        for group_index in range(0, len(abundance), 8):
            names = abundance[group_index : group_index + 8]
            save(
                f"composition_abundances_{group_index // 8 + 1:02d}",
                lambda names=names: plot_composition(model, quantities=names),
            )
        nuclear = [
            name
            for name in native.column_names
            if name.startswith("composition_quadruple_")
            and name.rsplit("_", 1)[-1] in {"Aav", "Zav", "Nav"}
            and not name.endswith("_available")
        ]
        for group_index in range(0, len(nuclear), 6):
            names = nuclear[group_index : group_index + 6]
            save(
                f"composition_nuclear_characteristics_{group_index // 6 + 1:02d}",
                lambda names=names: plot_composition(model, quantities=names),
            )
        if "phase_code" in native.column_names:
            save("phase_codes", lambda: plot_phase_codes(model))
        save("sequence_status", lambda: plot_sequence_status(sequence))
        for label, star in profile_stars.items():
            save(f"mass_profile_{label}", lambda star=star: plot_mass_profile(star))

        stop = branch.peak_index + 1
        figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.plot(
            branch.radius_km,
            branch.mass_msun,
            color=COLORS[0],
            linewidth=1.8,
            alpha=0.45,
            label="all calculated backgrounds",
        )
        ax.plot(
            branch.radius_km[:stop],
            branch.mass_msun[:stop],
            color=COLORS[0],
            linewidth=2.4,
            label="sampled pre-peak central-density segment",
        )
        peak = metrics["sampled_peak"]
        ax.scatter(
            [peak["radius_km"]],
            [peak["mass_msun"]],
            color=COLORS[1],
            marker="D",
            s=48,
            label="sampled hydrostatic peak",
            zorder=4,
        )
        if causal.get("mass_msun") is not None:
            ax.scatter(
                [causal["radius_km"]],
                [causal["mass_msun"]],
                color=COLORS[2],
                marker="^",
                s=52,
                label=r"positive-$c_s^2$, $c_s^2\leq1$ endpoint",
                zorder=4,
            )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title(model.model_name)
        ax.legend(loc="best")
        _save_ax(ax, figure_directory / "calculated_mass_radius.png")
        created.append("calculated_mass_radius.png")
        required.append("calculated_mass_radius.png")

        figure, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
        ax.plot(
            branch.baryon_density_fm3,
            branch.mass_msun,
            color=COLORS[0],
            linewidth=2.0,
        )
        ax.axvline(
            float(peak["central_baryon_density_fm3"]),
            color=COLORS[1],
            linestyle="--",
            label="sampled peak",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"Central baryon density $n_{B,c}$ [fm$^{-3}$]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title(f"{model.model_name}: central-density sequence")
        ax.legend(loc="best")
        _save_ax(ax, figure_directory / "mass_central_density.png")
        created.append("mass_central_density.png")
        required.append("mass_central_density.png")

        if reference is not None:
            ref_mass, ref_radius, _selection = _reference_selected_peak_side(reference)
            figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
            ax.scatter(
                reference.radius_km,
                reference.mass_msun,
                color=COLORS[4],
                s=18,
                label="source-order eos.mr points",
            )
            ax.set_xlabel("CompOSE reference radius [km]")
            ax.set_ylabel(r"CompOSE reference mass [$M_\odot$]")
            ax.set_title(f"{model.model_name}: independent eos.mr reference")
            ax.legend(loc="best")
            _save_ax(ax, figure_directory / "reference_mass_radius.png")
            created.append("reference_mass_radius.png")
            required.append("reference_mass_radius.png")

            figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
            ax.plot(
                branch.radius_km[:stop],
                branch.mass_msun[:stop],
                color=COLORS[0],
                linewidth=2.2,
                label="toolkit TOV calculation",
            )
            ax.scatter(
                ref_radius,
                ref_mass,
                color=COLORS[4],
                marker="o",
                facecolors="none",
                s=26,
                label="CompOSE eos.mr reference",
            )
            ax.set_xlabel("Radius [km]")
            ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
            ax.set_title(f"{model.model_name}: calculated vs independent reference")
            ax.legend(loc="best")
            _save_ax(ax, figure_directory / "calculated_vs_reference_mass_radius.png")
            created.append("calculated_vs_reference_mass_radius.png")
            required.append("calculated_vs_reference_mass_radius.png")

            comparison = _comparison_to_reference(branch, reference)
            fixed = comparison["fixed_mass_comparisons"]
            sample_mass = np.asarray([row["mass_msun"] for row in fixed], dtype=float)
            residual = np.asarray(
                [row["calculated_minus_reference_radius_km"] for row in fixed],
                dtype=float,
            )
            figure, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
            ax.plot(
                sample_mass,
                residual,
                color=COLORS[1],
                marker="o",
                linewidth=1.8,
            )
            ax.axhline(0.0, color="#555555", linewidth=1.0)
            ax.set_xlabel(r"Gravitational mass [$M_\odot$]")
            ax.set_ylabel(r"$R_{\rm TOV}-R_{\tt eos.mr}$ [km]")
            ax.set_title(f"{model.model_name}: fixed-mass reference residuals")
            _save_ax(ax, figure_directory / "reference_radius_residual.png")
            created.append("reference_radius_residual.png")
            required.append("reference_radius_residual.png")
    missing = sorted(set(required) - set(created))
    return {
        "created": created,
        "skipped": skipped,
        "required": required,
        "missing_required": missing,
        "required_coverage_passed": not missing,
    }


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
        "source": "current CompOSE catalogue/data sheet",
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
    for name in _CLOSURE_RESIDUAL_COLUMNS:
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


def _ordering_sensitivity(
    spec: Mapping[str, Any],
    archive: Path,
    baseline_policy: str,
    policies: Sequence[str],
    baseline_metrics: Mapping[str, Any],
    baseline_branch: BranchData,
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
        derived_directory / "sequence_ordering_keep_later.csv",
        SEQUENCE_FIELDS,
        _sequence_rows(sequence, _compose_eos(alternative)),
    )
    _write_json(
        derived_directory / "sequence_ordering_keep_later.json", sequence.to_dict()
    )
    baseline_peak = baseline_metrics["sampled_peak"]
    alternate_peak = metrics["sampled_peak"]
    baseline_1_4 = baseline_metrics["at_1_4_msun"]
    alternate_1_4 = metrics["at_1_4_msun"]
    delta_mass = float(alternate_peak["mass_msun"]) - float(baseline_peak["mass_msun"])
    delta_r14 = (
        None
        if baseline_1_4 is None or alternate_1_4 is None
        else float(alternate_1_4["radius_km"]) - float(baseline_1_4["radius_km"])
    )
    remaining_failures = sum(item.star is None for item in sequence.attempts)
    passed = (
        abs(delta_mass) <= SEAM_MASS_TOLERANCE_MSUN
        and delta_r14 is not None
        and abs(delta_r14) <= SEAM_RADIUS_TOLERANCE_KM
        and remaining_failures == 0
        and bool(metrics["peak_bracketed_by_sampled_central_densities"])
        and int(metrics["pre_peak_mass_decrease_count"]) == 0
    )
    with plt.rc_context(_plot_style()):
        figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.plot(
            baseline_branch.radius_km[: baseline_branch.peak_index + 1],
            baseline_branch.mass_msun[: baseline_branch.peak_index + 1],
            color=COLORS[0],
            linewidth=2.2,
            label="keep-first diagnostic reduction",
        )
        ax.plot(
            branch.radius_km[: branch.peak_index + 1],
            branch.mass_msun[: branch.peak_index + 1],
            color=COLORS[1],
            linestyle="--",
            linewidth=2.0,
            label="keep-later diagnostic reduction",
        )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title(f"{spec['model_id']}: ordering-seam sensitivity")
        ax.legend(loc="best")
        _save_ax(ax, figure_directory / "ordering_keep_later_mass_radius.png")
    return {
        "baseline_policy": baseline_policy,
        "alternative_policy": policy,
        "alternative_sampling": sampling,
        "alternative_metrics": metrics,
        "remaining_sequence_failures": remaining_failures,
        "alternative_minus_baseline": {
            "sampled_peak_mass_msun": delta_mass,
            "radius_at_1_4_msun_km": delta_r14,
        },
        "tolerances": {
            "sampled_peak_mass_msun": SEAM_MASS_TOLERANCE_MSUN,
            "radius_at_1_4_msun_km": SEAM_RADIUS_TOLERANCE_KM,
        },
        "plot_filename": "ordering_keep_later_mass_radius.png",
        "passed": bool(passed),
    }


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
    ordering_sensitivity = _ordering_sensitivity(
        spec,
        archive,
        ordering_policy,
        sensitivity_policies,
        metrics,
        branch,
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
    remaining_sequence_failures = sum(item.star is None for item in sequence.attempts)
    acceptance = {
        "catalogue": bool(catalogue["passed"]),
        "convention_classified_literature": bool(literature["acceptance_gate_passed"]),
        "ode_convergence": all(item["passed"] for item in convergence.values()),
        "ordering_sensitivity": (
            True
            if ordering_sensitivity is None
            else bool(ordering_sensitivity["passed"])
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
        "eos_mr_fixed_mass_coverage": (
            reference_comparison is None
            or int(reference_comparison["comparison_points"]) > 0
        ),
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
        "primary_citations": spec["primary_citations"],
        "analysis_notes": spec["analysis_notes"],
        "plots": plots,
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


def _save_comparison_plots(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparison = FIGURE_ROOT / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    required = [
        "all_calculated_mass_radius.png",
        "catalogue_peak_mass_crosscheck.png",
        "catalogue_peak_radius_crosscheck.png",
        "catalogue_1_4_radius_crosscheck.png",
    ]
    created: list[str] = []
    with plt.rc_context(_plot_style()):
        figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for index, summary in enumerate(summaries):
            path = DERIVED_ROOT / str(summary["slug"]) / "sequence.csv"
            data = np.genfromtxt(
                path, delimiter=",", names=True, dtype=None, encoding="utf-8"
            )
            solved = np.asarray(data["status"] == "solved")
            mass = np.asarray(data["mass_msun"][solved], dtype=float)
            radius = np.asarray(data["radius_km"][solved], dtype=float)
            peak = int(np.argmax(mass))
            ax.plot(
                radius[: peak + 1],
                mass[: peak + 1],
                color=COLORS[index % len(COLORS)],
                linewidth=2.0,
                label=str(summary["model_id"]),
            )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title("Calculated cold-CompOSE mass-radius comparison")
        ax.legend(loc="best", ncols=2)
        _save_ax(ax, comparison / "all_calculated_mass_radius.png")
        created.append("all_calculated_mass_radius.png")

        for key, label, filename in (
            (
                "sampled_peak_mass_msun",
                r"Sampled peak mass [$M_\odot$]",
                "catalogue_peak_mass_crosscheck.png",
            ),
            (
                "radius_at_sampled_peak_km",
                "Radius at sampled peak [km]",
                "catalogue_peak_radius_crosscheck.png",
            ),
            (
                "radius_at_1_4_msun_km",
                r"Radius at $1.4\,M_\odot$ [km]",
                "catalogue_1_4_radius_crosscheck.png",
            ),
        ):
            slugs = [str(item["slug"]) for item in summaries]
            values = [
                item["compose_catalogue_crosscheck"]["calculated_minus_benchmark"][key]
                for item in summaries
            ]
            deltas = [np.nan if value is None else float(value) for value in values]
            figure, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
            ax.bar(
                slugs,
                deltas,
                color=[COLORS[i % len(COLORS)] for i in range(len(slugs))],
            )
            ax.axhline(0.0, color="#333333", linewidth=1.0)
            ax.set_ylabel(f"Calculated minus catalogue: {label}")
            ax.set_title("CompOSE catalogue cross-check")
            ax.tick_params(axis="x", rotation=35)
            _save_ax(ax, comparison / filename)
            created.append(filename)
    missing = sorted(set(required) - set(created))
    return {
        "created": created,
        "required": required,
        "missing_required": missing,
        "required_coverage_passed": not missing,
    }


def _summary_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summaries:
        peak = item["metrics"]["sampled_peak"]
        at_1_4 = item["metrics"]["at_1_4_msun"]
        causal = item["causal_endpoint"]
        catalogue = item["compose_catalogue_crosscheck"]
        literature = item["literature_crosscheck"]
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
    "causal_endpoint_mass_msun",
    "causal_endpoint_density_fm3",
    "catalogue_delta_peak_mass_msun",
    "catalogue_delta_peak_radius_km",
    "catalogue_delta_r1_4_km",
    "catalogue_check_passed",
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


def _write_report(summaries: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Cold CompOSE comparison results",
        "",
        "All primary mass-radius curves below were calculated by the toolkit's TOV solver. Optional `eos.mr` files were used only after calculation as independent references.",
        "",
        "| Model | Sampled peak [Msun] | R(peak) [km] | R1.4 [km] | Causal-domain limit [Msun] | Catalogue | Literature convention |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        peak = item["metrics"]["sampled_peak"]
        at_1_4 = item["metrics"]["at_1_4_msun"]
        causal_mass = item["causal_endpoint"].get("mass_msun")
        literature = item["literature_crosscheck"]
        radius_1_4_text = (
            "n/a" if at_1_4 is None else f"{float(at_1_4['radius_km']):.4f}"
        )
        lines.append(
            f"| {item['model_id']} | {peak['mass_msun']:.5f} | {peak['radius_km']:.4f} | "
            f"{radius_1_4_text} | "
            f"{'n/a' if causal_mass is None else f'{causal_mass:.5f}'} | "
            f"{'PASS' if item['compose_catalogue_crosscheck']['passed'] else 'FAIL'} | "
            f"{literature['comparability']} ({literature['acceptance_status']}) |"
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
            "- Diagnostic ordering reductions are explicit and their keep-first/keep-later sensitivity is part of acceptance.",
            "- Literature values gate acceptance only when classified as like-for-like; contextual and provenance-only comparisons are still reported numerically when available.",
            "- `eos.mr` radius residuals use fixed masses below both sampled turning points; peak coordinates are reported separately.",
            "",
        )
    )
    (RESULTS_ROOT / "report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=EXPERIMENT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def _path_label(path: Path, *, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _hashed_file(path: Path, *, base: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"manifest input is missing: {path}")
    return {
        "path": _path_label(path, base=base),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest(
    summaries: Sequence[Mapping[str, Any]],
    config_path: Path,
    *,
    raw_root: Path,
) -> None:
    repository_root = EXPERIMENT_ROOT.parents[1]
    raw_files: list[dict[str, Any]] = []
    generated_files: list[dict[str, Any]] = []
    selected_slugs = [str(item["slug"]) for item in summaries]

    for summary in summaries:
        slug = str(summary["slug"])
        archive = raw_root / slug / str(summary["archive"]["archive_filename"])
        sidecar = archive.parent / "download.json"
        raw_files.extend(
            (
                _hashed_file(archive, base=EXPERIMENT_ROOT, role="pinned_raw_archive"),
                _hashed_file(
                    sidecar,
                    base=EXPERIMENT_ROOT,
                    role="acquisition_verification_sidecar",
                ),
            )
        )
        for root, role in (
            (DERIVED_ROOT / slug, "selected_model_derived_data"),
            (FIGURE_ROOT / slug, "selected_model_figure"),
        ):
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                generated_files.append(
                    _hashed_file(path, base=EXPERIMENT_ROOT, role=role)
                )

    comparison = FIGURE_ROOT / "comparison"
    if comparison.exists():
        for path in sorted(item for item in comparison.rglob("*") if item.is_file()):
            generated_files.append(
                _hashed_file(
                    path, base=EXPERIMENT_ROOT, role="selected_campaign_figure"
                )
            )
    for path in (
        RESULTS_ROOT / "all_models_summary.csv",
        RESULTS_ROOT / "acceptance.json",
        RESULTS_ROOT / "report.md",
    ):
        generated_files.append(
            _hashed_file(path, base=EXPERIMENT_ROOT, role="selected_campaign_result")
        )

    code_paths = [
        Path(__file__).resolve(),
        EXPERIMENT_ROOT / "acquire.py",
        config_path,
        repository_root / "pyproject.toml",
    ]
    source_root = repository_root / "src" / "neutron_star_eos"
    code_paths.extend(
        sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and (path.suffix in {".py", ".mplstyle"} or path.name == "py.typed")
        )
    )
    unique_code_paths = tuple(dict.fromkeys(path.resolve() for path in code_paths))
    code_inputs = [
        _hashed_file(path, base=repository_root, role="exact_code_input")
        for path in unique_code_paths
    ]
    files = [*raw_files, *generated_files]
    _write_json(
        MANIFEST_PATH,
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "config": {
                "path": _path_label(config_path, base=EXPERIMENT_ROOT),
                "sha256": sha256_file(config_path),
            },
            "software": {
                "neutron_star_eos_toolkit": toolkit_version,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy_version,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
                "git": _git_state(),
                "stellar_physical_constants": {
                    "authority": STELLAR_CONSTANT_AUTHORITY,
                    "authority_url": STELLAR_CONSTANT_REFERENCE_URL,
                    "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
                    "newtonian_gravitational_constant_m3_kg_s2": (
                        NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
                    ),
                    "solar_mass_kg": SOLAR_MASS_KG,
                    "MeV_J": MEV_J,
                    "fm3_m3": FM3_M3,
                    "gravity_conversion_Msun_per_km3_per_MeV_fm3": (GRAVITY_CONVERSION),
                    "solar_mass_length_km": SOLAR_MASS_LENGTH_KM,
                },
            },
            "models": selected_slugs,
            "enumeration_policy": {
                "raw": (
                    "selected canonical archive and download.json only; legacy live "
                    "downloads and unselected model files are excluded"
                ),
                "generated": (
                    "selected model outputs plus freshly cleaned campaign comparison "
                    "and result artifacts only"
                ),
            },
            "canonical_raw_files": raw_files,
            "generated_artifacts": generated_files,
            "exact_code_inputs": code_inputs,
            "files": files,
            "raw_archives_relicensed_under_mit": False,
        },
    )


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
    _write_json(
        RESULTS_ROOT / "acceptance.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "models": {item["slug"]: item["acceptance"] for item in summaries},
            "comparison_plots": comparison_plots,
            "campaign_gates": campaign_gates,
            "passed": campaign_passed,
        },
    )
    _write_report(summaries)
    (RESULTS_ROOT / "failure.json").unlink(missing_ok=True)
    _manifest(summaries, config_path, raw_root=raw_root)
    passed = campaign_passed
    print(
        f"campaign {'PASS' if passed else 'FAIL'}: {len(summaries)} models; "
        f"results in {RESULTS_ROOT}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
