"""Choose CompOSE models and sample their calculated TOV sequences."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from settings import (
    BASE_CONFIG,
    CAUSALITY_THRESHOLD_TOLERANCE,
    CENTRAL_DENSITY_FLOOR_FM3,
    PRE_PEAK_MASS_DECREASE_TOLERANCE_MSUN,
    PRESSURE_MERGE_RELATIVE_TOLERANCE,
    RETRY_CONFIG,
    BranchData,
)

from neutron_star_eos import (
    EosInputError,
    EosModel,
    SequenceAttempt,
    SequenceResult,
    StarResult,
    StellarConfig,
    open_eos,
)
from neutron_star_eos.compose import ComposeEos


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
    candidates = sorted(
        (attempt for sequence in sequences for attempt in sequence.attempts),
        key=lambda attempt: attempt.central_pressure_mev_fm3,
    )
    merged: list[SequenceAttempt] = []
    for attempt in candidates:
        if merged and math.isclose(
            attempt.central_pressure_mev_fm3,
            merged[-1].central_pressure_mev_fm3,
            rel_tol=PRESSURE_MERGE_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        ):
            if merged[-1].star is None and attempt.star is not None:
                merged[-1] = attempt
            continue
        merged.append(attempt)
    attempts = tuple(merged)
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
    combined_candidates = len(coarse.attempts) + len(refined.attempts)
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
            "combined_candidate_attempts": combined_candidates,
            "combined_attempts": len(combined.attempts),
            "combined_solved": len(combined.stars),
            "near_duplicate_pressure_attempts_removed": (
                combined_candidates - len(combined.attempts)
            ),
            "pressure_merge_relative_tolerance": (PRESSURE_MERGE_RELATIVE_TOLERANCE),
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
