"""Calculate derived stellar and thermodynamic campaign products."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from campaign_io import write_columns as _write_columns
from campaign_io import write_json as _write_json
from sampling import (
    _compose_eos,
    _cs2_within_causal_threshold,
    _open_model,
    _validation_mode,
)
from scipy.optimize import brentq
from settings import (
    BASE_CONFIG,
    CAUSALITY_THRESHOLD_TOLERANCE,
    COMMON_BOUNDARY_DENSITY_FM3,
    CONVERGENCE_MASS_TOLERANCE_MSUN,
    CONVERGENCE_RADIUS_TOLERANCE_KM,
    BranchData,
)

from neutron_star_eos import EosModel, SequenceResult, StarResult
from neutron_star_eos.compose import ComposeEos


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
