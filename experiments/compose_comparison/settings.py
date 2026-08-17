"""Declared numerical settings and paths for the pinned CompOSE campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from neutron_star_eos import StellarConfig

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DERIVED_ROOT = EXPERIMENT_ROOT / "data" / "derived"
FIGURE_ROOT = EXPERIMENT_ROOT / "figures"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
MANIFEST_PATH = EXPERIMENT_ROOT / "manifest.json"
RUN_SCHEMA_VERSION = "compose-comparison-run-v2"

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
NOMINAL_SEAM_MASS_DELTA_MSUN = 1.0e-3
NOMINAL_SEAM_RADIUS_DELTA_KM = 0.01
PRE_PEAK_MASS_DECREASE_TOLERANCE_MSUN = 1.0e-8
PRESSURE_MERGE_RELATIVE_TOLERANCE = 64.0 * np.finfo(float).eps
CAUSALITY_THRESHOLD_TOLERANCE = 1.0e-10
REFERENCE_PEAK_EXCLUSION_MARGIN_MSUN = 0.05
REFERENCE_FIXED_MASSES_MSUN = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
OPTIONAL_REFERENCE_RADIUS_MATERIALITY_THRESHOLD_KM = 0.15
ORDERING_ACCEPTANCE_POLICY = (
    "both_diagnostic_reductions_complete_and_compose_catalogue_consistent"
)

CLOSURE_RESIDUAL_COLUMNS = (
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
    """Calculated mass-radius samples ordered by central pressure."""

    pressure_mev_fm3: np.ndarray
    baryon_density_fm3: np.ndarray
    mass_msun: np.ndarray
    radius_km: np.ndarray
    peak_index: int
    pre_peak_mass_decrease_count: int
