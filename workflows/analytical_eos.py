"""Define and inspect your own analytical pressure function P(epsilon).

Run from the repository with: ``uv run python workflows/analytical_eos.py``
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from neutron_star_eos import EosModel
except ModuleNotFoundError as exc:
    raise SystemExit("Install the project first: uv sync --all-extras") from exc

# ------------------------- parameters to edit -------------------------
MODEL_NAME = "my-analytical-eos"
ENERGY_DENSITY_MIN_MEV_FM3 = 1.0
ENERGY_DENSITY_MAX_MEV_FM3 = 400.0
COEFFICIENT = 1.0e-3


def pressure_from_energy_density(energy_density_mev_fm3: Any) -> Any:
    """Return pressure P(epsilon) in MeV/fm^3."""

    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    return COEFFICIENT * epsilon**2


def sound_speed_squared(energy_density_mev_fm3: Any) -> Any:
    """Return dP/d(epsilon), consistently with P(epsilon) above."""

    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    return 2.0 * COEFFICIENT * epsilon


model = EosModel.from_analytical(
    name=MODEL_NAME,
    pressure_from_energy_density=pressure_from_energy_density,
    sound_speed_squared_from_energy_density=sound_speed_squared,
    energy_density_domain_mev_fm3=(
        ENERGY_DENSITY_MIN_MEV_FM3,
        ENERGY_DENSITY_MAX_MEV_FM3,
    ),
    source="User-edited workflows/analytical_eos.py",
)

print(model.summary())
validation = model.require_barotrope().validate()
print(f"Validation: {'passed' if validation.passed else 'failed'}")
if validation.issues:
    print("Issues:", ", ".join(issue.code for issue in validation.issues))
