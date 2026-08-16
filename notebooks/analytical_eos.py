"""Authoritative user-editable analytical ``P(epsilon)`` definition.

The functions in this file, rather than a fixed parameterization, define the
analytical equation of state used by ``eos_experiments.ipynb``.  Keep the
pressure and sound-speed functions consistent and rerun the notebook from its
model-loading cell after every edit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neutron_star_eos import EosModel

MODEL_NAME = "user-analytical-eos"
ENERGY_DENSITY_DOMAIN_MEV_FM3 = (1.0, 400.0)
SOURCE_DESCRIPTION = "User-defined analytical pressure function"


def pressure_from_energy_density(epsilon: Any) -> Any:
    """Return ``P(epsilon)`` in MeV/fm^3.

    Replace the expression below with the intended analytical model.  The
    quadratic is only a small, runnable starter expression; it is not the
    analytical interface of the toolkit.
    """

    values = np.asarray(epsilon, dtype=float)
    return 1.0e-3 * values**2


def sound_speed_squared_from_energy_density(epsilon: Any) -> Any:
    """Return ``dP/d(epsilon)`` consistently with the function above."""

    values = np.asarray(epsilon, dtype=float)
    return 2.0e-3 * values


def build_model(*, source_identity: str | None = None) -> EosModel:
    """Build the toolkit model from the current definitions in this file."""

    return EosModel.from_analytical(
        name=MODEL_NAME,
        pressure_from_energy_density=pressure_from_energy_density,
        sound_speed_squared_from_energy_density=(
            sound_speed_squared_from_energy_density
        ),
        energy_density_domain_mev_fm3=ENERGY_DENSITY_DOMAIN_MEV_FM3,
        source=source_identity or SOURCE_DESCRIPTION,
    )
