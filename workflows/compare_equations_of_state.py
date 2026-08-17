"""Calculate and compare mass-radius curves for two equations of state."""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt

    from neutron_star_eos import EosModel, open_eos
except ModuleNotFoundError as exc:
    raise SystemExit("Install plotting support: uv sync --all-extras") from exc

# ------------------------- parameters to edit -------------------------
ANALYTICAL_COEFFICIENT = 8.0e-4
CENTRAL_PRESSURES_MEV_FM3 = np.geomspace(40.0, 120.0, 30)
OUTPUT_PNG = Path(__file__).resolve().parent / "eos_comparison.png"


def pressure(energy_density_mev_fm3: Any) -> Any:
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    return ANALYTICAL_COEFFICIENT * epsilon**2


def sound_speed_squared(energy_density_mev_fm3: Any) -> Any:
    epsilon = np.asarray(energy_density_mev_fm3, dtype=float)
    return 2.0 * ANALYTICAL_COEFFICIENT * epsilon


repository = Path(__file__).resolve().parents[1]
models = (
    open_eos(
        repository / "examples" / "tabulated.csv",
        kind="csv",
        name="sampled quadratic EoS",
        source_description="examples/tabulated.csv",
    ),
    EosModel.from_analytical(
        name="softer analytical EoS",
        pressure_from_energy_density=pressure,
        sound_speed_squared_from_energy_density=sound_speed_squared,
        energy_density_domain_mev_fm3=(1.0, 400.0),
        source="workflows/compare_equations_of_state.py",
    ),
)

figure, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)
for model in models:
    sequence = model.solve_sequence(CENTRAL_PRESSURES_MEV_FM3)
    if len(sequence.stars) != len(sequence.attempts):
        raise SystemExit(f"{model.model_name}: some stellar models failed")
    ax.plot(
        [star.radius_km for star in sequence.stars],
        [star.mass_msun for star in sequence.stars],
        marker="o",
        markersize=3,
        label=model.model_name,
    )
ax.set(xlabel="Source-boundary radius [km]", ylabel="Mass [Msun]")
ax.legend()
figure.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved: {OUTPUT_PNG}")
