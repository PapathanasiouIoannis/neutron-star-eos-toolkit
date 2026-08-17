"""Calculate and save a mass-radius sequence from explicit central pressures."""

from pathlib import Path

import numpy as np

try:
    from neutron_star_eos import open_eos
    from neutron_star_eos.plotting import plot_mass_radius
except ModuleNotFoundError as exc:
    raise SystemExit("Install plotting support: uv sync --all-extras") from exc

# ------------------------- parameters to edit -------------------------
CENTRAL_PRESSURE_MIN_MEV_FM3 = 40.0
CENTRAL_PRESSURE_MAX_MEV_FM3 = 150.0
NUMBER_OF_STARS = 30
OUTPUT_PNG = Path(__file__).resolve().parent / "mass_radius.png"

csv_path = Path(__file__).resolve().parents[1] / "examples" / "tabulated.csv"
model = open_eos(
    csv_path,
    kind="csv",
    name="example-csv-eos",
    source_description=str(csv_path),
)
central_pressures = np.geomspace(
    CENTRAL_PRESSURE_MIN_MEV_FM3,
    CENTRAL_PRESSURE_MAX_MEV_FM3,
    NUMBER_OF_STARS,
)

sequence = model.solve_sequence(central_pressures)
if len(sequence.stars) != len(sequence.attempts):
    raise SystemExit("Some stars failed; inspect sequence.attempts before plotting")

ax = plot_mass_radius(sequence, connect=True)
figure = ax.get_figure(root=True)
if figure is None:
    raise RuntimeError("Matplotlib did not create a figure")
figure.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")

print(f"Solved all {len(sequence.stars)} stars")
print(f"Saved: {OUTPUT_PNG}")
