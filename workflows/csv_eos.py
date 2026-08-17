"""Load and inspect an EoS supplied as epsilon and pressure CSV columns.

Run from the repository with: ``uv run python workflows/csv_eos.py``
"""

from pathlib import Path

try:
    from neutron_star_eos import open_eos
except ModuleNotFoundError as exc:
    raise SystemExit("Install the project first: uv sync --all-extras") from exc

# ------------------------- parameters to edit -------------------------
CSV_PATH = Path(__file__).resolve().parents[1] / "examples" / "tabulated.csv"
MODEL_NAME = "example-csv-eos"
ENERGY_DENSITY_COLUMN = "epsilon_mev_fm3"
PRESSURE_COLUMN = "pressure_mev_fm3"

if not CSV_PATH.is_file():
    raise SystemExit(f"CSV file not found: {CSV_PATH}")

model = open_eos(
    CSV_PATH,
    kind="csv",
    name=MODEL_NAME,
    source_description=str(CSV_PATH),
    epsilon_column=ENERGY_DENSITY_COLUMN,
    pressure_column=PRESSURE_COLUMN,
)
eos = model.require_barotrope()

print(model.summary())
print(
    f"Energy-density domain: {eos.energy_density_min_mev_fm3:g} to "
    f"{eos.energy_density_max_mev_fm3:g} MeV/fm^3"
)
print(
    f"Pressure domain: {eos.pressure_min_mev_fm3:g} to "
    f"{eos.pressure_max_mev_fm3:g} MeV/fm^3"
)
