"""Calculate one nonrotating star by integrating the TOV equations.

The bundled CSV is used so this example runs without downloading new data.
"""

from pathlib import Path

try:
    from neutron_star_eos import open_eos
except ModuleNotFoundError as exc:
    raise SystemExit("Install the project first: uv sync --all-extras") from exc

# ------------------------- parameters to edit -------------------------
CENTRAL_PRESSURE_MEV_FM3 = 100.0
CSV_PATH = Path(__file__).resolve().parents[1] / "examples" / "tabulated.csv"

if not CSV_PATH.is_file():
    raise SystemExit(f"CSV file not found: {CSV_PATH}")

model = open_eos(
    CSV_PATH,
    kind="csv",
    name="example-csv-eos",
    source_description=str(CSV_PATH),
)
eos = model.require_barotrope()
if not eos.pressure_min_mev_fm3 < CENTRAL_PRESSURE_MEV_FM3 <= eos.pressure_max_mev_fm3:
    raise SystemExit(
        f"Central pressure must lie above {eos.pressure_min_mev_fm3:g} and at or "
        f"below {eos.pressure_max_mev_fm3:g} MeV/fm^3"
    )

star = model.solve_star(CENTRAL_PRESSURE_MEV_FM3)

print(f"Model: {star.model_name}")
print(f"Central pressure: {star.central_pressure_mev_fm3:.6g} MeV/fm^3")
print(f"Gravitational mass: {star.mass_msun:.8f} Msun")
print(f"Source-boundary radius: {star.radius_km:.8f} km")
print(f"Boundary status: {star.boundary_status}")
