"""Inspect the supplied CSV and calculate one source-boundary star."""

from __future__ import annotations

from pathlib import Path

from neutron_star_eos import open_eos


def main() -> None:
    table = Path(__file__).with_name("tabulated.csv")
    model = open_eos(table, kind="csv")
    print(model.summary())

    star = model.solve_star(central_pressure_mev_fm3=100.0)
    print(f"Truncated mass: {star.mass_msun:.6f} Msun")
    print(f"Boundary radius: {star.radius_km:.6f} km")
    print("Boundary status:", star.boundary_status)


if __name__ == "__main__":
    main()
