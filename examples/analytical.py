"""Define, validate, and integrate a small analytical example."""

from __future__ import annotations

import numpy as np

from neutron_star_eos import AnalyticalEos, solve_star


def build_eos() -> AnalyticalEos:
    coefficient = 1.0e-3
    return AnalyticalEos(
        name="example-polytrope",
        pressure_from_energy_density=lambda epsilon: coefficient
        * np.asarray(epsilon) ** 2,
        sound_speed_squared_from_energy_density=lambda epsilon: 2.0
        * coefficient
        * np.asarray(epsilon),
        energy_density_domain_mev_fm3=(1.0, 400.0),
        source="educational P=K epsilon^2 example",
    )


def main() -> None:
    eos = build_eos()
    report = eos.validate().require_pass()
    star = solve_star(eos, central_pressure_mev_fm3=100.0)
    print(f"Validation: {'PASS' if report.passed else 'FAIL'}")
    print(f"Truncated mass: {star.mass_msun:.6f} Msun")
    print(f"Boundary radius: {star.radius_km:.6f} km")
    print("Boundary status:", star.boundary_status)


if __name__ == "__main__":
    main()
