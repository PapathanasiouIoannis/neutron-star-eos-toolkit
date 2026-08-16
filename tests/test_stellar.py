from __future__ import annotations

import unittest

import numpy as np

from neutron_star_eos import (
    AnalyticalEos,
    EosDomainError,
    TabulatedEos,
    solve_sequence,
    solve_star,
)


K = 1.0e-3


def analytical_polytrope() -> AnalyticalEos:
    return AnalyticalEos(
        name="test-polytrope",
        pressure_from_energy_density=lambda epsilon: K * np.asarray(epsilon) ** 2,
        sound_speed_squared_from_energy_density=lambda epsilon: 2.0
        * K
        * np.asarray(epsilon),
        energy_density_domain_mev_fm3=(1.0, 400.0),
        source="independent test definition P=K epsilon^2",
    )


def sampled_polytrope(points: int) -> TabulatedEos:
    epsilon = np.geomspace(1.0, 400.0, points)
    return TabulatedEos(
        name=f"sampled-polytrope-{points}",
        energy_density_mev_fm3=epsilon,
        pressure_mev_fm3=K * epsilon**2,
        source="independent test sampling",
    )


class StellarTests(unittest.TestCase):
    def test_preserved_background_regression_values(self) -> None:
        eos = analytical_polytrope()
        expected = {
            50.0: (3.4870031312155505, 24.230455199462565),
            100.0: (3.6724594285203196, 22.258559971427193),
            150.0: (3.714225684192685, 21.106783795149045),
            160.0: (3.716536224992158, 20.92585940641555),
        }
        for pressure, (mass, radius) in expected.items():
            star = solve_star(eos, pressure)
            self.assertAlmostEqual(star.mass_msun, mass, delta=5.0e-9)
            self.assertAlmostEqual(star.radius_km, radius, delta=5.0e-9)

    def test_background_uses_explicit_nonvacuum_boundary(self) -> None:
        eos = analytical_polytrope()
        star = solve_star(eos, 100.0, retain_profile=True)
        self.assertGreater(star.mass_msun, 0.0)
        self.assertGreater(star.radius_km, 0.0)
        self.assertEqual(
            star.boundary_pressure_mev_fm3,
            eos.pressure_min_mev_fm3,
        )
        self.assertEqual(
            star.boundary_status,
            "truncated_at_eos_lower_pressure_not_vacuum",
        )
        self.assertEqual(len(star.radius_profile_km), 300)
        self.assertEqual(len(star.mass_profile_msun), 300)
        self.assertTrue(np.all(np.diff(star.radius_profile_km) > 0.0))
        self.assertTrue(np.all(np.diff(star.mass_profile_msun) >= 0.0))

    def test_sampled_tables_converge_to_analytical_background(self) -> None:
        reference = solve_star(analytical_polytrope(), 100.0)
        errors = []
        for points in (65, 129, 257):
            candidate = solve_star(sampled_polytrope(points), 100.0)
            errors.append(
                max(
                    abs(candidate.mass_msun / reference.mass_msun - 1.0),
                    abs(candidate.radius_km / reference.radius_km - 1.0),
                )
            )
        self.assertLess(errors[-1], 2.0e-8)
        self.assertLess(max(errors), 2.0e-8)

    def test_sequence_retains_every_requested_status(self) -> None:
        eos = analytical_polytrope()
        sequence = solve_sequence(eos, [50.0, 100.0, 150.0])
        self.assertEqual(sequence.status, "complete")
        self.assertEqual(len(sequence.attempts), 3)
        self.assertEqual(len(sequence.stars), 3)
        self.assertTrue(all(item.status == "solved" for item in sequence.attempts))
        self.assertEqual(
            sequence.boundary_status,
            "truncated_at_eos_lower_pressure_not_vacuum",
        )

    def test_default_sequence_stays_inside_a_narrow_domain(self) -> None:
        eos = AnalyticalEos(
            name="narrow-domain",
            pressure_from_energy_density=lambda epsilon: 0.1 * np.asarray(epsilon),
            sound_speed_squared_from_energy_density=lambda epsilon: 0.1
            * np.ones_like(np.asarray(epsilon)),
            energy_density_domain_mev_fm3=(100.0, 100.00004),
            source="narrow-domain regression",
        )
        sequence = solve_sequence(eos, points=9)
        pressures = [item.central_pressure_mev_fm3 for item in sequence.attempts]
        self.assertTrue(all(eos.pressure_min_mev_fm3 < p for p in pressures))
        self.assertTrue(all(p <= eos.pressure_max_mev_fm3 for p in pressures))
        self.assertTrue(np.all(np.diff(pressures) > 0.0))

    def test_stellar_domain_is_fail_closed(self) -> None:
        eos = analytical_polytrope()
        with self.assertRaises(EosDomainError):
            solve_star(eos, eos.pressure_min_mev_fm3)
        with self.assertRaises(EosDomainError):
            solve_star(eos, eos.pressure_max_mev_fm3 * 1.01)
        with self.assertRaises(EosDomainError):
            solve_sequence(eos, [1.0, eos.pressure_max_mev_fm3 * 1.01])


if __name__ == "__main__":
    unittest.main()
