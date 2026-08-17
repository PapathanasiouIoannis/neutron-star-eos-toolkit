from __future__ import annotations

import math
import unittest

import numpy as np

from neutron_star_eos import (
    AnalyticalEos,
    EosDomainError,
    EosInputError,
    StellarConfig,
    TabulatedEos,
    solve_sequence,
    solve_star,
)
from neutron_star_eos.stellar import (
    FM3_M3,
    GRAVITY_CONVERSION,
    MEV_J,
    NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2,
    SOLAR_MASS_KG,
    SOLAR_MASS_LENGTH_KM,
    SPEED_OF_LIGHT_M_S,
    STELLAR_CONSTANT_AUTHORITY,
    STELLAR_CONSTANT_REFERENCE_URL,
)

K = 1.0e-3


def analytical_polytrope() -> AnalyticalEos:
    return AnalyticalEos(
        name="test-polytrope",
        pressure_from_energy_density=lambda epsilon: K * np.asarray(epsilon) ** 2,
        sound_speed_squared_from_energy_density=lambda epsilon: (
            2.0 * K * np.asarray(epsilon)
        ),
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
    def test_tov_conversion_constants_follow_declared_compose_authority(self) -> None:
        expected_gravity_conversion = (
            4.0
            * math.pi
            * 1.0e9
            * (MEV_J / FM3_M3)
            / (SPEED_OF_LIGHT_M_S**2 * SOLAR_MASS_KG)
        )
        expected_solar_length = (
            NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
            * SOLAR_MASS_KG
            / SPEED_OF_LIGHT_M_S**2
            / 1.0e3
        )
        self.assertEqual(GRAVITY_CONVERSION, expected_gravity_conversion)
        self.assertEqual(SOLAR_MASS_LENGTH_KM, expected_solar_length)
        self.assertAlmostEqual(GRAVITY_CONVERSION, 1.1266082139640147e-5)
        self.assertAlmostEqual(SOLAR_MASS_LENGTH_KM, 1.4766251340718246)
        serialized = solve_star(analytical_polytrope(), 100.0).to_dict()
        constants = serialized["physical_constants"]
        self.assertEqual(constants["authority"], STELLAR_CONSTANT_AUTHORITY)
        self.assertEqual(constants["authority_url"], STELLAR_CONSTANT_REFERENCE_URL)
        self.assertEqual(
            constants["gravity_conversion_Msun_per_km3_per_MeV_fm3"],
            GRAVITY_CONVERSION,
        )
        sequence_constants = solve_sequence(
            analytical_polytrope(), [50.0, 100.0]
        ).to_dict()["physical_constants"]
        self.assertEqual(sequence_constants, constants)

    def test_preserved_background_regression_values(self) -> None:
        eos = analytical_polytrope()
        expected = {
            50.0: (3.482875480314816, 24.202184966277226),
            100.0: (3.6681122484640776, 22.23259039694679),
            150.0: (3.7098290643979612, 21.082158025191987),
            160.0: (3.712136870153775, 20.901444726084485),
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

    def test_log_pressure_and_radius_routes_agree_on_analytical_benchmark(self) -> None:
        radius_eos = analytical_polytrope()
        radius_result = solve_star(radius_eos, 100.0)
        log_pressure_eos = analytical_polytrope()
        log_pressure_eos.preferred_stellar_integration_variable = "log_pressure"
        log_pressure_result = solve_star(log_pressure_eos, 100.0)
        self.assertEqual(radius_result.integration_variable, "radius")
        self.assertEqual(log_pressure_result.integration_variable, "log_pressure")
        self.assertAlmostEqual(
            log_pressure_result.mass_msun,
            radius_result.mass_msun,
            delta=2.0e-8,
        )
        self.assertAlmostEqual(
            log_pressure_result.radius_km,
            radius_result.radius_km,
            delta=2.0e-8,
        )

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

    def test_background_diagnostic_integrates_acausal_region_explicitly(self) -> None:
        eos = AnalyticalEos(
            name="acausal-polytrope",
            pressure_from_energy_density=lambda epsilon: (
                2.0e-3 * np.asarray(epsilon) ** 2
            ),
            sound_speed_squared_from_energy_density=lambda epsilon: (
                4.0e-3 * np.asarray(epsilon)
            ),
            energy_density_domain_mev_fm3=(1.0, 400.0),
            source="diagnostic acausal integration regression",
        )
        with self.assertRaises(EosInputError):
            solve_star(eos, 320.0)

        for route in ("radius", "log_pressure"):
            eos.preferred_stellar_integration_variable = route
            star = solve_star(
                eos,
                320.0,
                validation_mode="background_diagnostic",
                config=StellarConfig(radius_max_km=50.0),
            )
            self.assertGreater(star.mass_msun, 0.0)
            self.assertGreater(star.radius_km, 0.0)
            self.assertGreater(star.central_sound_speed_squared, 1.0)
            self.assertEqual(star.eos_validation_status, "diagnostic_with_issues")
            self.assertEqual(star.eos_validation_issues, ("acausal",))

    def test_default_sequence_stays_inside_a_narrow_domain(self) -> None:
        eos = AnalyticalEos(
            name="narrow-domain",
            pressure_from_energy_density=lambda epsilon: 0.1 * np.asarray(epsilon),
            sound_speed_squared_from_energy_density=lambda epsilon: (
                0.1 * np.ones_like(np.asarray(epsilon))
            ),
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

    def test_solver_counts_require_integers(self) -> None:
        eos = analytical_polytrope()
        with self.assertRaisesRegex(TypeError, "points must be an integer"):
            solve_sequence(eos, points=9.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "profile_points must be an integer"):
            solve_star(eos, 100.0, config=StellarConfig(profile_points=2.5))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
