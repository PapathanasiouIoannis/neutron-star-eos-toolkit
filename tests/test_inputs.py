from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from neutron_star_eos import (
    AnalyticalEos,
    EosDomainError,
    EosInputError,
    TabulatedEos,
    build_compose_eos,
    load_compose_dataset,
    load_compose_eos,
    load_csv_eos,
)
from neutron_star_eos.cli import main as eos_tool_main
from neutron_star_eos.tabulated import (
    _deduplicate_derived_validation_grid,
    _validation_grid_with_exact_endpoints,
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


def sampled_polytrope(points: int = 65) -> TabulatedEos:
    epsilon = np.geomspace(1.0, 400.0, points)
    return TabulatedEos(
        name="sampled-polytrope",
        energy_density_mev_fm3=epsilon,
        pressure_mev_fm3=K * epsilon**2,
        source="independent test sampling",
    )


class EosInputTests(unittest.TestCase):
    def test_derived_validation_grid_collapses_ulp_near_duplicates(self) -> None:
        left = 9.307536220056526e-05
        right = 9.307536220056534e-05
        distinct = 1.0e-03
        collapsed = _deduplicate_derived_validation_grid([distinct, right, left])
        np.testing.assert_array_equal(collapsed, [left, distinct])
        upper = 1607.1196094212467
        grid = _validation_grid_with_exact_endpoints(
            [right, distinct, 1607.1196094212464],
            lower=left,
            upper=upper,
        )
        np.testing.assert_array_equal(grid, [left, distinct, upper])

    def test_analytical_forward_inverse_validation_and_domain(self) -> None:
        eos = analytical_polytrope()
        epsilon = np.asarray([1.0, 10.0, 100.0, 400.0])
        pressure = K * epsilon**2
        np.testing.assert_allclose(eos.pressure_from_energy_density(epsilon), pressure)
        np.testing.assert_allclose(eos.energy_density_from_pressure(pressure), epsilon)
        np.testing.assert_allclose(
            eos.sound_speed_squared_from_energy_density(epsilon), 2.0 * K * epsilon
        )
        observed_epsilon, observed_cs2 = eos(float(pressure[2]))
        self.assertAlmostEqual(observed_epsilon, 100.0, places=11)
        self.assertAlmostEqual(observed_cs2, 0.2, places=12)
        self.assertTrue(eos.validate(points=257).passed)
        with self.assertRaises(EosDomainError):
            eos.energy_density_from_pressure(eos.pressure_max_mev_fm3 * 1.01)

    def test_analytical_adapter_accepts_ordinary_scalar_functions(self) -> None:
        eos = AnalyticalEos(
            name="scalar-functions",
            pressure_from_energy_density=lambda epsilon: K * float(epsilon) ** 2,
            sound_speed_squared_from_energy_density=lambda epsilon: (
                2.0 * K * float(epsilon)
            ),
            energy_density_domain_mev_fm3=(1.0, 400.0),
            source="scalar-only test functions",
        )
        np.testing.assert_allclose(
            eos.pressure_from_energy_density([1.0, 10.0, 100.0]),
            [0.001, 0.1, 10.0],
        )
        self.assertTrue(eos.validate(points=65).passed)

    def test_analytical_sound_speed_must_match_pressure_derivative(self) -> None:
        eos = AnalyticalEos(
            name="inconsistent-derivative",
            pressure_from_energy_density=lambda epsilon: K * np.asarray(epsilon) ** 2,
            sound_speed_squared_from_energy_density=lambda epsilon: (
                0.1 * np.ones_like(np.asarray(epsilon, dtype=float))
            ),
            energy_density_domain_mev_fm3=(1.0, 400.0),
            source="deliberately inconsistent test definition",
        )
        report = eos.validate(points=257)
        self.assertFalse(report.passed)
        self.assertIn(
            "inconsistent_sound_speed", {issue.code for issue in report.issues}
        )

        bad_inverse = AnalyticalEos(
            name="inconsistent-inverse",
            pressure_from_energy_density=lambda epsilon: K * np.asarray(epsilon) ** 2,
            sound_speed_squared_from_energy_density=lambda epsilon: (
                2.0 * K * np.asarray(epsilon)
            ),
            energy_density_from_pressure=lambda pressure: (
                2.0 * np.ones_like(np.asarray(pressure, dtype=float))
            ),
            energy_density_domain_mev_fm3=(1.0, 400.0),
            source="deliberately inconsistent test inverse",
        )
        inverse_report = bad_inverse.validate(points=257)
        self.assertFalse(inverse_report.passed)
        self.assertIn(
            "inconsistent_pressure_inverse",
            {issue.code for issue in inverse_report.issues},
        )

    def test_csv_and_analytical_routes_agree_for_same_polytrope(self) -> None:
        analytical = analytical_polytrope()
        epsilon = np.geomspace(1.0, 400.0, 65)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "polytrope.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("epsilon_mev_fm3", "pressure_mev_fm3"))
                writer.writerows(zip(epsilon, K * epsilon**2))
            table = load_csv_eos(path)
            probes = np.geomspace(1.0, 400.0, 31)
            np.testing.assert_allclose(
                table.pressure_from_energy_density(probes),
                analytical.pressure_from_energy_density(probes),
                rtol=5.0e-14,
                atol=0.0,
            )
            np.testing.assert_allclose(
                table.sound_speed_squared_from_energy_density(probes),
                analytical.sound_speed_squared_from_energy_density(probes),
                rtol=1.0e-12,
                atol=0.0,
            )
            self.assertTrue(table.validate(points=257).passed)
            self.assertEqual(
                table.provenance()["source_metadata"]["source_file"]["bytes"],
                path.stat().st_size,
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = eos_tool_main(
                    ["validate", "csv", str(path), "--format", "json"]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(output.getvalue())["validation"]["status"], "pass"
            )

    def test_tabulated_input_fails_closed_without_repair(self) -> None:
        epsilon = np.asarray([1.0, 2.0, 3.0, 4.0])
        with self.assertRaisesRegex(EosInputError, "strictly increasing order"):
            TabulatedEos(
                name="unsorted",
                energy_density_mev_fm3=[1.0, 3.0, 2.0, 4.0],
                pressure_mev_fm3=[1.0, 2.0, 3.0, 4.0],
                source="test",
            )
        with self.assertRaisesRegex(EosInputError, "plateaus or jumps"):
            TabulatedEos(
                name="plateau",
                energy_density_mev_fm3=epsilon,
                pressure_mev_fm3=[1.0, 2.0, 2.0, 4.0],
                source="test",
            )
        acausal = TabulatedEos(
            name="acausal",
            energy_density_mev_fm3=epsilon,
            pressure_mev_fm3=2.0 * epsilon**2,
            source="test",
        )
        report = acausal.validate(points=257)
        self.assertFalse(report.passed)
        self.assertIn("acausal", {issue.code for issue in report.issues})

        narrow_epsilon = np.linspace(1.0, 101.0, 101)
        narrow_pressure = 0.01 + 0.05 * (narrow_epsilon - 1.0)
        narrow_pressure[51:] += 2.0
        narrow = TabulatedEos(
            name="narrow-acausal-interval",
            energy_density_mev_fm3=narrow_epsilon,
            pressure_mev_fm3=narrow_pressure,
            source="test",
        )
        narrow_report = narrow.validate(points=17)
        self.assertGreaterEqual(narrow_report.assessed_points, 501)
        self.assertIn("acausal", {issue.code for issue in narrow_report.issues})

        missed_by_fixed_sampling = TabulatedEos(
            name="interior-causality-maximum",
            energy_density_mev_fm3=np.geomspace(1.0, 100.0, 6),
            pressure_mev_fm3=[
                0.003296706917364005,
                0.8260063694980755,
                0.8509602842456219,
                0.9143505022963047,
                0.9315113231533734,
                0.9604693575133728,
            ],
            source="exact-extremum regression",
        )
        extrema_report = missed_by_fixed_sampling.validate(points=17)
        self.assertFalse(extrema_report.passed)
        self.assertGreater(extrema_report.cs2_max, 1.0)

        near_endpoint_critical_root = TabulatedEos(
            name="near-endpoint-critical-root",
            energy_density_mev_fm3=[
                1.0,
                1.089206276345296,
                1.1954509556025863,
                1.2387525296239172,
                1.2890843723482979,
                2.1454301187208205,
            ],
            pressure_mev_fm3=[
                0.03573226987154275,
                0.037189509814483684,
                0.9020034371564208,
                1.676301302995462,
                1.6983515763065538,
                1.8122183158931,
            ],
            source="near-endpoint extrema-grid regression",
        )
        endpoint_report = near_endpoint_critical_root.validate(points=17)
        self.assertFalse(endpoint_report.passed)
        self.assertIn("acausal", {issue.code for issue in endpoint_report.issues})

        pressure_probes = np.geomspace(
            missed_by_fixed_sampling.pressure_min_mev_fm3,
            missed_by_fixed_sampling.pressure_max_mev_fm3,
            101,
        )
        recovered_pressure = missed_by_fixed_sampling.pressure_from_energy_density(
            missed_by_fixed_sampling.energy_density_from_pressure(pressure_probes)
        )
        np.testing.assert_allclose(recovered_pressure, pressure_probes, rtol=2.0e-13)

        protected_epsilon = np.geomspace(1.0, 400.0, 65)
        protected = TabulatedEos(
            name="protected-metadata",
            energy_density_mev_fm3=protected_epsilon,
            pressure_mev_fm3=K * protected_epsilon**2,
            source="test",
            source_metadata={
                "units": "caller value must not override authority",
                "nested": {"value": 1},
            },
        )
        with self.assertRaises(ValueError):
            protected.pressure_mev_fm3[0] = 10.0
        metadata = protected.provenance()
        metadata["source_metadata"]["nested"]["value"] = 2
        self.assertEqual(protected.provenance()["units"], "MeV/fm^3")
        self.assertEqual(
            protected.provenance()["source_metadata"]["nested"]["value"], 1
        )

        with tempfile.TemporaryDirectory() as temporary:
            malformed = Path(temporary) / "surplus-fields.csv"
            malformed.write_text(
                "epsilon_mev_fm3,pressure_mev_fm3\n"
                "1,0.001,unexpected\n"
                "10,0.1,unexpected\n"
                "100,10,unexpected\n"
                "400,160,unexpected\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EosInputError, "exactly the number of fields"):
                load_csv_eos(malformed)

    def test_compose_cold_1d_parser_maps_indices_and_last_duplicate(self) -> None:
        neutron_mass = 939.5651828
        densities = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        epsilon = np.asarray([100.0, 200.0, 300.0, 400.0, 500.0])
        pressure = K * epsilon**2

        def row(index: int, q1: float, q3: float, q7: float) -> str:
            return f"0 {index} 2 {q1:.17g} 0 {q3:.17g} 0 0 {q7:.17g} {q7:.17g} 1 42"

        rows = []
        for index, density, eps, p in zip(range(5, 10), densities, epsilon, pressure):
            chemical_potential = (eps + p) / density
            rows.append(
                row(
                    index,
                    p / density,
                    chemical_potential / neutron_mass - 1.0,
                    eps / (density * neutron_mass) - 1.0,
                )
            )
        bad_duplicate = row(
            6,
            999.0,
            (epsilon[1] + pressure[1]) / (densities[1] * neutron_mass) - 1.0,
            epsilon[1] / (densities[1] * neutron_mass) - 1.0,
        )
        thermo = "\n".join(
            [
                f"{neutron_mass} 938.2718440 1",
                rows[2],
                bad_duplicate,
                rows[0],
                rows[4],
                rows[3],
                rows[1],
            ]
        )
        files = {
            "eos.t": "0\n0\n0.0\n",
            "eos.nb": "5\n9\n0.1\n0.2\n0.3\n0.4\n0.5\n",
            "eos.yq": "2\n2\n0.0\n",
            "eos.thermo": thermo + "\n",
            "eos.compo": "\n".join(
                [f"0 {index} 2 {1 if index < 7 else 2} 0 0" for index in range(5, 10)]
            )
            + "\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "compose.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, content in files.items():
                    archive.writestr(f"model/{name}", content)
            before = sorted(Path(temporary).iterdir())
            dataset = load_compose_dataset(
                archive_path,
                model_id="synthetic-compose-polytrope",
                source_url="https://example.invalid/compose-test",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            self.assertEqual(cold_slice.additional_values[0], (42.0,))
            self.assertTrue(cold_slice.report().continuous_barotrope_available)
            eos = build_compose_eos(
                dataset,
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            after = sorted(Path(temporary).iterdir())
            self.assertEqual(before, after)
            np.testing.assert_allclose(eos.baryon_density_fm3, densities)
            np.testing.assert_allclose(
                eos.energy_density_mev_fm3, epsilon, rtol=2.0e-15
            )
            np.testing.assert_allclose(eos.pressure_mev_fm3, pressure, rtol=2.0e-15)
            np.testing.assert_allclose(
                eos.baryon_chemical_potential_mev,
                (epsilon + pressure) / densities,
                rtol=2.0e-15,
            )
            self.assertEqual(eos.phase_codes, (1, 1, 2, 2, 2))
            self.assertEqual(
                eos.compose_metadata["thermodynamic_duplicate_indices_last_row_wins"], 1
            )
            self.assertEqual(eos.compose_metadata["phase_code_rows_missing"], 0)
            self.assertFalse(
                eos.compose_metadata["phase_codes_interpreted_as_discontinuities"]
            )
            self.assertTrue(eos.validate(points=257).passed)

            directory_path = Path(temporary) / "compose-directory"
            directory_path.mkdir()
            for name, content in files.items():
                (directory_path / name).write_text(content, encoding="utf-8")
            directory_eos = load_compose_eos(
                directory_path,
                model_id="synthetic-compose-directory",
                source_url="https://example.invalid/compose-test",
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            np.testing.assert_array_equal(
                directory_eos.energy_density_mev_fm3,
                eos.energy_density_mev_fm3,
            )
            np.testing.assert_array_equal(
                directory_eos.pressure_mev_fm3,
                eos.pressure_mev_fm3,
            )

            retained = load_compose_eos(
                archive_path,
                model_id="synthetic-compose-polytrope",
                source_url="https://example.invalid/compose-test",
                matter="cold_beta_equilibrated",
                includes_leptons=True,
                baryon_density_max_fm3=0.41,
            )
            self.assertEqual(len(retained.baryon_density_fm3), 4)
            self.assertEqual(retained.baryon_density_fm3[-1], 0.4)
            self.assertEqual(retained.compose_metadata["source_rows"], 5)
            self.assertEqual(retained.compose_metadata["retained_rows"], 4)

            without_leptons = Path(temporary) / "compose-no-leptons.zip"
            with zipfile.ZipFile(without_leptons, "w") as archive:
                for name, content in files.items():
                    if name == "eos.thermo":
                        content = content.replace(
                            f"{neutron_mass} 938.2718440 1",
                            f"{neutron_mass} 938.2718440 0",
                            1,
                        )
                    archive.writestr(f"model/{name}", content)
            with self.assertRaisesRegex(EosInputError, "leptons are absent"):
                load_compose_eos(
                    without_leptons,
                    model_id="no-leptons",
                    source_url="https://example.invalid/compose-test",
                    matter="cold_beta_equilibrated",
                    includes_leptons=True,
                )

            for label, token_index, replacement, diagnostic_code in (
                ("beta-diagnostic", 7, "1e-4", "beta_equilibrium_Q5_residual"),
                (
                    "temperature-diagnostic",
                    8,
                    "1e-4",
                    "zero_temperature_Q6_minus_Q7_residual",
                ),
            ):
                invalid_lines = thermo.splitlines()
                invalid_tokens = invalid_lines[1].split()
                invalid_tokens[token_index] = replacement
                invalid_lines[1] = " ".join(invalid_tokens)
                invalid_archive = Path(temporary) / f"{label}.zip"
                with zipfile.ZipFile(invalid_archive, "w") as archive:
                    for name, content in files.items():
                        if name == "eos.thermo":
                            content = "\n".join(invalid_lines) + "\n"
                        archive.writestr(f"model/{name}", content)
                diagnostic_dataset = load_compose_dataset(
                    invalid_archive,
                    model_id=label,
                    source_url="https://example.invalid/compose-test",
                )
                diagnostic_slice = diagnostic_dataset.cold_beta_equilibrium_slice(
                    matter="cold_beta_equilibrated",
                    includes_leptons=True,
                )
                self.assertIn(
                    diagnostic_code,
                    {item.code for item in diagnostic_slice.report().diagnostics},
                )
                diagnostic_eos = build_compose_eos(diagnostic_slice)
                self.assertTrue(diagnostic_eos.validate(points=257).passed)

    def test_compose_requires_explicit_physical_declarations(self) -> None:
        with self.assertRaisesRegex(EosInputError, "cold beta-equilibrated"):
            load_compose_eos(
                "missing",
                model_id="x",
                source_url="x",
                matter="finite_temperature",
                includes_leptons=True,
            )
        with self.assertRaisesRegex(EosInputError, "include leptons"):
            load_compose_eos(
                "missing",
                model_id="x",
                source_url="x",
                matter="cold_beta_equilibrated",
                includes_leptons=False,
            )


if __name__ == "__main__":
    unittest.main()
