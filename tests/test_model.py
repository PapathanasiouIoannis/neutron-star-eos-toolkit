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

import neutron_star_eos
from neutron_star_eos import EosInputError, EosModel, StellarConfig, open_eos
from neutron_star_eos.cli import main as eos_tool_main


K = 1.0e-3
NEUTRON_MASS_MEV = 939.5651828


def analytical_model() -> EosModel:
    return EosModel.from_analytical(
        name="facade-polytrope",
        pressure_from_energy_density=lambda epsilon: K * np.asarray(epsilon) ** 2,
        sound_speed_squared_from_energy_density=lambda epsilon: 2.0
        * K
        * np.asarray(epsilon),
        energy_density_domain_mev_fm3=(1.0, 400.0),
        source="independent facade test P=K epsilon^2",
    )


def write_csv_fixture(root: Path) -> Path:
    path = root / "polytrope.csv"
    epsilon = np.geomspace(1.0, 400.0, 65)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("epsilon_mev_fm3", "pressure_mev_fm3"))
        writer.writerows(
            (format(float(value), ".17g"), format(float(K * value**2), ".17g"))
            for value in epsilon
        )
    return path


def write_compose_fixture(
    root: Path,
    *,
    pressure_reversal: bool = False,
    coefficient: float = 5.0e-4,
) -> Path:
    """Write the same minimal cold one-dimensional shape used by workflow tests."""

    densities = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), dtype=float)
    epsilon = 1000.0 * densities
    pressure = coefficient * epsilon**2
    if pressure_reversal:
        pressure[2] = 0.99 * pressure[1]

    thermo_rows: list[str] = []
    for density_index, density, energy, p in zip(
        range(5, 5 + len(densities)), densities, epsilon, pressure
    ):
        chemical_potential = (energy + p) / density
        q_values = (
            p / density,
            0.0,
            chemical_potential / NEUTRON_MASS_MEV - 1.0,
            0.0,
            0.0,
            energy / (density * NEUTRON_MASS_MEV) - 1.0,
            energy / (density * NEUTRON_MASS_MEV) - 1.0,
        )
        thermo_rows.append(
            f"0 {density_index} 2 "
            + " ".join(format(float(value), ".17g") for value in q_values)
            + " 0"
        )

    files = {
        "eos.t": "0\n0\n0.0\n",
        "eos.nb": (
            f"5\n{4 + len(densities)}\n"
            + "\n".join(format(float(value), ".17g") for value in densities)
            + "\n"
        ),
        "eos.yq": "2\n2\n0.0\n",
        "eos.thermo": (
            f"{NEUTRON_MASS_MEV:.17g} 938.2718440 1\n"
            + "\n".join(thermo_rows)
            + "\n"
        ),
    }
    archive_path = root / "compose.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"model/{name}", content)
    return archive_path


def run_cli(arguments: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exit_code = eos_tool_main(arguments)
    return exit_code, output.getvalue()


def file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ModelFacadeTests(unittest.TestCase):
    def test_root_wildcard_surface_is_small_but_advanced_names_remain_importable(self) -> None:
        self.assertEqual(
            neutron_star_eos.__all__,
            [
                "Capability",
                "CapabilityReport",
                "EosInputError",
                "EosModel",
                "StellarConfig",
                "open_eos",
            ],
        )
        self.assertTrue(callable(neutron_star_eos.load_compose_dataset))
        self.assertTrue(callable(neutron_star_eos.interpolate_compose_thermodynamics))
        self.assertTrue(callable(neutron_star_eos.solve_star))

    def test_cli_help_presents_new_commands_and_validation_alias(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                eos_tool_main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("{inspect,star,sequence,validate}", help_text)
        self.assertIn("inspect", help_text)
        self.assertIn("star", help_text)
        self.assertIn("sequence", help_text)
        self.assertIn("validate", help_text)
        self.assertIn("compatibility alias", help_text)

    def test_cli_inspect_writes_nothing_without_an_output_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_csv_fixture(root)
            before = file_snapshot(root)
            exit_code, output = run_cli(
                [
                    "inspect",
                    str(source),
                    "--kind",
                    "csv",
                    "--format",
                    "json",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(file_snapshot(root), before)
            payload = json.loads(output)
            self.assertEqual(payload["schema_version"], "eos-capability-report-v1")
            self.assertEqual(payload["input_kind"], "csv")
            self.assertNotIn("output_directory", payload)

    def test_cli_inspect_output_has_exact_bundle_and_refuses_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_csv_fixture(root)
            target = root / "inspection"
            arguments = [
                "inspect",
                str(source),
                "--kind",
                "csv",
                "--format",
                "json",
                "--output",
                str(target),
            ]
            exit_code, output = run_cli(arguments)
            self.assertEqual(exit_code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["output_directory"], str(target.resolve()))
            self.assertEqual(
                tuple(sorted(path.name for path in target.iterdir())),
                ("report.json", "summary.txt", "thermodynamics.csv"),
            )

            before = file_snapshot(target)
            refused_code, refused_output = run_cli(arguments)
            self.assertEqual(refused_code, 2)
            refused_payload = json.loads(refused_output)
            self.assertEqual(refused_payload["status"], "fail")
            self.assertIn("already exists", refused_payload["error"])
            self.assertEqual(file_snapshot(target), before)

    def test_cli_star_returns_csv_result_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = write_csv_fixture(Path(temporary))
            exit_code, output = run_cli(
                [
                    "star",
                    str(source),
                    "--kind",
                    "csv",
                    "--central-pressure-mev-fm3",
                    "100",
                    "--format",
                    "json",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output)
            self.assertEqual(payload["model"]["input_kind"], "csv")
            self.assertEqual(
                payload["star"]["schema_version"], "stellar-background-star-v1"
            )
            self.assertEqual(payload["star"]["solver_config"]["ode_rtol"], 1.0e-10)
            self.assertEqual(
                payload["star"]["model"]["eos_provenance_sha256"],
                payload["model"]["details"]["barotrope_provenance_sha256"],
            )
            self.assertGreater(payload["star"]["mass_Msun"], 0.0)
            self.assertGreater(payload["star"]["radius_km"], 0.0)
            self.assertFalse(payload["star"]["boundary"]["is_vacuum_surface"])

    def test_cli_star_returns_structured_failure_for_radius_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = write_csv_fixture(Path(temporary))
            exit_code, output = run_cli(
                [
                    "star",
                    str(source),
                    "--kind",
                    "csv",
                    "--central-pressure-mev-fm3",
                    "10",
                    "--format",
                    "json",
                ]
            )
            self.assertEqual(exit_code, 2)
            payload = json.loads(output)
            self.assertEqual(payload["status"], "fail")
            self.assertIn("lower-pressure boundary", payload["error"])

    def test_cli_sequence_reports_output_and_retains_all_attempt_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_csv_fixture(root)
            target = root / "sequence"
            exit_code, output = run_cli(
                [
                    "sequence",
                    str(source),
                    "--kind",
                    "csv",
                    "--points",
                    "9",
                    "--format",
                    "json",
                    "--output",
                    str(target),
                ]
            )
            payload = json.loads(output)
            attempts = payload["sequence"]["attempts"]
            self.assertEqual(
                payload["sequence"]["schema_version"],
                "stellar-background-sequence-v1",
            )
            self.assertEqual(
                payload["sequence"]["solver_config"]["profile_points"], 300
            )
            self.assertEqual(len(attempts), 9)
            self.assertTrue(
                all(
                    item["status"] in {"solved", "unavailable"}
                    for item in attempts
                )
            )
            self.assertEqual(
                exit_code,
                0 if payload["sequence"]["status"] == "complete" else 1,
            )
            self.assertEqual(payload["output_directory"], str(target.resolve()))

            saved_payload = json.loads(
                (target / "sequence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [item["status"] for item in saved_payload["attempts"]],
                [item["status"] for item in attempts],
            )
            with (target / "sequence.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual(
                [row["status"] for row in rows],
                [item["status"] for item in attempts],
            )

    def test_analytical_and_csv_models_report_capabilities_and_solve_stars(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_model = open_eos(
                write_csv_fixture(Path(temporary)),
                kind="csv",
                name="facade-csv-polytrope",
                source_description="independent sampled facade test",
            )
            models = (("analytical", analytical_model()), ("csv", csv_model))
            for expected_kind, model in models:
                with self.subTest(kind=expected_kind):
                    report = model.report()
                    self.assertEqual(model.kind, expected_kind)
                    self.assertEqual(report.input_kind, expected_kind)
                    for capability in (
                        "source",
                        "thermodynamics",
                        "continuous_barotrope",
                        "stellar_background",
                    ):
                        self.assertTrue(report.capability(capability).available)
                    self.assertEqual(
                        report.capability("composition").status, "not_applicable"
                    )
                    self.assertEqual(report.capability("tidal").status, "unavailable")
                    self.assertIs(model.require_barotrope(), model.barotrope)

                    star = model.solve_star(central_pressure_mev_fm3=100.0)
                    self.assertGreater(star.mass_msun, 0.0)
                    self.assertGreater(star.radius_km, 0.0)
                    self.assertEqual(
                        star.boundary_status,
                        "truncated_at_eos_lower_pressure_not_vacuum",
                    )
                    self.assertFalse(star.to_dict()["boundary"]["is_vacuum_surface"])

    def test_monotone_compose_model_retains_native_profile_and_barotrope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = open_eos(
                write_compose_fixture(Path(temporary)),
                kind="compose",
                model_id="facade-monotone",
                source_url="https://example.invalid/facade-monotone",
                includes_leptons=True,
                native_points=101,
            )

            report = model.report()
            self.assertEqual(model.kind, "compose")
            self.assertTrue(report.capability("source").available)
            self.assertTrue(report.capability("thermodynamics").available)
            self.assertTrue(report.capability("continuous_barotrope").available)
            self.assertTrue(report.capability("stellar_background").available)
            self.assertIsNotNone(model.dataset)
            self.assertIsNotNone(model.cold_slice)
            self.assertIsNotNone(model.native_thermodynamics)
            self.assertEqual(model.native_thermodynamics.source_rows, 6)
            self.assertIs(model.require_barotrope(), model.barotrope)
            self.assertEqual(report.to_dict()["details"]["barotrope"]["status"], "available")

    def test_compose_reversal_keeps_native_profile_but_blocks_barotrope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = open_eos(
                write_compose_fixture(Path(temporary), pressure_reversal=True),
                kind="compose",
                model_id="facade-reversal",
                source_url="https://example.invalid/facade-reversal",
                includes_leptons=True,
                native_points=101,
            )

            report = model.report()
            native = report.capability("thermodynamics")
            continuous = report.capability("continuous_barotrope")
            self.assertTrue(native.available)
            self.assertEqual(native.status, "available_with_diagnostics")
            self.assertIn("pressure_not_strictly_increasing", native.diagnostic_codes)
            self.assertEqual(continuous.status, "unavailable")
            self.assertIn(
                "pressure_not_strictly_increasing", continuous.diagnostic_codes
            )
            self.assertIsNotNone(model.native_thermodynamics)
            self.assertIsNone(model.barotrope)
            with self.assertRaises(EosInputError):
                model.require_barotrope()

    def test_background_diagnostic_remains_reachable_through_facade(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary), coefficient=1.0e-3
            )
            model = open_eos(
                archive,
                kind="compose",
                model_id="facade-acausal-diagnostic",
                source_url="https://example.invalid/facade-acausal-diagnostic",
                includes_leptons=True,
                native_points=101,
            )
            self.assertEqual(
                model.report().capability("continuous_barotrope").status,
                "unavailable",
            )
            with self.assertRaises(EosInputError):
                model.solve_star(100.0)
            diagnostic = model.solve_star(
                100.0,
                validation_mode="background_diagnostic",
            )
            self.assertEqual(
                diagnostic.eos_validation_status,
                "diagnostic_with_issues",
            )
            self.assertEqual(diagnostic.eos_validation_issues, ("acausal",))
            exit_code, output = run_cli(
                [
                    "star",
                    str(archive),
                    "--kind",
                    "compose",
                    "--model-id",
                    "facade-acausal-diagnostic",
                    "--source-url",
                    "https://example.invalid/facade-acausal-diagnostic",
                    "--includes-leptons",
                    "--central-pressure-mev-fm3",
                    "100",
                    "--validation-mode",
                    "background_diagnostic",
                    "--format",
                    "json",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(output)["star"]["eos_validation"]["status"],
                "diagnostic_with_issues",
            )

    def test_inspection_bundle_contents_are_deterministic_and_not_overwritten(self) -> None:
        model = analytical_model()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = model.write_inspection(root / "inspection-one")
            second = model.write_inspection(root / "inspection-two")
            expected_files = ("report.json", "summary.txt", "thermodynamics.csv")
            self.assertEqual(
                tuple(sorted(path.name for path in first.iterdir())), expected_files
            )
            self.assertEqual(
                tuple(sorted(path.name for path in second.iterdir())), expected_files
            )
            for name in expected_files:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            report = json.loads((first / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "eos-capability-report-v1")
            self.assertEqual(report["input_kind"], "analytical")
            self.assertEqual(report["software"]["python_implementation"], "CPython")
            for name in ("toolkit_version", "python_version", "numpy_version", "scipy_version"):
                self.assertTrue(report["software"][name])
            header = (first / "thermodynamics.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertEqual(
                header,
                "energy_density_mev_fm3,pressure_mev_fm3,sound_speed_squared",
            )
            original = {name: (first / name).read_bytes() for name in expected_files}
            with self.assertRaisesRegex(EosInputError, "already exists"):
                model.write_inspection(first)
            self.assertEqual(
                original, {name: (first / name).read_bytes() for name in expected_files}
            )

    def test_stellar_bundle_rejects_results_from_a_different_model(self) -> None:
        first = analytical_model()
        lower, upper = 1.0, 400.0
        amplitude = 0.05

        def different_pressure(epsilon):
            values = np.asarray(epsilon)
            x = (values - lower) / (upper - lower)
            return K * values**2 * (1.0 + amplitude * x * (1.0 - x))

        def different_sound_speed(epsilon):
            values = np.asarray(epsilon)
            x = (values - lower) / (upper - lower)
            shape = 1.0 + amplitude * x * (1.0 - x)
            shape_derivative = amplitude * (1.0 - 2.0 * x) / (upper - lower)
            return K * (2.0 * values * shape + values**2 * shape_derivative)

        second = EosModel.from_analytical(
            name="facade-polytrope",
            pressure_from_energy_density=different_pressure,
            sound_speed_squared_from_energy_density=different_sound_speed,
            energy_density_domain_mev_fm3=(lower, upper),
            source="independent facade test P=K epsilon^2",
        )
        star = second.solve_star(100.0)
        sequence = second.solve_sequence((50.0, 100.0, 150.0))
        self.assertNotEqual(
            first.solve_star(100.0).eos_provenance_sha256,
            star.eos_provenance_sha256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(EosInputError, "does not belong"):
                first.write_star(root / "star", star)
            with self.assertRaisesRegex(EosInputError, "does not belong"):
                first.write_sequence(root / "sequence", sequence)
            self.assertFalse((root / "star").exists())
            self.assertFalse((root / "sequence").exists())

    def test_analytical_identity_includes_supplied_inverse_behavior(self) -> None:
        first = analytical_model()
        lower, upper = 1.0, 400.0

        def perturbed_inverse(pressure):
            epsilon = np.sqrt(np.asarray(pressure) / K)
            x = (epsilon - lower) / (upper - lower)
            return epsilon * (1.0 + 1.0e-9 * x * (1.0 - x))

        second = EosModel.from_analytical(
            name="facade-polytrope",
            pressure_from_energy_density=lambda epsilon: K
            * np.asarray(epsilon) ** 2,
            sound_speed_squared_from_energy_density=lambda epsilon: 2.0
            * K
            * np.asarray(epsilon),
            energy_density_from_pressure=perturbed_inverse,
            energy_density_domain_mev_fm3=(lower, upper),
            source="independent facade test P=K epsilon^2",
        )
        self.assertTrue(first.require_barotrope().validate().passed)
        self.assertTrue(second.require_barotrope().validate().passed)
        first_star = first.solve_star(100.0)
        second_star = second.solve_star(100.0)
        self.assertNotEqual(
            first_star.eos_provenance_sha256,
            second_star.eos_provenance_sha256,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "star"
            with self.assertRaisesRegex(EosInputError, "does not belong"):
                first.write_star(output, second_star)
            self.assertFalse(output.exists())

    def test_retained_stellar_profile_is_serialized(self) -> None:
        model = analytical_model()
        star = model.solve_star(100.0, retain_profile=True)
        self.assertEqual(len(star.radius_profile_km), 300)
        with tempfile.TemporaryDirectory() as temporary:
            output = model.write_star(Path(temporary) / "star", star)
            payload = json.loads((output / "star.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["profile"]["retained"])
            self.assertEqual(len(payload["profile"]["radius_km"]), 300)
            self.assertEqual(len(payload["profile"]["mass_Msun"]), 300)
            self.assertEqual(payload["model"]["name"], model.model_name)
            self.assertEqual(len(payload["model"]["eos_provenance_sha256"]), 64)

    def test_sequence_bundle_retains_every_solved_and_unavailable_attempt(self) -> None:
        model = analytical_model()
        pressures = (50.0, 100.0, 150.0)
        sequence = model.solve_sequence(
            pressures,
            config=StellarConfig(radius_max_km=23.0),
        )
        self.assertEqual(sequence.status, "partial")
        self.assertEqual(len(sequence.attempts), len(pressures))
        self.assertEqual(
            tuple(attempt.status for attempt in sequence.attempts),
            ("unavailable", "solved", "solved"),
        )
        self.assertIsNotNone(sequence.attempts[0].reason)

        with tempfile.TemporaryDirectory() as temporary:
            output = model.write_sequence(Path(temporary) / "sequence", sequence)
            payload = json.loads(
                (output / "sequence.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["attempts"]), len(pressures))
            self.assertEqual(payload["solver_config"]["radius_max_km"], 23.0)
            self.assertEqual(
                [item["central_pressure_MeV_fm3"] for item in payload["attempts"]],
                list(pressures),
            )
            self.assertIsNone(payload["attempts"][0]["star"])
            self.assertIsNotNone(payload["attempts"][0]["reason"])

            with (output / "sequence.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(pressures))
            self.assertEqual(
                [float(row["central_pressure_mev_fm3"]) for row in rows],
                list(pressures),
            )
            self.assertEqual(
                [row["status"] for row in rows],
                [attempt.status for attempt in sequence.attempts],
            )
            self.assertTrue(rows[0]["reason"])
            self.assertEqual(rows[0]["mass_msun"], "")
            self.assertTrue(rows[1]["mass_msun"])
            self.assertTrue(rows[2]["mass_msun"])


if __name__ == "__main__":
    unittest.main()
