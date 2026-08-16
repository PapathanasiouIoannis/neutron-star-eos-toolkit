from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from neutron_star_eos import (
    EosInputError,
    StellarConfig,
    build_compose_eos,
    interpolate_compose_thermodynamics,
    load_compose_dataset,
    solve_star,
)
from neutron_star_eos.cli import main as eos_tool_main

NEUTRON_MASS_MEV = 939.5651828
K = 5.0e-4


def write_compose_fixture(
    root: Path,
    *,
    temperatures: tuple[float, ...] = (0.0,),
    charge_fraction: float = 0.0,
    neutron_mass_mev: float = NEUTRON_MASS_MEV,
    q5: float = 0.0,
    pressure_reversal: bool = False,
    coefficient: float = K,
    valid_microphysics: bool = True,
    density_values: tuple[float, ...] | None = None,
    energy_density_values: tuple[float, ...] | None = None,
    pressure_values: tuple[float, ...] | None = None,
    native_q_values: tuple[tuple[float, float, float, float, float, float, float], ...]
    | None = None,
    composition_quadruple: tuple[int, float, float, float] | None = None,
    additional_values_by_row: tuple[tuple[float, ...], ...] | None = None,
) -> Path:
    densities = np.asarray(
        density_values or (0.1, 0.2, 0.3, 0.4, 0.5, 0.6), dtype=float
    )
    epsilon = np.asarray(
        energy_density_values or tuple(1000.0 * densities), dtype=float
    )
    pressure = np.asarray(
        pressure_values if pressure_values is not None else coefficient * epsilon**2,
        dtype=float,
    )
    if not (len(densities) == len(epsilon) == len(pressure)):
        raise ValueError("synthetic CompOSE arrays must have the same length")
    if native_q_values is not None and len(native_q_values) != len(densities):
        raise ValueError("synthetic native-Q rows must align with the density axis")
    if additional_values_by_row is not None and len(additional_values_by_row) != len(
        densities
    ):
        raise ValueError("synthetic Nadd rows must align with the density axis")
    if pressure_reversal:
        pressure[2] = pressure[1] * 0.99
    rows: list[str] = []
    composition: list[str] = []
    for temperature_index, _temperature in enumerate(temperatures):
        for source_position, (density_index, density, eps, p) in enumerate(
            zip(range(5, 5 + len(densities)), densities, epsilon, pressure)
        ):
            if native_q_values is None:
                chemical_potential = (eps + p) / density
                q1 = p / density
                q_values = (
                    q1,
                    0.0,
                    chemical_potential / neutron_mass_mev - 1.0,
                    0.0,
                    q5,
                    eps / (density * neutron_mass_mev) - 1.0,
                    eps / (density * neutron_mass_mev) - 1.0,
                )
            else:
                q_values = native_q_values[source_position]
            additional_values = (
                (42.0,)
                if additional_values_by_row is None
                else additional_values_by_row[source_position]
            )
            rows.append(
                f"{temperature_index} {density_index} 2 "
                + " ".join(f"{value:.17g}" for value in q_values)
                + f" {len(additional_values)}"
                + (
                    ""
                    if not additional_values
                    else " " + " ".join(f"{value:.17g}" for value in additional_values)
                )
            )
            composition_payload = "0 0"
            if composition_quadruple is not None:
                code, average_mass, average_charge, abundance = composition_quadruple
                composition_payload = (
                    f"0 1 {code} {average_mass:.17g} "
                    f"{average_charge:.17g} {abundance:.17g}"
                )
            composition.append(
                f"{temperature_index} {density_index} 2 "
                f"{1 if density_index < 8 else 2} {composition_payload}"
            )
    thermo = "\n".join([f"{neutron_mass_mev:.17g} 938.2718440 1", *rows]) + "\n"
    files = {
        "eos.t": (
            f"0\n{len(temperatures) - 1}\n"
            + "\n".join(str(value) for value in temperatures)
            + "\n"
        ),
        "eos.nb": (
            f"5\n{4 + len(densities)}\n"
            + "\n".join(f"{value:.17g}" for value in densities)
            + "\n"
        ),
        "eos.yq": f"2\n2\n{charge_fraction:.17g}\n",
        "eos.thermo": thermo,
        "eos.compo": "\n".join(composition) + "\n",
        "eos.micro": (
            "\n".join(
                f"0 {density_index} 2 1 999 {density_index / 10:.17g}"
                for density_index in range(5, 5 + len(densities))
            )
            + "\n"
            if valid_microphysics
            else "synthetic optional microphysics payload\n"
        ),
        "eos.init": "synthetic initialization metadata\n",
        "eos.mr": "10.0 1.0\n",
    }
    archive_path = root / "compose.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"model/{name}", content)
    return archive_path


class ComposeWorkflowTests(unittest.TestCase):
    def test_density_axis_requires_strictly_positive_baryon_density(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary),
                density_values=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                native_q_values=((1.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1),) * 6,
            )
            with self.assertRaisesRegex(
                EosInputError, "eos.nb must be strictly positive"
            ):
                load_compose_dataset(
                    archive,
                    model_id="zero-density",
                    source_url="https://example.invalid/zero-density",
                )

    def test_zero_pressure_remains_native_visible_but_blocks_log_barotrope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary),
                pressure_values=(0.0, 20.0, 45.0, 80.0, 125.0, 180.0),
            )
            dataset = load_compose_dataset(
                archive,
                model_id="zero-pressure",
                source_url="https://example.invalid/zero-pressure",
            )
            cold = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated", includes_leptons=True
            )
            report = cold.report()
            self.assertFalse(report.continuous_barotrope_available)
            self.assertIn(
                "zero_pressure_source_node",
                {item.code for item in report.diagnostics},
            )
            profile = interpolate_compose_thermodynamics(cold, points=17)
            self.assertEqual(float(profile.column("pressure_mev_fm3")[0]), 0.0)
            self.assertIn(
                "sampled_pressure_nonpositive",
                {item.code for item in profile.diagnostics},
            )
            with self.assertRaisesRegex(
                EosInputError, "continuous invertible barotrope"
            ):
                build_compose_eos(cold)

    def test_variable_nadd_width_is_preserved_with_explicit_availability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary),
                additional_values_by_row=(
                    (1.0,),
                    (2.0, 20.0),
                    (),
                    (4.0, 40.0),
                    (5.0,),
                    (6.0, 60.0),
                ),
            )
            dataset = load_compose_dataset(
                archive,
                model_id="variable-nadd",
                source_url="https://example.invalid/variable-nadd",
            )
            cold = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated", includes_leptons=True
            )
            profile = interpolate_compose_thermodynamics(cold, points=17)
            source = profile.column("source_node_position") >= 0.0
            np.testing.assert_allclose(
                profile.column("additional_1")[source],
                np.asarray((1.0, 2.0, np.nan, 4.0, 5.0, 6.0)),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                profile.column("additional_2")[source],
                np.asarray((np.nan, 20.0, np.nan, 40.0, np.nan, 60.0)),
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                profile.column("additional_2_available")[source],
                np.asarray((0.0, 1.0, 0.0, 1.0, 0.0, 1.0)),
            )

    def test_dataset_preserves_source_before_barotrope_construction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary), temperatures=(0.0, 1.0))
            dataset = load_compose_dataset(
                archive,
                model_id="synthetic",
                source_url="https://example.invalid/synthetic",
            )
            self.assertEqual(len(dataset.thermodynamic_rows), 12)
            with self.assertRaisesRegex(AttributeError, "immutable"):
                dataset.model_id = "changed"
            self.assertIn("eos.micro", dataset.available_files)
            self.assertEqual(
                dataset.source_file_bytes("eos.micro"),
                (
                    "\n".join(
                        f"0 {density_index} 2 1 999 {density_index / 10:.17g}"
                        for density_index in range(5, 11)
                    )
                    + "\n"
                ).encode("ascii"),
            )
            with self.assertRaises(TypeError):
                dataset._source_files["eos.micro"] = b"changed"  # type: ignore[index]
            provenance = dataset.provenance()
            provenance["source_identity"]["kind"] = "changed"
            self.assertEqual(dataset.provenance()["source_identity"]["kind"], "zip")
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            self.assertEqual(cold_slice.additional_values[0], (42.0,))
            with self.assertRaisesRegex(AttributeError, "immutable"):
                cold_slice.matter_declaration = "changed"
            self.assertEqual(cold_slice.phase_codes, (1, 1, 1, 2, 2, 2))
            self.assertTrue(cold_slice.report().continuous_barotrope_available)
            eos = build_compose_eos(cold_slice)
            self.assertTrue(eos.validate(points=257).passed)
            np.testing.assert_allclose(
                eos.pressure_from_energy_density(eos.energy_density_mev_fm3),
                eos.pressure_mev_fm3,
                rtol=5.0e-14,
                atol=0.0,
            )

    def test_source_diagnostics_warn_without_invalidating_pressure_energy_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary), q5=1.0e-4)
            dataset = load_compose_dataset(
                archive,
                model_id="diagnostic",
                source_url="https://example.invalid/diagnostic",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            report = cold_slice.report()
            self.assertEqual(
                report.status,
                "continuous_barotrope_available_with_source_diagnostics",
            )
            self.assertIn(
                "beta_equilibrium_Q5_residual",
                {item.code for item in report.diagnostics},
            )
            self.assertTrue(build_compose_eos(cold_slice).validate(points=257).passed)

    def test_native_q_profile_reconstructs_all_cold_quantities_before_barotrope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary),
                valid_microphysics=True,
            )
            dataset = load_compose_dataset(
                archive,
                model_id="native-q",
                source_url="https://example.invalid/native-q",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            density = np.asarray([0.1, 0.15, 0.2])
            profile = interpolate_compose_thermodynamics(
                cold_slice,
                density,
                include_source_nodes=False,
            )
            np.testing.assert_allclose(
                profile.column("pressure_mev_fm3"),
                K * (1000.0 * density) ** 2,
                rtol=2.0e-15,
                atol=1.0e-14,
            )
            np.testing.assert_allclose(
                profile.column("energy_density_mev_fm3"),
                1000.0 * density,
                rtol=2.0e-15,
                atol=0.0,
            )
            np.testing.assert_allclose(
                profile.column("sound_speed_squared_curve_derivative"),
                density,
                rtol=5.0e-15,
                atol=0.0,
            )
            expected_compose_cs2 = (1000.0 * density) / (1000.0 + 500.0 * density)
            np.testing.assert_allclose(
                profile.column("sound_speed_squared_compose_thermodynamic"),
                expected_compose_cs2,
                rtol=5.0e-15,
                atol=0.0,
            )
            self.assertIn("q1_pressure_per_baryon_mev", profile.column_names)
            self.assertIn(
                "q7_internal_energy_per_baryon_over_mn_minus_1", profile.column_names
            )
            self.assertNotIn("composition_pair_999", profile.column_names)
            self.assertIn("micro_999", profile.column_names)
            self.assertEqual(profile.column("micro_999")[0], 0.5)
            with self.assertRaises(ValueError):
                profile.column("pressure_mev_fm3")[0] = 0.0

    def test_native_q_consistent_fixture_closes_and_sound_speed_routes_coincide(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            densities = (1.0, 2.0, 3.0)
            native_q_values = tuple(
                (density, 0.0, 0.2 * density, 0.0, 0.0, 0.1 * density, 0.1 * density)
                for density in densities
            )
            archive = write_compose_fixture(
                Path(temporary),
                neutron_mass_mev=10.0,
                density_values=densities,
                native_q_values=native_q_values,
            )
            dataset = load_compose_dataset(
                archive,
                model_id="consistent-native-q",
                source_url="https://example.invalid/consistent-native-q",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            profile = interpolate_compose_thermodynamics(
                cold_slice,
                np.asarray((2.0,)),
                include_source_nodes=False,
            )

            expected_scalars = {
                "pressure_mev_fm3": 4.0,
                "energy_density_mev_fm3": 24.0,
                "baryon_chemical_potential_mev": 14.0,
                "free_energy_per_baryon_mev": 12.0,
                "internal_energy_per_baryon_mev": 12.0,
                "enthalpy_per_baryon_mev": 14.0,
                "gibbs_free_energy_per_baryon_mev": 14.0,
                "dpressure_dnB_mev": 4.0,
                "denergy_density_dnB_mev": 14.0,
                "barotropic_adiabatic_index": 2.0,
                "compose_heat_capacity_ratio_at_zero_temperature": 1.0,
                "isothermal_compressibility_mev_inverse_fm3": 1.0 / 8.0,
                "isentropic_compressibility_mev_inverse_fm3": 1.0 / 8.0,
            }
            for column, expected in expected_scalars.items():
                np.testing.assert_allclose(
                    profile.column(column),
                    np.asarray((expected,)),
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )

            for column in (
                "sound_speed_squared_curve_derivative",
                "sound_speed_squared_compose_thermodynamic",
                "sound_speed_squared_cold_beta_mu_derivative",
            ):
                np.testing.assert_allclose(
                    profile.column(column),
                    np.asarray((2.0 / 7.0,)),
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )
            for column in (
                "euler_residual_mev_fm3",
                "first_law_residual_mev",
                "gibbs_duhem_residual_mev",
                "free_energy_pressure_residual_mev_fm3",
                "free_energy_muB_residual_mev",
                "q5_beta_equilibrium_residual",
                "q6_minus_q7_zero_temperature_residual",
                "sound_speed_squared_definition_difference",
                "sound_speed_squared_mu_minus_compose",
            ):
                np.testing.assert_allclose(
                    profile.column(column),
                    np.zeros(1),
                    rtol=0.0,
                    atol=2.0e-14,
                )

    def test_native_q_inconsistent_fixture_exposes_route_and_closure_divergence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            densities = (1.0, 2.0, 3.0)
            native_q_values = tuple(
                (density, 0.0, 0.4 * density, 0.0, 0.0, 0.15 * density, 0.05 * density)
                for density in densities
            )
            archive = write_compose_fixture(
                Path(temporary),
                neutron_mass_mev=10.0,
                density_values=densities,
                native_q_values=native_q_values,
            )
            dataset = load_compose_dataset(
                archive,
                model_id="inconsistent-native-q",
                source_url="https://example.invalid/inconsistent-native-q",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            profile = interpolate_compose_thermodynamics(
                cold_slice,
                np.asarray((2.0,)),
                include_source_nodes=False,
            )

            expected_routes = {
                "sound_speed_squared_curve_derivative": 1.0 / 3.0,
                "sound_speed_squared_compose_thermodynamic": 4.0 / 13.0,
                "sound_speed_squared_cold_beta_mu_derivative": 4.0 / 9.0,
            }
            for column, expected in expected_routes.items():
                np.testing.assert_allclose(
                    profile.column(column),
                    np.asarray((expected,)),
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )
            self.assertEqual(len(set(expected_routes.values())), 3)

            expected_residuals = {
                "euler_residual_mev_fm3": -10.0,
                "first_law_residual_mev": -6.0,
                "gibbs_duhem_residual_mev": -4.0,
                "free_energy_pressure_residual_mev_fm3": -2.0,
                "free_energy_muB_residual_mev": 2.0,
                "q6_minus_q7_zero_temperature_residual": 0.2,
            }
            for column, expected in expected_residuals.items():
                np.testing.assert_allclose(
                    profile.column(column),
                    np.asarray((expected,)),
                    rtol=2.0e-14,
                    atol=2.0e-14,
                )

    def test_composition_quadruple_preserves_compose_field_order_and_neutron_number(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary),
                composition_quadruple=(7, 101.0, 37.0, 0.123),
            )
            dataset = load_compose_dataset(
                archive,
                model_id="composition-quadruple",
                source_url="https://example.invalid/composition-quadruple",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            profile = interpolate_compose_thermodynamics(
                cold_slice,
                np.asarray((0.15,)),
                include_source_nodes=False,
            )
            expected = {
                "composition_quadruple_7_Aav": 101.0,
                "composition_quadruple_7_Zav": 37.0,
                "composition_quadruple_7_Yav": 0.123,
                "composition_quadruple_7_Nav": 64.0,
            }
            for column, value in expected.items():
                np.testing.assert_allclose(
                    profile.column(column),
                    np.asarray((value,)),
                    rtol=2.0e-15,
                    atol=2.0e-15,
                )

    def test_cold_beta_reduction_rejects_nonzero_single_yq_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary), charge_fraction=0.1)
            dataset = load_compose_dataset(
                archive,
                model_id="fixed-nonzero-yq",
                source_url="https://example.invalid/fixed-nonzero-yq",
            )
            with self.assertRaisesRegex(EosInputError, "Yq=0 sentinel"):
                dataset.cold_beta_equilibrium_slice(
                    matter="cold_beta_equilibrated",
                    includes_leptons=True,
                )

    def test_native_profile_summary_retains_hashes_positions_and_cold_na_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary))
            dataset = load_compose_dataset(
                archive,
                model_id="native-provenance",
                source_url="https://example.invalid/native-provenance",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            summary = interpolate_compose_thermodynamics(
                cold_slice,
                np.asarray((0.1, 0.2)),
                include_source_nodes=False,
            ).summary()

            provenance = summary["provenance"]
            source_identity = provenance["dataset"]["source_identity"]
            self.assertEqual(
                source_identity["archive_sha256"],
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                source_identity["files"]["eos.thermo"]["sha256"],
                hashlib.sha256(dataset.source_file_bytes("eos.thermo")).hexdigest(),
            )
            self.assertEqual(
                provenance["matter_declaration"],
                "cold_beta_equilibrated",
            )
            self.assertEqual(provenance["source_positions"], list(range(6)))
            query_grid = provenance["query_grid"]
            self.assertEqual(query_grid["source"], "caller_supplied")
            self.assertIsNone(query_grid["requested_geometric_points"])
            self.assertFalse(query_grid["include_source_nodes_effective"])
            self.assertEqual(query_grid["final_points"], 2)
            self.assertEqual(
                query_grid["float64_little_endian_sha256"],
                hashlib.sha256(
                    np.asarray((0.1, 0.2), dtype="<f8").tobytes()
                ).hexdigest(),
            )
            self.assertEqual(summary["interpolation"]["query_grid"], query_grid)
            self.assertEqual(
                set(summary["official_cold_1d_quantities_not_applicable"]),
                {"10", "11", "13", "14", "16", "17"},
            )

    def test_native_q_profile_preserves_and_marks_pressure_reversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(
                Path(temporary),
                pressure_reversal=True,
            )
            dataset = load_compose_dataset(
                archive,
                model_id="native-seam",
                source_url="https://example.invalid/native-seam",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            profile = interpolate_compose_thermodynamics(cold_slice, points=301)
            self.assertEqual(profile.status, "available_with_diagnostics")
            codes = {item.code for item in profile.diagnostics}
            self.assertIn("pressure_not_strictly_increasing", codes)
            self.assertIn("sampled_pressure_gradient_nonpositive", codes)
            np.testing.assert_array_equal(
                profile.column("pressure_mev_fm3")[
                    profile.column("source_node_position") >= 0.0
                ],
                cold_slice.pressure_mev_fm3,
            )

    def test_native_q_interpolation_is_linear_in_raw_density_and_right_differentiated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            densities = (0.1, 0.2, 0.5, 1.0)
            q1 = np.asarray((1.0, 3.0, 6.0, 16.0))
            pressures = tuple(np.asarray(densities) * q1)
            archive = write_compose_fixture(
                Path(temporary),
                density_values=densities,
                energy_density_values=(100.0, 200.0, 500.0, 1000.0),
                pressure_values=pressures,
            )
            dataset = load_compose_dataset(
                archive,
                model_id="nonuniform",
                source_url="https://example.invalid/nonuniform",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            profile = interpolate_compose_thermodynamics(
                cold_slice,
                np.asarray((0.2, 0.35, 1.0)),
                include_source_nodes=False,
            )
            self.assertAlmostEqual(
                profile.column("q1_pressure_per_baryon_mev")[1],
                4.5,
                places=14,
            )
            # At the exact 0.2 source node CompOSE selects the interval to
            # the right: dQ1/dn=(6-3)/(0.5-0.2)=10, hence dP/dn=3+0.2*10=5.
            self.assertAlmostEqual(
                profile.column("dpressure_dnB_mev")[0],
                5.0,
                places=14,
            )
            # At the upper endpoint CompOSE necessarily selects the final
            # interval to the left: dQ1/dn=(16-6)/(1.0-0.5)=20.
            self.assertAlmostEqual(
                profile.column("dpressure_dnB_mev")[2],
                36.0,
                places=14,
            )
            for outside_domain in (
                np.asarray((0.09,)),
                np.asarray((1.01,)),
                np.asarray((np.nextafter(0.1, -np.inf),)),
                np.asarray((np.nextafter(1.0, np.inf),)),
            ):
                with self.subTest(outside_domain=float(outside_domain[0])):
                    with self.assertRaisesRegex(EosInputError, "forbids extrapolation"):
                        interpolate_compose_thermodynamics(
                            cold_slice,
                            outside_domain,
                            include_source_nodes=False,
                        )

    def test_native_density_validation_retains_causality_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary), coefficient=1.0e-3)
            dataset = load_compose_dataset(
                archive,
                model_id="acausal",
                source_url="https://example.invalid/acausal",
            )
            report = build_compose_eos(dataset).validate(points=17)
            self.assertFalse(report.passed)
            self.assertGreater(report.cs2_max, 1.0)
            self.assertIn("acausal", {item.code for item in report.issues})
            with self.assertRaisesRegex(EosInputError, "acausal"):
                solve_star(build_compose_eos(dataset), 100.0)
            diagnostic = solve_star(
                build_compose_eos(dataset),
                100.0,
                validation_mode="background_diagnostic",
            )
            self.assertEqual(
                diagnostic.eos_validation_status,
                "diagnostic_with_issues",
            )
            self.assertEqual(diagnostic.eos_validation_issues, ("acausal",))

    def test_nonmonotone_seam_is_reported_and_requires_explicit_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary), pressure_reversal=True)
            dataset = load_compose_dataset(
                archive,
                model_id="seam",
                source_url="https://example.invalid/seam",
            )
            cold_slice = dataset.cold_beta_equilibrium_slice(
                matter="cold_beta_equilibrated",
                includes_leptons=True,
            )
            report = cold_slice.report()
            self.assertFalse(report.continuous_barotrope_available)
            self.assertEqual(len(report.pressure_ordering_issues), 1)
            self.assertEqual(cold_slice.continuous_segments(), ((0, 2), (2, 6)))
            with self.assertRaisesRegex(EosInputError, "continuous invertible"):
                build_compose_eos(cold_slice)
            diagnostic = build_compose_eos(
                cold_slice,
                ordering_policy="diagnostic_monotone_subsequence",
            )
            selection = diagnostic.provenance()["selection"]
            self.assertEqual(
                selection["ordering_policy"], "diagnostic_monotone_subsequence"
            )
            self.assertEqual(selection["omitted_source_positions"], [2])
            self.assertFalse(
                selection["diagnostic_reduction_is_physical_transition_policy"]
            )
            self.assertTrue(diagnostic.validate(points=257).passed)
            keep_later = build_compose_eos(
                cold_slice,
                ordering_policy="diagnostic_keep_later_monotone_subsequence",
            )
            later_selection = keep_later.provenance()["selection"]
            self.assertEqual(later_selection["omitted_source_positions"], [1])
            later_validation = keep_later.validate(points=257)
            self.assertFalse(later_validation.passed)
            self.assertTrue(later_validation.issues)
            core_only = build_compose_eos(
                cold_slice,
                baryon_density_min_fm3=0.3,
            )
            self.assertEqual(len(core_only.baryon_density_fm3), 4)
            self.assertTrue(core_only.validate(points=257).passed)

    def test_compose_stars_use_log_pressure_boundary_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary))
            dataset = load_compose_dataset(
                archive,
                model_id="stellar",
                source_url="https://example.invalid/stellar",
            )
            eos = build_compose_eos(dataset)
            star = solve_star(eos, 100.0, retain_profile=True)
            tighter = solve_star(
                eos,
                100.0,
                config=StellarConfig(ode_rtol=3.0e-11, ode_atol=3.0e-13),
            )
            self.assertEqual(star.integration_variable, "log_pressure")
            self.assertEqual(star.boundary_pressure_mev_fm3, eos.pressure_min_mev_fm3)
            self.assertEqual(len(star.radius_profile_km), 300)
            self.assertTrue(np.all(np.diff(star.radius_profile_km) > 0.0))
            self.assertTrue(np.all(np.diff(star.mass_profile_msun) >= 0.0))
            self.assertAlmostEqual(star.mass_msun, tighter.mass_msun, delta=2.0e-8)
            self.assertAlmostEqual(star.radius_km, tighter.radius_km, delta=2.0e-8)

    def test_cli_reports_dataset_and_barotrope_as_separate_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary), pressure_reversal=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = eos_tool_main(
                    [
                        "validate",
                        "compose",
                        str(archive),
                        "--model-id",
                        "cli-seam",
                        "--source-url",
                        "https://example.invalid/cli-seam",
                        "--includes-leptons",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["dataset"]["status"], "parsed")
            self.assertEqual(
                payload["cold_slice"]["status"],
                "parsed_but_continuous_barotrope_unavailable",
            )
            self.assertEqual(
                payload["native_thermodynamics"]["status"],
                "available_with_diagnostics",
            )
            self.assertIn(
                "pressure_mev_fm3",
                payload["native_thermodynamics"]["columns"],
            )
            self.assertEqual(payload["barotrope"]["status"], "unavailable")

            required_output = io.StringIO()
            with contextlib.redirect_stdout(required_output):
                required_exit = eos_tool_main(
                    [
                        "validate",
                        "compose",
                        str(archive),
                        "--model-id",
                        "cli-seam",
                        "--source-url",
                        "https://example.invalid/cli-seam",
                        "--includes-leptons",
                        "--require-barotrope",
                        "--format",
                        "json",
                    ]
                )
            required_payload = json.loads(required_output.getvalue())
            self.assertEqual(required_exit, 2)
            self.assertTrue(required_payload["barotrope"]["required"])

            diagnostic_output = io.StringIO()
            with contextlib.redirect_stdout(diagnostic_output):
                diagnostic_exit = eos_tool_main(
                    [
                        "validate",
                        "compose",
                        str(archive),
                        "--model-id",
                        "cli-seam",
                        "--source-url",
                        "https://example.invalid/cli-seam",
                        "--includes-leptons",
                        "--ordering-policy",
                        "diagnostic_monotone_subsequence",
                        "--format",
                        "json",
                    ]
                )
            diagnostic_payload = json.loads(diagnostic_output.getvalue())
            self.assertEqual(diagnostic_exit, 0)
            selection = diagnostic_payload["barotrope"]["provenance"]["selection"]
            self.assertEqual(selection["omitted_source_positions"], [2])

    def test_cli_density_selection_applies_to_native_and_barotrope_layers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = write_compose_fixture(Path(temporary))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = eos_tool_main(
                    [
                        "validate",
                        "compose",
                        str(archive),
                        "--model-id",
                        "cli-domain",
                        "--source-url",
                        "https://example.invalid/cli-domain",
                        "--includes-leptons",
                        "--baryon-density-min-fm3",
                        "0.2",
                        "--baryon-density-max-fm3",
                        "0.5",
                        "--format",
                        "json",
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["density_selection"]["source_rows"], 4)
            self.assertEqual(
                payload["density_selection"]["requested_baryon_density_min_fm3"],
                0.2,
            )
            self.assertEqual(
                payload["density_selection"]["requested_baryon_density_max_fm3"],
                0.5,
            )
            self.assertEqual(
                payload["native_thermodynamics"]["provenance"][
                    "source_baryon_density_min_fm3"
                ],
                0.2,
            )
            self.assertEqual(
                payload["native_thermodynamics"]["provenance"][
                    "source_baryon_density_max_fm3"
                ],
                0.5,
            )
            barotrope_selection = payload["barotrope"]["provenance"]["selection"]
            self.assertEqual(
                barotrope_selection["selected_source_rows_before_ordering_policy"],
                4,
            )
            self.assertEqual(
                barotrope_selection["requested_baryon_density_min_fm3"],
                0.2,
            )
            self.assertEqual(
                barotrope_selection["requested_baryon_density_max_fm3"],
                0.5,
            )

            two_row_output = io.StringIO()
            with contextlib.redirect_stdout(two_row_output):
                two_row_exit = eos_tool_main(
                    [
                        "validate",
                        "compose",
                        str(archive),
                        "--model-id",
                        "cli-two-row-domain",
                        "--source-url",
                        "https://example.invalid/cli-two-row-domain",
                        "--includes-leptons",
                        "--baryon-density-min-fm3",
                        "0.2",
                        "--baryon-density-max-fm3",
                        "0.3",
                        "--format",
                        "json",
                    ]
                )
            two_row_payload = json.loads(two_row_output.getvalue())
            self.assertEqual(two_row_exit, 0)
            self.assertEqual(two_row_payload["density_selection"]["source_rows"], 2)
            self.assertTrue(
                two_row_payload["native_thermodynamics"]["status"].startswith(
                    "available"
                )
            )
            self.assertEqual(two_row_payload["barotrope"]["status"], "unavailable")

            text_output = io.StringIO()
            with contextlib.redirect_stdout(text_output):
                text_exit = eos_tool_main(
                    [
                        "validate",
                        "compose",
                        str(archive),
                        "--model-id",
                        "cli-domain-text",
                        "--source-url",
                        "https://example.invalid/cli-domain-text",
                        "--includes-leptons",
                        "--baryon-density-min-fm3",
                        "0.2",
                        "--baryon-density-max-fm3",
                        "0.5",
                    ]
                )
            self.assertEqual(text_exit, 0)
            text = text_output.getvalue()
            self.assertIn(
                "Selected native density path: 4 source rows over [0.2, 0.5] fm^-3",
                text,
            )
            self.assertIn("requested bounds: min=0.2, max=0.5 fm^-3", text)


if __name__ == "__main__":
    unittest.main()
