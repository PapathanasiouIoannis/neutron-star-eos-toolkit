from __future__ import annotations

import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from neutron_star_eos import EosModel
from neutron_star_eos.plotting import (
    plot_compose_closure_residuals,
    plot_compose_cold_residuals,
    plot_compose_free_energy_closure_residuals,
    plot_composition,
    plot_mass_profile,
    plot_mass_radius,
    plot_phase_codes,
    plot_pressure_energy,
    plot_sequence_status,
    plot_sound_speed_squared,
)
from neutron_star_eos.stellar import SequenceAttempt, SequenceResult, StarResult
from neutron_star_eos.thermodynamics import ThermodynamicSeries, ThermodynamicView

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def _series(
    role: str,
    label: str,
    columns: dict[str, np.ndarray],
) -> ThermodynamicSeries:
    return ThermodynamicSeries(
        role=role,
        label=label,
        columns=columns,
        units={name: "dimensionless" for name in columns},
        descriptions={name: name.replace("_", " ") for name in columns},
    )


def compose_view() -> ThermodynamicView:
    density = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5))
    epsilon = np.asarray((100.0, 200.0, 300.0, 400.0, 500.0))
    pressure = np.asarray((0.0, 40.0, 35.0, 100.0, 160.0))
    native = _series(
        "native_thermodynamics",
        "CompOSE native thermodynamics",
        {
            "baryon_density_fm3": density,
            "energy_density_mev_fm3": epsilon,
            "pressure_mev_fm3": pressure,
            "source_node_position": np.arange(5, dtype=float),
            "sound_speed_squared_curve_derivative": np.asarray(
                (-0.1, 0.2, 0.3, 1.1, 0.8)
            ),
            "sound_speed_squared_compose_thermodynamic": np.asarray(
                (0.0, 0.25, 0.35, 0.9, 0.7)
            ),
            "sound_speed_squared_cold_beta_mu_derivative": np.asarray(
                (0.05, 0.15, 0.4, 1.2, 0.6)
            ),
            "euler_normalized_residual": np.asarray(
                (0.0, 1.0e-9, 2.0e-7, 3.0e-8, 4.0e-8)
            ),
            "first_law_normalized_residual": np.asarray(
                (0.0, 2.0e-9, 1.0e-7, 2.0e-8, 5.0e-8)
            ),
            "gibbs_duhem_normalized_residual": np.asarray(
                (0.0, 3.0e-9, 8.0e-8, 4.0e-8, 6.0e-8)
            ),
            "free_energy_pressure_normalized_residual": np.asarray(
                (0.0, 4.0e-9, 7.0e-8, 5.0e-8, 7.0e-8)
            ),
            "free_energy_muB_normalized_residual": np.asarray(
                (0.0, 5.0e-9, 6.0e-8, 6.0e-8, 8.0e-8)
            ),
            "q5_beta_equilibrium_residual": np.asarray(
                (0.0, 1.0e-8, -2.0e-7, 3.0e-8, 4.0e-8)
            ),
            "q6_minus_q7_zero_temperature_residual": np.asarray(
                (0.0, -1.0e-8, 3.0e-7, 2.0e-8, 5.0e-8)
            ),
            "composition_pair_1": np.asarray((0.1, np.nan, 0.3, 0.4, 0.5)),
            "composition_pair_1_available": np.asarray((1.0, 0.0, 1.0, 1.0, 1.0)),
            "composition_pair_0": np.asarray((0.2, 0.2, 0.2, 0.2, 0.2)),
            "composition_pair_10": np.asarray((0.5, 0.5, 0.5, 0.5, 0.5)),
            "composition_pair_11": np.asarray((0.1, 0.1, 0.1, 0.1, 0.1)),
            "composition_pair_999": np.asarray((0.3, 0.3, 0.3, 0.3, 0.3)),
            "composition_quadruple_7_Yav": np.asarray((0.4, 0.4, 0.4, 0.4, 0.4)),
            "phase_code": np.asarray((1.0, 1.0, 2.0, np.nan, 2.0)),
            "phase_code_available": np.asarray((1.0, 1.0, 1.0, 0.0, 1.0)),
        },
    )
    retained = _series(
        "source_nodes",
        "Selected CompOSE stellar-barotrope nodes",
        {
            "baryon_density_fm3": density[[0, 1, 3, 4]],
            "energy_density_mev_fm3": epsilon[[0, 1, 3, 4]],
            "pressure_mev_fm3": pressure[[0, 1, 3, 4]],
            "source_node_position": np.arange(4, dtype=float),
        },
    )
    continuous = _series(
        "continuous_barotrope",
        "Evaluated continuous barotrope",
        {
            "energy_density_mev_fm3": np.geomspace(100.0, 500.0, 17),
            "pressure_mev_fm3": np.geomspace(1.0, 160.0, 17),
            "sound_speed_squared": np.linspace(0.1, 0.8, 17),
        },
    )
    return ThermodynamicView(
        "compose-plot-fixture",
        "compose",
        (native, retained, continuous),
    )


def zero_cold_view() -> ThermodynamicView:
    density = np.geomspace(1.0e-8, 1.0, 101)
    zeros = np.zeros_like(density)
    native = _series(
        "native_thermodynamics",
        "zero cold residuals",
        {
            "baryon_density_fm3": density,
            "source_node_position": np.arange(len(density), dtype=float),
            "q5_beta_equilibrium_residual": zeros,
            "q6_minus_q7_zero_temperature_residual": zeros,
        },
    )
    return ThermodynamicView("zero-cold-fixture", "compose", (native,))


class ViewOnlyModel:
    def __init__(self, view: ThermodynamicView) -> None:
        self.view = view
        self.thermodynamic_calls: list[int] = []

    def thermodynamics(self, *, curve_points: int = 513) -> ThermodynamicView:
        self.thermodynamic_calls.append(curve_points)
        return self.view

    def solve_star(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("plotting must not solve a star")

    def solve_sequence(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("plotting must not solve a sequence")


def star(
    pressure: float,
    mass: float,
    radius: float,
    *,
    retain_profile: bool = False,
) -> StarResult:
    return StarResult(
        central_pressure_mev_fm3=pressure,
        central_energy_density_mev_fm3=200.0,
        central_sound_speed_squared=0.4,
        mass_msun=mass,
        radius_km=radius,
        boundary_pressure_mev_fm3=0.01,
        boundary_energy_density_mev_fm3=1.0,
        boundary_status="truncated_at_eos_lower_pressure_not_vacuum",
        radius_profile_km=(0.1, radius / 2.0, radius) if retain_profile else (),
        mass_profile_msun=(1.0e-6, mass / 3.0, mass) if retain_profile else (),
        model_name="plot-star",
        eos_provenance_sha256="fixture",
    )


def partial_sequence() -> SequenceResult:
    first = star(10.0, 1.1, 11.0)
    last = star(100.0, 1.8, 12.5)
    return SequenceResult(
        model_name="plot-sequence",
        attempts=(
            SequenceAttempt(10.0, "solved", first, None, None),
            SequenceAttempt(
                30.0,
                "unavailable",
                None,
                "radius limit reached",
                "radius_limit_reached",
            ),
            SequenceAttempt(100.0, "solved", last, None, None),
        ),
        status="partial",
        boundary_status="truncated_at_eos_lower_pressure_not_vacuum",
    )


class OptionalDependencyTests(unittest.TestCase):
    def test_plotting_module_imports_when_matplotlib_is_blocked(self) -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        code = f"""
import builtins
import sys
sys.path.insert(0, {str(source_root)!r})
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('blocked for optional-dependency test')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import neutron_star_eos.plotting
print('imported-without-matplotlib')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("imported-without-matplotlib", completed.stdout)


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib plotting extra unavailable")
class PlottingTests(unittest.TestCase):
    def tearDown(self) -> None:
        assert plt is not None
        plt.close("all")

    def test_pressure_energy_preserves_native_order_zero_and_omission(self) -> None:
        model = ViewOnlyModel(compose_view())
        ax = plot_pressure_energy(model, curve_points=37)
        self.assertEqual(model.thermodynamic_calls, [37])
        self.assertEqual(ax.get_yscale(), "symlog")
        native = next(
            line
            for line in ax.lines
            if line.get_label() == "CompOSE native-Q reconstruction"
        )
        np.testing.assert_array_equal(
            native.get_ydata(), np.asarray((0.0, 40.0, 35.0, 100.0, 160.0))
        )
        collection_labels = {item.get_label() for item in ax.collections}
        self.assertIn("not retained in diagnostic stellar reduction", collection_labels)
        self.assertTrue(
            any(
                "Non-positive pressure retained" in item.get_text() for item in ax.texts
            )
        )
        native_nodes = next(
            item for item in ax.collections if item.get_label() == "native source nodes"
        )
        retained_nodes = next(
            item
            for item in ax.collections
            if item.get_label() == "stellar-barotrope source nodes"
        )
        self.assertGreater(native_nodes.get_zorder(), retained_nodes.get_zorder())
        self.assertTrue(np.all(native_nodes.get_sizes() > retained_nodes.get_sizes()))
        self.assertEqual(len(retained_nodes.get_facecolors()), 0)

    def test_analytical_sampling_stays_in_domain_and_reuses_axes(self) -> None:
        evaluations: list[np.ndarray] = []

        def pressure(epsilon: object) -> np.ndarray:
            values = np.asarray(epsilon, dtype=float)
            evaluations.append(values.copy())
            return 1.0e-3 * values**2

        model = EosModel.from_analytical(
            name="bounded-analytical",
            pressure_from_energy_density=pressure,
            sound_speed_squared_from_energy_density=lambda epsilon: (
                2.0e-3 * np.asarray(epsilon, dtype=float)
            ),
            energy_density_domain_mev_fm3=(1.0, 400.0),
            source="plotting unit test",
        )
        evaluations.clear()
        figure, supplied = plt.subplots()
        returned = plot_pressure_energy(model, ax=supplied, curve_points=41)
        self.assertIs(returned, supplied)
        self.assertIs(returned.figure, figure)
        self.assertTrue(evaluations)
        self.assertTrue(
            all(
                np.min(values) >= 1.0 and np.max(values) <= 400.0
                for values in evaluations
            )
        )
        self.assertEqual(returned.get_xscale(), "log")
        self.assertEqual(returned.get_yscale(), "log")

    def test_sound_speed_keeps_all_routes_and_physical_reference_lines(self) -> None:
        ax = plot_sound_speed_squared(compose_view())
        curve = next(
            line for line in ax.lines if line.get_label() == r"native $dP/d\epsilon$"
        )
        np.testing.assert_array_equal(
            curve.get_ydata(), np.asarray((-0.1, 0.2, 0.3, 1.1, 0.8))
        )
        route_labels = {line.get_label() for line in ax.lines}
        self.assertIn("CompOSE thermodynamic route", route_labels)
        self.assertIn(r"cold-$\beta$ chemical-potential route", route_labels)
        horizontal_values = [
            np.asarray(line.get_ydata(), dtype=float)
            for line in ax.lines
            if len(np.asarray(line.get_ydata())) == 2
        ]
        self.assertTrue(any(np.all(values == 0.0) for values in horizontal_values))
        self.assertTrue(any(np.all(values == 1.0) for values in horizontal_values))

    def test_compose_diagnostic_plots_render_with_thresholds(self) -> None:
        closure = plot_compose_closure_residuals(compose_view())
        combined = plot_compose_closure_residuals(
            compose_view(), include_free_energy=True
        )
        free_energy = plot_compose_free_energy_closure_residuals(compose_view())
        cold = plot_compose_cold_residuals(compose_view())
        self.assertEqual(closure.get_yscale(), "log")
        self.assertEqual(free_energy.get_yscale(), "log")
        self.assertEqual(cold.get_yscale(), "linear")
        self.assertTrue(
            any("diagnostic threshold" in line.get_label() for line in closure.lines)
        )
        self.assertTrue(
            any("diagnostic threshold" in line.get_label() for line in cold.lines)
        )
        ordinary_labels = {line.get_label() for line in closure.lines}
        free_energy_labels = {line.get_label() for line in free_energy.lines}
        combined_labels = {line.get_label() for line in combined.lines}
        self.assertNotIn("free-energy pressure", ordinary_labels)
        self.assertEqual(
            free_energy_labels
            - {
                label for label in free_energy_labels if "diagnostic threshold" in label
            },
            {"free-energy pressure", "free-energy chemical potential"},
        )
        self.assertIn("Euler", combined_labels)
        self.assertIn("free-energy pressure", combined_labels)
        self.assertTrue(any("exact-zero" in item.get_text() for item in closure.texts))
        closure.figure.canvas.draw()
        combined.figure.canvas.draw()
        free_energy.figure.canvas.draw()
        cold.figure.canvas.draw()

    def test_zero_cold_residuals_have_visible_linear_scale_and_summary(self) -> None:
        ax = plot_compose_cold_residuals(zero_cold_view())
        tolerance = 1.0e-7
        self.assertEqual(ax.get_yscale(), "linear")
        lower, upper = ax.get_ylim()
        self.assertLess(lower, -tolerance)
        self.assertGreater(upper, tolerance)
        all_text = " ".join(item.get_text() for item in ax.texts)
        self.assertIn("identically zero", all_text)
        diagnostic_lines = [line for line in ax.lines if "residual" in line.get_label()]
        self.assertEqual(
            {(line.get_linestyle(), line.get_marker()) for line in diagnostic_lines},
            {("-", "o"), ("--", "s")},
        )
        self.assertLessEqual(
            max(len(line.get_markevery()) for line in diagnostic_lines),
            36,
        )
        ax.figure.canvas.draw()

    def test_composition_preserves_missing_values_as_gaps(self) -> None:
        view = compose_view()
        before = (
            view.series_for("native_thermodynamics").column("composition_pair_1").copy()
        )
        ax = plot_composition(view)
        values = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        self.assertTrue(np.isnan(values[1]))
        self.assertFalse(np.any(values[np.isfinite(values)] == 0.0))
        np.testing.assert_equal(
            view.series_for("native_thermodynamics").column("composition_pair_1"),
            before,
        )
        disclosure = next(
            text for text in ax.texts if "missing\nsource coverage" in text.get_text()
        )
        self.assertGreater(disclosure.get_position()[0], 1.0)
        self.assertFalse(disclosure.get_clip_on())
        self.assertIsNotNone(disclosure.get_bbox_patch())
        self.assertNotIn("fraction", ax.get_ylabel().lower())
        self.assertLessEqual(len(ax.lines[0].get_markevery()), 36)
        ax.figure.canvas.draw()

    def test_composition_labels_verified_codes_and_marks_unknown_codes(self) -> None:
        ax = plot_composition(
            compose_view(),
            quantities=(
                "composition_pair_0",
                "composition_pair_1",
                "composition_pair_10",
                "composition_pair_11",
                "composition_pair_999",
                "composition_quadruple_7_Yav",
            ),
        )
        labels = {line.get_label() for line in ax.lines}
        self.assertIn("electrons (code 0)", labels)
        self.assertIn("muons (code 1)", labels)
        self.assertIn("neutrons (code 10)", labels)
        self.assertIn("protons (code 11)", labels)
        self.assertIn("source-defined particle (code 999)", labels)
        self.assertIn("source-defined nuclear group (code 7): Yav", labels)

    def test_phase_codes_are_explicitly_uninterpreted(self) -> None:
        ax = plot_phase_codes(compose_view())
        self.assertIn("not interpreted", ax.get_title().lower())
        self.assertIn("physical transitions", ax.get_title().lower())
        self.assertFalse(ax.texts)
        self.assertTrue(np.isnan(np.asarray(ax.lines[0].get_ydata())[3]))
        ax.figure.canvas.draw()

    def test_mass_profile_requires_retention_and_marks_source_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "retain_profile=True"):
            plot_mass_profile(star(50.0, 1.4, 12.0))
        result = star(50.0, 1.4, 12.0, retain_profile=True)
        ax = plot_mass_profile(result)
        profile = next(line for line in ax.lines if line.get_label() == "enclosed mass")
        self.assertEqual(float(profile.get_xdata()[-1]), result.radius_km)
        self.assertEqual(float(profile.get_ydata()[-1]), result.mass_msun)
        self.assertTrue(any("not $P=0$" in text.get_text() for text in ax.texts))
        all_text = " ".join(text.get_text().lower() for text in ax.texts)
        self.assertNotIn("maximum mass", all_text)
        self.assertNotIn("stable branch", all_text)

    def test_mass_radius_never_connects_across_failed_attempt(self) -> None:
        sequence = partial_sequence()
        ax = plot_mass_radius(sequence, connect=True)
        # The two solved points are separated by an unavailable attempt, so
        # no connecting Line2D is permitted.
        self.assertEqual(len(ax.lines), 0)
        self.assertTrue(
            any(
                "2/3 requested backgrounds solved" in text.get_text()
                for text in ax.texts
            )
        )
        self.assertIn("Source-boundary radius", ax.get_xlabel())
        rendered_text = " ".join(
            [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
            + [text.get_text() for text in ax.texts]
        ).lower()
        self.assertNotIn("maximum mass", rendered_text)
        self.assertNotIn("stable branch", rendered_text)

    def test_status_plot_has_one_marker_per_attempt_and_reason_codes(self) -> None:
        ax = plot_sequence_status(partial_sequence())
        marker_count = sum(
            len(collection.get_offsets()) for collection in ax.collections
        )
        self.assertEqual(marker_count, 3)
        self.assertTrue(
            any("radius_limit_reached (1)" in text.get_text() for text in ax.texts)
        )

    def test_plot_functions_never_show_save_or_solve(self) -> None:
        model = ViewOnlyModel(compose_view())
        with (
            mock.patch.object(plt, "show", side_effect=AssertionError("show called")),
            mock.patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=AssertionError("savefig called"),
            ),
        ):
            plot_pressure_energy(model)
            plot_sound_speed_squared(model)
        self.assertEqual(len(model.thermodynamic_calls), 2)

    def test_style_context_does_not_mutate_global_rcparams(self) -> None:
        before = matplotlib.rcParams["grid.alpha"]
        plot_pressure_energy(compose_view())
        self.assertEqual(matplotlib.rcParams["grid.alpha"], before)

    def test_png_svg_and_pdf_render(self) -> None:
        ax = plot_pressure_energy(compose_view())
        for format_name in ("png", "svg", "pdf"):
            output = io.BytesIO()
            ax.figure.savefig(output, format=format_name)
            self.assertGreater(len(output.getvalue()), 500)


if __name__ == "__main__":
    unittest.main()
