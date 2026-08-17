"""Create one professional PNG per campaign diagnostic or comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.typing import RcKeyType

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from reference import _comparison_to_reference, _reference_selected_peak_side
from settings import COLORS, DERIVED_ROOT, FIGURE_ROOT, BranchData

from neutron_star_eos import EosInputError, EosModel, SequenceResult, StarResult
from neutron_star_eos.compose import ComposeMassRadiusReference
from neutron_star_eos.plotting import (
    plot_compose_closure_residuals,
    plot_compose_cold_residuals,
    plot_compose_free_energy_closure_residuals,
    plot_composition,
    plot_mass_profile,
    plot_phase_codes,
    plot_pressure_energy,
    plot_sequence_status,
    plot_sound_speed_squared,
)


def _save_ax(ax: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        ax.figure.savefig(
            temporary,
            format="png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        temporary.replace(path)
    finally:
        plt.close(ax.figure)
        temporary.unlink(missing_ok=True)


def _plot_style() -> dict[RcKeyType, Any]:
    return {
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titleweight": "bold",
    }


def _save_model_plots(
    model: EosModel,
    view: Any,
    sequence: SequenceResult,
    branch: BranchData,
    metrics: Mapping[str, Any],
    causal: Mapping[str, Any],
    reference: ComposeMassRadiusReference | None,
    figure_directory: Path,
    profile_stars: Mapping[str, StarResult],
) -> dict[str, Any]:
    created: list[str] = []
    required: list[str] = []
    skipped: dict[str, str] = {}

    def save(name: str, factory: Any) -> None:
        filename = f"{name}.png"
        required.append(filename)
        try:
            ax = factory()
        except (EosInputError, ValueError) as exc:
            skipped[name] = str(exc)
        else:
            _save_ax(ax, figure_directory / filename)
            created.append(filename)

    with plt.rc_context(_plot_style()):
        save(
            "pressure_energy",
            lambda: plot_pressure_energy(
                model, curve_points=3001, show_source_nodes=False
            ),
        )
        save(
            "pressure_energy_source_nodes",
            lambda: plot_pressure_energy(
                model, curve_points=3001, show_stellar_barotrope=False
            ),
        )
        save(
            "sound_speed_squared",
            lambda: plot_sound_speed_squared(
                model, curve_points=3001, include_stellar_barotrope=True
            ),
        )
        save(
            "closure_residuals",
            lambda: plot_compose_closure_residuals(
                model, curve_points=3001, include_free_energy=False
            ),
        )
        save(
            "free_energy_closure_residuals",
            lambda: plot_compose_free_energy_closure_residuals(
                model, curve_points=3001
            ),
        )
        save(
            "cold_condition_residuals",
            lambda: plot_compose_cold_residuals(model, curve_points=3001),
        )
        native = view.series_for("native_thermodynamics")
        abundance = [
            name
            for name in native.column_names
            if (
                name.startswith("composition_pair_")
                or (name.startswith("composition_quadruple_") and name.endswith("_Yav"))
            )
            and not name.endswith("_available")
        ]
        for group_index in range(0, len(abundance), 8):
            names = abundance[group_index : group_index + 8]
            save(
                f"composition_abundances_{group_index // 8 + 1:02d}",
                lambda names=names: plot_composition(model, quantities=names),
            )
        nuclear = [
            name
            for name in native.column_names
            if name.startswith("composition_quadruple_")
            and name.rsplit("_", 1)[-1] in {"Aav", "Zav", "Nav"}
            and not name.endswith("_available")
        ]
        for group_index in range(0, len(nuclear), 6):
            names = nuclear[group_index : group_index + 6]
            save(
                f"composition_nuclear_characteristics_{group_index // 6 + 1:02d}",
                lambda names=names: plot_composition(model, quantities=names),
            )
        if "phase_code" in native.column_names:
            save("phase_codes", lambda: plot_phase_codes(model))
        save("sequence_status", lambda: plot_sequence_status(sequence))
        for label, star in profile_stars.items():
            save(f"mass_profile_{label}", lambda star=star: plot_mass_profile(star))

        stop = branch.peak_index + 1
        figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
        ax.plot(
            branch.radius_km,
            branch.mass_msun,
            color=COLORS[0],
            linewidth=1.8,
            alpha=0.45,
            label="all calculated backgrounds",
        )
        ax.plot(
            branch.radius_km[:stop],
            branch.mass_msun[:stop],
            color=COLORS[0],
            linewidth=2.4,
            label="sampled pre-peak central-density segment",
        )
        peak = metrics["sampled_peak"]
        ax.scatter(
            [peak["radius_km"]],
            [peak["mass_msun"]],
            color=COLORS[1],
            marker="D",
            s=48,
            label="sampled hydrostatic peak",
            zorder=4,
        )
        if causal.get("mass_msun") is not None:
            ax.scatter(
                [causal["radius_km"]],
                [causal["mass_msun"]],
                color=COLORS[2],
                marker="^",
                s=52,
                label=r"positive-$c_s^2$, $c_s^2\leq1$ endpoint",
                zorder=4,
            )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title(model.model_name)
        ax.legend(loc="best")
        _save_ax(ax, figure_directory / "calculated_mass_radius.png")
        created.append("calculated_mass_radius.png")
        required.append("calculated_mass_radius.png")

        figure, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
        ax.plot(
            branch.baryon_density_fm3,
            branch.mass_msun,
            color=COLORS[0],
            linewidth=2.0,
        )
        ax.axvline(
            float(peak["central_baryon_density_fm3"]),
            color=COLORS[1],
            linestyle="--",
            label="sampled peak",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"Central baryon density $n_{B,c}$ [fm$^{-3}$]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title(f"{model.model_name}: central-density sequence")
        ax.legend(loc="best")
        _save_ax(ax, figure_directory / "mass_central_density.png")
        created.append("mass_central_density.png")
        required.append("mass_central_density.png")

        if reference is not None:
            ref_mass, ref_radius, _selection = _reference_selected_peak_side(reference)
            figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
            ax.scatter(
                reference.radius_km,
                reference.mass_msun,
                color=COLORS[4],
                s=18,
                label="source-order eos.mr points",
            )
            ax.set_xlabel("CompOSE reference radius [km]")
            ax.set_ylabel(r"CompOSE reference mass [$M_\odot$]")
            ax.set_title(f"{model.model_name}: independent eos.mr reference")
            ax.legend(loc="best")
            _save_ax(ax, figure_directory / "reference_mass_radius.png")
            created.append("reference_mass_radius.png")
            required.append("reference_mass_radius.png")

            figure, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
            ax.plot(
                branch.radius_km[:stop],
                branch.mass_msun[:stop],
                color=COLORS[0],
                linewidth=2.2,
                label="toolkit TOV calculation",
            )
            ax.scatter(
                ref_radius,
                ref_mass,
                color=COLORS[4],
                marker="o",
                facecolors="none",
                s=26,
                label="CompOSE eos.mr reference",
            )
            ax.set_xlabel("Radius [km]")
            ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
            ax.set_title(f"{model.model_name}: calculated vs independent reference")
            ax.legend(loc="best")
            _save_ax(ax, figure_directory / "calculated_vs_reference_mass_radius.png")
            created.append("calculated_vs_reference_mass_radius.png")
            required.append("calculated_vs_reference_mass_radius.png")

            comparison = _comparison_to_reference(branch, reference)
            fixed = comparison["fixed_mass_comparisons"]
            sample_mass = np.asarray([row["mass_msun"] for row in fixed], dtype=float)
            residual = np.asarray(
                [row["calculated_minus_reference_radius_km"] for row in fixed],
                dtype=float,
            )
            figure, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
            ax.plot(
                sample_mass,
                residual,
                color=COLORS[1],
                marker="o",
                linewidth=1.8,
            )
            ax.axhline(0.0, color="#555555", linewidth=1.0)
            ax.set_xlabel(r"Gravitational mass [$M_\odot$]")
            ax.set_ylabel(r"$R_{\rm TOV}-R_{\tt eos.mr}$ [km]")
            ax.set_title(f"{model.model_name}: fixed-mass reference residuals")
            _save_ax(ax, figure_directory / "reference_radius_residual.png")
            created.append("reference_radius_residual.png")
            required.append("reference_radius_residual.png")
    missing = sorted(set(required) - set(created))
    return {
        "created": created,
        "skipped": skipped,
        "required": required,
        "missing_required": missing,
        "required_coverage_passed": not missing,
    }


def _save_comparison_plots(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparison = FIGURE_ROOT / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    required = [
        "all_calculated_mass_radius.png",
        "catalogue_peak_mass_crosscheck.png",
        "catalogue_peak_radius_crosscheck.png",
        "catalogue_1_4_radius_crosscheck.png",
    ]
    created: list[str] = []
    with plt.rc_context(_plot_style()):
        figure, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for index, summary in enumerate(summaries):
            path = DERIVED_ROOT / str(summary["slug"]) / "sequence.csv"
            data = np.genfromtxt(
                path, delimiter=",", names=True, dtype=None, encoding="utf-8"
            )
            solved = np.asarray(data["status"] == "solved")
            mass = np.asarray(data["mass_msun"][solved], dtype=float)
            radius = np.asarray(data["radius_km"][solved], dtype=float)
            peak = int(np.argmax(mass))
            model_label = str(summary["model_id"])
            if summary["non_gating_findings"]["material_source_seam_systematic"]:
                model_label += " [conditional seam]"
            ax.plot(
                radius[: peak + 1],
                mass[: peak + 1],
                color=COLORS[index % len(COLORS)],
                linewidth=2.0,
                label=model_label,
            )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Gravitational mass [$M_\odot$]")
        ax.set_title("Calculated cold-CompOSE mass-radius comparison")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            ncols=1,
        )
        _save_ax(ax, comparison / "all_calculated_mass_radius.png")
        created.append("all_calculated_mass_radius.png")

        for key, label, filename in (
            (
                "sampled_peak_mass_msun",
                r"Sampled peak mass [$M_\odot$]",
                "catalogue_peak_mass_crosscheck.png",
            ),
            (
                "radius_at_sampled_peak_km",
                "Radius at sampled peak [km]",
                "catalogue_peak_radius_crosscheck.png",
            ),
            (
                "radius_at_1_4_msun_km",
                r"Radius at $1.4\,M_\odot$ [km]",
                "catalogue_1_4_radius_crosscheck.png",
            ),
        ):
            slugs = [str(item["slug"]) for item in summaries]
            values = [
                item["compose_catalogue_crosscheck"]["calculated_minus_benchmark"][key]
                for item in summaries
            ]
            deltas = [np.nan if value is None else float(value) for value in values]
            figure, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
            ax.bar(
                slugs,
                deltas,
                color=[COLORS[i % len(COLORS)] for i in range(len(slugs))],
            )
            ax.axhline(0.0, color="#333333", linewidth=1.0)
            ax.set_ylabel(f"Calculated minus catalogue: {label}")
            ax.set_title("CompOSE catalogue cross-check")
            ax.tick_params(axis="x", rotation=35)
            _save_ax(ax, comparison / filename)
            created.append(filename)
    missing = sorted(set(required) - set(created))
    return {
        "created": created,
        "required": required,
        "missing_required": missing,
        "required_coverage_passed": not missing,
    }
