"""Diagnostic plots for native CompOSE thermodynamics and composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neutron_star_eos.compose import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
)
from neutron_star_eos.eos import EosInputError
from neutron_star_eos.model import EosModel
from neutron_star_eos.plotting.common import (
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    SKY,
    VERMILLION,
    Axes,
    axes,
    finish_axes,
    require_matplotlib,
    series,
    set_positive_xscale,
    source_indices,
    sparse_markevery,
    style_context,
    thermodynamic_view,
)
from neutron_star_eos.thermodynamics import ThermodynamicSeries, ThermodynamicView

CLOSURE_LABELS = {
    "euler_normalized_residual": "Euler",
    "first_law_normalized_residual": "first law",
    "gibbs_duhem_normalized_residual": "Gibbs-Duhem",
    "free_energy_pressure_normalized_residual": "free-energy pressure",
    "free_energy_muB_normalized_residual": "free-energy chemical potential",
}

ORDINARY_CLOSURE_NAMES = (
    "euler_normalized_residual",
    "first_law_normalized_residual",
    "gibbs_duhem_normalized_residual",
)
FREE_ENERGY_CLOSURE_NAMES = (
    "free_energy_pressure_normalized_residual",
    "free_energy_muB_normalized_residual",
)

# Only codes verified for the standard CompOSE tables receive species names.
COMPOSE_PARTICLE_LABELS = {
    0: "electrons",
    1: "muons",
    10: "neutrons",
    11: "protons",
}


def native_thermodynamics(view: ThermodynamicView) -> ThermodynamicSeries:
    """Return native CompOSE thermodynamics required by diagnostic plots."""

    native = series(view, "native_thermodynamics")
    if native is None:
        raise EosInputError("this plot requires native CompOSE thermodynamics")
    if "baryon_density_fm3" not in native.column_names:
        raise EosInputError("native baryon-density data are unavailable")
    return native


def plot_compose_closure_residuals(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
    include_free_energy: bool = False,
) -> Axes:
    """Plot ordinary closure magnitudes as diagnostics, not repairs."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    native = native_thermodynamics(view)
    names = list(ORDINARY_CLOSURE_NAMES)
    if include_free_energy:
        names.extend(FREE_ENERGY_CLOSURE_NAMES)
    return plot_closure_group(
        plt,
        view,
        native,
        tuple(names),
        ax=ax,
        title="CompOSE closure diagnostics",
    )


def plot_compose_free_energy_closure_residuals(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
) -> Axes:
    """Plot the two normalized free-energy closure magnitudes."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    native = native_thermodynamics(view)
    return plot_closure_group(
        plt,
        view,
        native,
        FREE_ENERGY_CLOSURE_NAMES,
        ax=ax,
        title="CompOSE free-energy closure diagnostics",
    )


def plot_closure_group(
    plt: Any,
    view: ThermodynamicView,
    native: ThermodynamicSeries,
    names: tuple[str, ...],
    *,
    ax: Axes | None,
    title: str,
) -> Axes:
    """Draw one selected family of non-negative normalized residuals."""

    missing = [name for name in names if name not in native.column_names]
    if missing:
        raise EosInputError(
            "CompOSE closure columns are unavailable: " + ", ".join(missing)
        )
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.8, 4.4))
        density = native.column("baryon_density_fm3")
        colors = (BLUE, ORANGE, GREEN, PURPLE, SKY)
        styles = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
        node_indices = source_indices(native)
        exact_zero_count = 0
        for index, name in enumerate(names):
            values = native.column(name)
            finite = values[np.isfinite(values)]
            if np.any(finite < 0.0):
                raise EosInputError(
                    f"CompOSE normalized closure column {name} contains negative values"
                )
            exact_zero_count += int(np.count_nonzero(finite == 0.0))
            plotted_values = np.where(values > 0.0, values, np.nan)
            ax.plot(
                density,
                plotted_values,
                color=colors[index],
                linestyle=styles[index],
                linewidth=1.7,
                marker="o" if len(node_indices) else None,
                markevery=sparse_markevery(node_indices, maximum=36, offset=index % 2),
                markersize=2.6,
                markerfacecolor="white",
                markeredgewidth=0.7,
                label=CLOSURE_LABELS[name],
            )
        tolerance = COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE
        ax.axhline(
            tolerance,
            color=VERMILLION,
            linestyle="--",
            linewidth=1.1,
            label=f"diagnostic threshold ({tolerance:.0e})",
        )
        set_positive_xscale(ax, [density])
        ax.set_yscale("log")
        if exact_zero_count:
            ax.text(
                0.02,
                0.03,
                (
                    f"Log magnitude; {exact_zero_count} exact-zero sample"
                    f"{'s' if exact_zero_count != 1 else ''} omitted"
                ),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize="small",
                color=GREY,
            )
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Normalized closure-residual magnitude")
        ax.set_title(f"{view.model_name}: {title}")
        ax.legend(loc="best")
        finish_axes(ax)
        return ax


def plot_compose_cold_residuals(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
) -> Axes:
    """Plot beta-equilibrium and zero-temperature residuals."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    native = native_thermodynamics(view)
    names = ("q5_beta_equilibrium_residual", "q6_minus_q7_zero_temperature_residual")
    missing = [name for name in names if name not in native.column_names]
    if missing:
        raise EosInputError(
            "CompOSE cold-condition columns are unavailable: " + ", ".join(missing)
        )
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.8, 4.4))
        density = native.column("baryon_density_fm3")
        node_indices = source_indices(native)
        finite_residuals: list[np.ndarray] = []
        summaries: list[str] = []
        plot_specs = (
            (names[0], r"$Q_5$ beta-equilibrium residual", BLUE, "-", "o"),
            (
                names[1],
                r"$Q_6-Q_7$ zero-temperature residual",
                ORANGE,
                "--",
                "s",
            ),
        )
        for index, (name, label, color, style, marker) in enumerate(plot_specs):
            values = native.column(name)
            finite = values[np.isfinite(values)]
            finite_residuals.append(finite)
            short_label = r"$Q_5$" if index == 0 else r"$Q_6-Q_7$"
            if finite.size:
                maximum = float(np.max(np.abs(finite)))
                suffix = " (identically zero)" if maximum == 0.0 else ""
                summaries.append(f"max |{short_label}| = {maximum:.2e}{suffix}")
            else:
                summaries.append(f"max |{short_label}| unavailable")
            ax.plot(
                density,
                values,
                color=color,
                linestyle=style,
                linewidth=1.8,
                marker=marker if len(node_indices) else None,
                markevery=sparse_markevery(node_indices, maximum=36, offset=index),
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=label,
                zorder=3 + index,
            )
        tolerance = COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE
        for sign in (-1.0, 1.0):
            ax.axhline(
                sign * tolerance,
                color=VERMILLION,
                linestyle=":" if sign < 0 else "--",
                linewidth=1.0,
                label=(
                    f"diagnostic threshold ({tolerance:.0e})"
                    if sign > 0
                    else "_nolegend_"
                ),
            )
        ax.axhline(0.0, color=GREY, linewidth=0.9)
        set_positive_xscale(ax, [density])
        finite_values = np.concatenate(finite_residuals)
        observed_minimum = float(np.min(finite_values)) if finite_values.size else 0.0
        observed_maximum = float(np.max(finite_values)) if finite_values.size else 0.0
        visible_minimum = min(-tolerance, observed_minimum)
        visible_maximum = max(tolerance, observed_maximum)
        padding = 0.06 * (visible_maximum - visible_minimum)
        ax.set_yscale("linear")
        ax.set_ylim(visible_minimum - padding, visible_maximum + padding)
        ax.text(
            0.02,
            0.98,
            "\n".join(summaries),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color=GREY,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=8,
        )
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Dimensionless cold-condition residual")
        ax.set_title(f"{view.model_name}: CompOSE cold-condition diagnostics")
        ax.legend(loc="best")
        finish_axes(ax)
        return ax


def composition_label(name: str) -> str:
    """Give verified particle codes names and disclose unknown source codes."""

    if name.startswith("composition_pair_"):
        raw_code = name.removeprefix("composition_pair_")
        try:
            particle_code = int(raw_code)
        except ValueError:
            return f"source-defined particle (code {raw_code})"
        species = COMPOSE_PARTICLE_LABELS.get(particle_code)
        if species is None:
            return f"source-defined particle (code {particle_code})"
        return f"{species} (code {particle_code})"
    if name.startswith("composition_quadruple_"):
        remainder = name.removeprefix("composition_quadruple_")
        group_code, separator, quantity = remainder.partition("_")
        suffix = f": {quantity}" if separator else ""
        return f"source-defined nuclear group (code {group_code}){suffix}"
    return name.replace("_", " ")


def plot_composition(
    model: EosModel | ThermodynamicView,
    *,
    quantities: Sequence[str] | str | None = None,
    ax: Axes | None = None,
    curve_points: int = 513,
) -> Axes:
    """Plot source-defined composition quantities without zero-filling gaps."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    native = native_thermodynamics(view)
    candidates = tuple(
        name
        for name in native.column_names
        if (
            name.startswith("composition_pair_")
            or (name.startswith("composition_quadruple_") and name.endswith("_Yav"))
        )
        and not name.endswith("_available")
    )
    if quantities is None:
        selected = candidates
        if len(selected) > 8:
            raise EosInputError(
                "more than eight composition quantities are available; "
                "select quantities explicitly to avoid an unreadable plot"
            )
    elif isinstance(quantities, str):
        selected = (quantities,)
    else:
        selected = tuple(dict.fromkeys(quantities))
    if not selected:
        raise EosInputError(
            "the selected CompOSE profile has no composition quantities"
        )
    invalid = [
        name
        for name in selected
        if name not in native.column_names
        or not name.startswith("composition_")
        or name.endswith("_available")
    ]
    if invalid:
        raise EosInputError(
            "requested composition quantities are unavailable: " + ", ".join(invalid)
        )
    with style_context(plt):
        ax = axes(plt, ax, figsize=(7.0, 4.6))
        density = native.column("baryon_density_fm3")
        node_indices = source_indices(native)
        partial = False
        markers = ("o", "s", "^", "D", "v", "P", "X", "*")
        styles = ("-", "--", "-.", ":")
        for index, name in enumerate(selected):
            values = native.column(name).copy()
            availability_name = f"{name}_available"
            if availability_name in native.column_names:
                available = native.column(availability_name) > 0.5
                partial = partial or bool(np.any(~available))
                values[~available] = np.nan
            partial = partial or bool(np.any(~np.isfinite(values)))
            ax.plot(
                density,
                values,
                linewidth=1.7,
                linestyle=styles[index % len(styles)],
                marker=markers[index % len(markers)] if len(node_indices) else None,
                markevery=sparse_markevery(node_indices, maximum=36, offset=index % 2),
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=composition_label(name),
            )
        if partial:
            ax.text(
                1.01,
                0.03,
                "Gaps indicate missing\nsource coverage",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize="small",
                color=VERMILLION,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                    "pad": 2.5,
                },
                clip_on=False,
            )
        set_positive_xscale(ax, [density])
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Source-defined composition quantity")
        ax.set_title(f"{view.model_name}: CompOSE composition")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            ncols=1,
        )
        finish_axes(ax)
        return ax


def plot_phase_codes(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
) -> Axes:
    """Plot uninterpreted model-specific CompOSE phase codes."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    native = native_thermodynamics(view)
    if "phase_code" not in native.column_names:
        raise EosInputError("the selected CompOSE profile has no phase-code data")
    density = native.column("baryon_density_fm3")
    values = native.column("phase_code").copy()
    if "phase_code_available" in native.column_names:
        values[native.column("phase_code_available") <= 0.5] = np.nan
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.8, 3.8))
        ax.plot(
            density,
            values,
            color=PURPLE,
            linewidth=1.7,
            marker="o",
            markersize=3.0,
            drawstyle="steps-post",
        )
        set_positive_xscale(ax, [density])
        finite_codes = np.unique(values[np.isfinite(values)])
        if 0 < len(finite_codes) <= 12:
            ax.set_yticks(finite_codes)
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Model-specific phase code")
        ax.set_title(
            f"{view.model_name}: CompOSE phase codes\n"
            "Source-defined codes; not interpreted as physical transitions"
        )
        finish_axes(ax)
        return ax


_native_required = native_thermodynamics
_plot_compose_closure_group = plot_closure_group
_composition_label = composition_label
