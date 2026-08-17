"""Plots of pressure, energy density, and sound speed."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

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
    set_pressure_yscale,
    source_indices,
    style_context,
    thermodynamic_view,
)
from neutron_star_eos.thermodynamics import ThermodynamicSeries, ThermodynamicView

COMPOSE_SOUND_SPEED_ROUTES = (
    "sound_speed_squared_curve_derivative",
    "sound_speed_squared_compose_thermodynamic",
    "sound_speed_squared_cold_beta_mu_derivative",
)

SOUND_SPEED_LABELS = {
    "sound_speed_squared_curve_derivative": r"native $dP/d\epsilon$",
    "sound_speed_squared_compose_thermodynamic": "CompOSE thermodynamic route",
    "sound_speed_squared_cold_beta_mu_derivative": (
        r"cold-$\beta$ chemical-potential route"
    ),
    "sound_speed_squared": r"continuous-barotrope $dP/d\epsilon$",
}


def unretained_native_indices(
    native: ThermodynamicSeries,
    retained: ThermodynamicSeries | None,
) -> np.ndarray:
    """Locate native source nodes omitted from a diagnostic reduction."""

    if retained is None:
        return np.asarray([], dtype=int)
    required = {"energy_density_mev_fm3", "pressure_mev_fm3"}
    if not required.issubset(native.column_names) or not required.issubset(
        retained.column_names
    ):
        return np.asarray([], dtype=int)
    native_indices = source_indices(native)
    if len(native_indices) <= retained.rows:
        return np.asarray([], dtype=int)
    native_epsilon = native.column("energy_density_mev_fm3")
    native_pressure = native.column("pressure_mev_fm3")
    retained_epsilon = retained.column("energy_density_mev_fm3")
    retained_pressure = retained.column("pressure_mev_fm3")
    omitted: list[int] = []
    for index in native_indices:
        epsilon_match = np.isclose(
            retained_epsilon,
            native_epsilon[index],
            rtol=5.0e-12,
            atol=1.0e-14,
        )
        pressure_match = np.isclose(
            retained_pressure,
            native_pressure[index],
            rtol=5.0e-12,
            atol=1.0e-14,
        )
        if not np.any(epsilon_match & pressure_match):
            omitted.append(int(index))
    return np.asarray(omitted, dtype=int)


def plot_pressure_energy(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
    show_source_nodes: bool = True,
    show_stellar_barotrope: bool = True,
) -> Axes:
    """Plot P(epsilon) while keeping native and retained data distinct."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.4, 4.4))
        native = series(view, "native_thermodynamics")
        source = series(view, "source_nodes")
        continuous = series(view, "continuous_barotrope")
        pressure_values: list[np.ndarray] = []
        epsilon_values: list[np.ndarray] = []

        if native is not None and {
            "energy_density_mev_fm3",
            "pressure_mev_fm3",
        }.issubset(native.column_names):
            epsilon = native.column("energy_density_mev_fm3")
            pressure = native.column("pressure_mev_fm3")
            ax.plot(
                epsilon,
                pressure,
                color=ORANGE,
                linestyle="--",
                linewidth=1.8,
                label="CompOSE native-Q reconstruction",
                zorder=2,
            )
            epsilon_values.append(epsilon)
            pressure_values.append(pressure)
            if show_source_nodes:
                indices = source_indices(native)
                if len(indices):
                    ax.scatter(
                        epsilon[indices],
                        pressure[indices],
                        s=20,
                        marker="o",
                        facecolors="none",
                        edgecolors=ORANGE,
                        linewidths=0.9,
                        alpha=0.85,
                        label="native source nodes",
                        zorder=6,
                    )

        if source is not None and {
            "energy_density_mev_fm3",
            "pressure_mev_fm3",
        }.issubset(source.column_names):
            epsilon = source.column("energy_density_mev_fm3")
            pressure = source.column("pressure_mev_fm3")
            epsilon_values.append(epsilon)
            pressure_values.append(pressure)
            if show_source_nodes:
                label = (
                    "stellar-barotrope source nodes"
                    if view.input_kind == "compose"
                    else "source nodes"
                )
                ax.scatter(
                    epsilon,
                    pressure,
                    s=12,
                    marker="s" if view.input_kind == "compose" else "o",
                    facecolors="none",
                    edgecolors=BLUE,
                    linewidths=0.75,
                    alpha=0.55 if view.input_kind == "compose" else 0.8,
                    label=label,
                    zorder=4,
                )

        if continuous is not None and show_stellar_barotrope:
            epsilon = continuous.column("energy_density_mev_fm3")
            pressure = continuous.column("pressure_mev_fm3")
            label = (
                "stellar-barotrope interpolation"
                if view.input_kind == "compose"
                else "continuous barotrope"
            )
            ax.plot(
                epsilon,
                pressure,
                color=BLUE,
                linewidth=2.0,
                label=label,
                zorder=3,
            )
            epsilon_values.append(epsilon)
            pressure_values.append(pressure)

        if not pressure_values:
            raise EosInputError("pressure-energy data are unavailable")

        if native is not None and show_source_nodes:
            omitted = unretained_native_indices(native, source)
            if len(omitted):
                epsilon = native.column("energy_density_mev_fm3")
                pressure = native.column("pressure_mev_fm3")
                ax.scatter(
                    epsilon[omitted],
                    pressure[omitted],
                    s=55,
                    marker="x",
                    color=VERMILLION,
                    linewidths=1.6,
                    label="not retained in diagnostic stellar reduction",
                    zorder=7,
                )

        set_positive_xscale(ax, epsilon_values)
        nonpositive_visible = set_pressure_yscale(ax, pressure_values)
        if nonpositive_visible:
            ax.text(
                0.02,
                0.98,
                "Non-positive pressure retained; symmetric-log scale",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize="small",
                color=VERMILLION,
            )
        ax.set_xlabel(r"Total energy density $\epsilon$ [MeV fm$^{-3}$]")
        ax.set_ylabel(r"Pressure $P$ [MeV fm$^{-3}$]")
        ax.set_title(view.model_name)
        ax.legend(loc="best")
        finish_axes(ax)
        return ax


def plot_sound_speed_squared(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
    compose_routes: Sequence[str] | None = None,
    include_stellar_barotrope: bool = False,
) -> Axes:
    """Plot available c_s^2 definitions with zero and causal references."""

    plt, _log_norm = require_matplotlib()
    view = thermodynamic_view(model, curve_points=curve_points)
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.4, 4.4))
        native = series(view, "native_thermodynamics")
        continuous = series(view, "continuous_barotrope")
        x_values: list[np.ndarray] = []
        plotted = 0

        if native is not None:
            routes = (
                COMPOSE_SOUND_SPEED_ROUTES
                if compose_routes is None
                else tuple(dict.fromkeys(compose_routes))
            )
            missing = [name for name in routes if name not in native.column_names]
            if missing:
                raise EosInputError(
                    "requested CompOSE sound-speed columns are unavailable: "
                    + ", ".join(missing)
                )
            epsilon = native.column("energy_density_mev_fm3")
            node_indices = source_indices(native)
            colors = (ORANGE, GREEN, PURPLE, SKY)
            styles = ("-", "--", ":", "-.")
            for position, route in enumerate(routes):
                values = native.column(route)
                ax.plot(
                    epsilon,
                    values,
                    color=colors[position % len(colors)],
                    linestyle=styles[position % len(styles)],
                    linewidth=1.8,
                    marker="o" if len(node_indices) else None,
                    markevery=node_indices.tolist() if len(node_indices) else None,
                    markersize=3.2,
                    label=SOUND_SPEED_LABELS.get(route, route),
                )
                plotted += 1
            x_values.append(epsilon)

        if continuous is not None and (native is None or include_stellar_barotrope):
            epsilon = continuous.column("energy_density_mev_fm3")
            values = continuous.column("sound_speed_squared")
            label = (
                "stellar-barotrope route"
                if native is not None
                else SOUND_SPEED_LABELS["sound_speed_squared"]
            )
            ax.plot(epsilon, values, color=BLUE, linewidth=2.0, label=label)
            x_values.append(epsilon)
            plotted += 1

        if not plotted:
            raise EosInputError("sound-speed data are unavailable")
        set_positive_xscale(ax, x_values)
        ax.axhspan(0.0, 1.0, color=GREEN, alpha=0.045, zorder=0)
        ax.axhline(0.0, color=GREY, linewidth=1.0, zorder=1)
        ax.axhline(1.0, color=VERMILLION, linestyle="--", linewidth=1.2, zorder=1)
        ax.text(
            0.99,
            1.0,
            r"causal reference $c_s^2=1$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize="small",
            color=VERMILLION,
        )
        ax.set_xlabel(r"Total energy density $\epsilon$ [MeV fm$^{-3}$]")
        ax.set_ylabel(r"Sound speed squared $c_s^2=dP/d\epsilon$")
        ax.set_title(view.model_name)
        ax.legend(loc="best")
        finish_axes(ax)
        return ax


_unretained_native_indices = unretained_native_indices
