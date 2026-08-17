"""Optional, source-aware Matplotlib plots for loaded EoS results.

The functions in this module are deliberately atomic: they draw on an
existing :class:`~matplotlib.axes.Axes` (or create one), return that axes, and
never call ``show()``, ``savefig()``, or a stellar solver.  Matplotlib remains
an optional dependency; importing this module does not import Matplotlib.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from neutron_star_eos.compose import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
)
from neutron_star_eos.eos import EosInputError
from neutron_star_eos.model import EosModel
from neutron_star_eos.stellar import SequenceResult, StarResult
from neutron_star_eos.thermodynamics import (
    ThermodynamicSeries,
    ThermodynamicView,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


_STYLE_PATH = Path(__file__).resolve().parent / "styles" / "neutron_star_eos.mplstyle"

# Okabe-Ito-inspired colors.  Meaning is reinforced with markers and line
# styles, so none of the plots relies on color perception alone.
_BLUE = "#0072B2"
_ORANGE = "#E69F00"
_GREEN = "#009E73"
_VERMILLION = "#D55E00"
_PURPLE = "#CC79A7"
_SKY = "#56B4E9"
_BLACK = "#1A1A1A"
_GREY = "#6B7280"

_COMPOSE_SOUND_SPEED_ROUTES = (
    "sound_speed_squared_curve_derivative",
    "sound_speed_squared_compose_thermodynamic",
    "sound_speed_squared_cold_beta_mu_derivative",
)

_SOUND_SPEED_LABELS = {
    "sound_speed_squared_curve_derivative": r"native $dP/d\epsilon$",
    "sound_speed_squared_compose_thermodynamic": "CompOSE thermodynamic route",
    "sound_speed_squared_cold_beta_mu_derivative": (
        r"cold-$\beta$ chemical-potential route"
    ),
    "sound_speed_squared": r"continuous-barotrope $dP/d\epsilon$",
}

_CLOSURE_LABELS = {
    "euler_normalized_residual": "Euler",
    "first_law_normalized_residual": "first law",
    "gibbs_duhem_normalized_residual": "Gibbs-Duhem",
    "free_energy_pressure_normalized_residual": "free-energy pressure",
    "free_energy_muB_normalized_residual": "free-energy chemical potential",
}

# Standard indices used by the downloaded CompOSE tables in this project.  A
# provider-defined or otherwise unverified code must never acquire a guessed
# physical interpretation in a figure legend.
_COMPOSE_PARTICLE_LABELS = {
    0: "electrons",
    1: "muons",
    10: "neutrons",
    11: "protons",
}

_ORDINARY_CLOSURE_NAMES = (
    "euler_normalized_residual",
    "first_law_normalized_residual",
    "gibbs_duhem_normalized_residual",
)
_FREE_ENERGY_CLOSURE_NAMES = (
    "free_energy_pressure_normalized_residual",
    "free_energy_muB_normalized_residual",
)


def _require_matplotlib() -> tuple[Any, Any]:
    """Import the optional plotting dependency only when a plot is requested."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Plotting requires the optional Matplotlib dependency. "
            "Install neutron-star-eos-toolkit with its 'plot' extra."
        ) from exc
    return plt, LogNorm


@contextmanager
def _style_context(plt: Any) -> Iterator[None]:
    if _STYLE_PATH.is_file():
        with plt.style.context(str(_STYLE_PATH)):
            yield
    else:  # Defensive fallback for an incorrectly packaged style asset.
        with plt.rc_context():
            yield


def _axes(plt: Any, ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    _figure, created = plt.subplots(figsize=figsize, constrained_layout=True)
    return created


def _view(
    model_or_view: EosModel | ThermodynamicView,
    *,
    curve_points: int,
) -> ThermodynamicView:
    if isinstance(model_or_view, ThermodynamicView):
        return model_or_view
    thermodynamics = getattr(model_or_view, "thermodynamics", None)
    if not callable(thermodynamics):
        raise TypeError("expected an EosModel or ThermodynamicView")
    result = thermodynamics(curve_points=curve_points)
    if not isinstance(result, ThermodynamicView):
        raise TypeError("model.thermodynamics() did not return ThermodynamicView")
    return result


def _series(view: ThermodynamicView, role: str) -> ThermodynamicSeries | None:
    try:
        return view.series_for(role)
    except KeyError:
        return None


def _source_indices(series: ThermodynamicSeries) -> np.ndarray:
    if "source_node_position" not in series.column_names:
        return np.asarray([], dtype=int)
    positions = series.column("source_node_position")
    return np.flatnonzero(np.isfinite(positions) & (positions >= 0.0))


def _sparse_markevery(
    indices: np.ndarray,
    *,
    maximum: int = 36,
    offset: int = 0,
) -> list[int] | None:
    """Return representative source-node markers without covering the curve."""

    source_indices = np.asarray(indices, dtype=int)
    if not len(source_indices):
        return None
    if len(source_indices) <= maximum:
        selected = source_indices[offset::2] if offset else source_indices
        if len(selected):
            return [int(item) for item in selected]
        return [int(source_indices[-1])]
    positions = np.linspace(0, len(source_indices) - 1, maximum, dtype=int)
    selected = source_indices[np.unique(positions)]
    if offset:
        selected = selected[offset::2]
    return (
        [int(item) for item in selected] if len(selected) else [int(source_indices[-1])]
    )


def _set_positive_xscale(ax: Axes, values: Sequence[np.ndarray]) -> None:
    finite = np.concatenate(
        [np.asarray(item, dtype=float)[np.isfinite(item)] for item in values]
    )
    if finite.size and np.all(finite > 0.0):
        ax.set_xscale("log")
        return
    nonzero = np.abs(finite[finite != 0.0])
    linthresh = float(np.min(nonzero) * 0.1) if nonzero.size else 1.0
    ax.set_xscale("symlog", linthresh=linthresh)


def _set_pressure_yscale(ax: Axes, values: Sequence[np.ndarray]) -> bool:
    finite = np.concatenate(
        [np.asarray(item, dtype=float)[np.isfinite(item)] for item in values]
    )
    if finite.size and np.all(finite > 0.0):
        ax.set_yscale("log")
        return False
    nonzero = np.abs(finite[finite != 0.0])
    linthresh = float(np.min(nonzero) * 0.1) if nonzero.size else 1.0
    ax.set_yscale("symlog", linthresh=linthresh)
    return True


def _finish(ax: Axes) -> None:
    ax.grid(True, which="major")
    ax.grid(False, which="minor")


def _unretained_native_indices(
    native: ThermodynamicSeries,
    retained: ThermodynamicSeries | None,
) -> np.ndarray:
    """Locate visible native nodes absent from a reduced stellar barotrope."""

    if retained is None:
        return np.asarray([], dtype=int)
    required = {"energy_density_mev_fm3", "pressure_mev_fm3"}
    if not required.issubset(native.column_names) or not required.issubset(
        retained.column_names
    ):
        return np.asarray([], dtype=int)
    native_indices = _source_indices(native)
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
    """Plot pressure against total energy density without hiding source data."""

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.4, 4.4))
        native = _series(view, "native_thermodynamics")
        source = _series(view, "source_nodes")
        continuous = _series(view, "continuous_barotrope")
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
                color=_ORANGE,
                linestyle="--",
                linewidth=1.8,
                label="CompOSE native-Q reconstruction",
                zorder=2,
            )
            epsilon_values.append(epsilon)
            pressure_values.append(pressure)
            if show_source_nodes:
                indices = _source_indices(native)
                if len(indices):
                    ax.scatter(
                        epsilon[indices],
                        pressure[indices],
                        s=20,
                        marker="o",
                        facecolors="none",
                        edgecolors=_ORANGE,
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
                    edgecolors=_BLUE,
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
                color=_BLUE,
                linewidth=2.0,
                label=label,
                zorder=3,
            )
            epsilon_values.append(epsilon)
            pressure_values.append(pressure)

        if not pressure_values:
            raise EosInputError("pressure-energy data are unavailable")

        if native is not None and show_source_nodes:
            omitted = _unretained_native_indices(native, source)
            if len(omitted):
                epsilon = native.column("energy_density_mev_fm3")
                pressure = native.column("pressure_mev_fm3")
                ax.scatter(
                    epsilon[omitted],
                    pressure[omitted],
                    s=55,
                    marker="x",
                    color=_VERMILLION,
                    linewidths=1.6,
                    label="not retained in diagnostic stellar reduction",
                    zorder=7,
                )

        _set_positive_xscale(ax, epsilon_values)
        nonpositive_visible = _set_pressure_yscale(ax, pressure_values)
        if nonpositive_visible:
            ax.text(
                0.02,
                0.98,
                "Non-positive pressure retained; symmetric-log scale",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize="small",
                color=_VERMILLION,
            )
        ax.set_xlabel(r"Total energy density $\epsilon$ [MeV fm$^{-3}$]")
        ax.set_ylabel(r"Pressure $P$ [MeV fm$^{-3}$]")
        ax.set_title(view.model_name)
        ax.legend(loc="best")
        _finish(ax)
        return ax


def plot_sound_speed_squared(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
    compose_routes: Sequence[str] | None = None,
    include_stellar_barotrope: bool = False,
) -> Axes:
    """Plot available definitions of ``c_s^2`` with zero and causal bounds."""

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.4, 4.4))
        native = _series(view, "native_thermodynamics")
        continuous = _series(view, "continuous_barotrope")
        x_values: list[np.ndarray] = []
        plotted = 0

        if native is not None:
            routes = (
                _COMPOSE_SOUND_SPEED_ROUTES
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
            source_indices = _source_indices(native)
            colors = (_ORANGE, _GREEN, _PURPLE, _SKY)
            styles = ("-", "--", ":", "-.")
            for position, route in enumerate(routes):
                values = native.column(route)
                ax.plot(
                    epsilon,
                    values,
                    color=colors[position % len(colors)],
                    linestyle=styles[position % len(styles)],
                    linewidth=1.8,
                    marker="o" if len(source_indices) else None,
                    markevery=source_indices.tolist() if len(source_indices) else None,
                    markersize=3.2,
                    label=_SOUND_SPEED_LABELS.get(route, route),
                )
                plotted += 1
            x_values.append(epsilon)

        if continuous is not None and (native is None or include_stellar_barotrope):
            epsilon = continuous.column("energy_density_mev_fm3")
            values = continuous.column("sound_speed_squared")
            label = (
                "stellar-barotrope route"
                if native is not None
                else _SOUND_SPEED_LABELS["sound_speed_squared"]
            )
            ax.plot(
                epsilon,
                values,
                color=_BLUE,
                linewidth=2.0,
                label=label,
            )
            x_values.append(epsilon)
            plotted += 1

        if not plotted:
            raise EosInputError("sound-speed data are unavailable")
        _set_positive_xscale(ax, x_values)
        ax.axhspan(0.0, 1.0, color=_GREEN, alpha=0.045, zorder=0)
        ax.axhline(0.0, color=_GREY, linewidth=1.0, zorder=1)
        ax.axhline(1.0, color=_VERMILLION, linestyle="--", linewidth=1.2, zorder=1)
        ax.text(
            0.99,
            1.0,
            r"causal reference $c_s^2=1$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize="small",
            color=_VERMILLION,
        )
        ax.set_xlabel(r"Total energy density $\epsilon$ [MeV fm$^{-3}$]")
        ax.set_ylabel(r"Sound speed squared $c_s^2=dP/d\epsilon$")
        ax.set_title(view.model_name)
        ax.legend(loc="best")
        _finish(ax)
        return ax


def _native_required(view: ThermodynamicView) -> ThermodynamicSeries:
    native = _series(view, "native_thermodynamics")
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
    """Plot ordinary CompOSE closure magnitudes as diagnostics, not repairs.

    ``include_free_energy=True`` retains the historical combined view.  Use
    :func:`plot_compose_free_energy_closure_residuals` for a focused plot of
    only the two free-energy closures.
    """

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    native = _native_required(view)
    names = list(_ORDINARY_CLOSURE_NAMES)
    if include_free_energy:
        names.extend(_FREE_ENERGY_CLOSURE_NAMES)
    return _plot_compose_closure_group(
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
    """Plot only the two normalized CompOSE free-energy closure magnitudes."""

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    native = _native_required(view)
    return _plot_compose_closure_group(
        plt,
        view,
        native,
        _FREE_ENERGY_CLOSURE_NAMES,
        ax=ax,
        title="CompOSE free-energy closure diagnostics",
    )


def _plot_compose_closure_group(
    plt: Any,
    view: ThermodynamicView,
    native: ThermodynamicSeries,
    names: tuple[str, ...],
    *,
    ax: Axes | None,
    title: str,
) -> Axes:
    missing = [name for name in names if name not in native.column_names]
    if missing:
        raise EosInputError(
            "CompOSE closure columns are unavailable: " + ", ".join(missing)
        )
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.8, 4.4))
        density = native.column("baryon_density_fm3")
        colors = (_BLUE, _ORANGE, _GREEN, _PURPLE, _SKY)
        styles = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
        source_indices = _source_indices(native)
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
                marker="o" if len(source_indices) else None,
                markevery=_sparse_markevery(
                    source_indices,
                    maximum=36,
                    offset=index % 2,
                ),
                markersize=2.6,
                markerfacecolor="white",
                markeredgewidth=0.7,
                label=_CLOSURE_LABELS[name],
            )
        tolerance = COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE
        ax.axhline(
            tolerance,
            color=_VERMILLION,
            linestyle="--",
            linewidth=1.1,
            label=f"diagnostic threshold ({tolerance:.0e})",
        )
        _set_positive_xscale(ax, [density])
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
                color=_GREY,
            )
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Normalized closure-residual magnitude")
        ax.set_title(f"{view.model_name}: {title}")
        ax.legend(loc="best")
        _finish(ax)
        return ax


def plot_compose_cold_residuals(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
) -> Axes:
    """Plot the native beta-equilibrium and zero-temperature residuals."""

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    native = _native_required(view)
    names = ("q5_beta_equilibrium_residual", "q6_minus_q7_zero_temperature_residual")
    missing = [name for name in names if name not in native.column_names]
    if missing:
        raise EosInputError(
            "CompOSE cold-condition columns are unavailable: " + ", ".join(missing)
        )
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.8, 4.4))
        density = native.column("baryon_density_fm3")
        source_indices = _source_indices(native)
        finite_residuals: list[np.ndarray] = []
        summaries: list[str] = []
        for index, (name, label, color, style, marker) in enumerate(
            (
                (
                    names[0],
                    r"$Q_5$ beta-equilibrium residual",
                    _BLUE,
                    "-",
                    "o",
                ),
                (
                    names[1],
                    r"$Q_6-Q_7$ zero-temperature residual",
                    _ORANGE,
                    "--",
                    "s",
                ),
            )
        ):
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
                marker=marker if len(source_indices) else None,
                markevery=_sparse_markevery(
                    source_indices,
                    maximum=36,
                    offset=index,
                ),
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
                color=_VERMILLION,
                linestyle=":" if sign < 0 else "--",
                linewidth=1.0,
                label=(
                    f"diagnostic threshold ({tolerance:.0e})"
                    if sign > 0
                    else "_nolegend_"
                ),
            )
        ax.axhline(0.0, color=_GREY, linewidth=0.9)
        _set_positive_xscale(ax, [density])
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
            color=_GREY,
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
        _finish(ax)
        return ax


def _composition_label(name: str) -> str:
    if name.startswith("composition_pair_"):
        raw_code = name.removeprefix("composition_pair_")
        try:
            particle_code = int(raw_code)
        except ValueError:
            return f"source-defined particle (code {raw_code})"
        species = _COMPOSE_PARTICLE_LABELS.get(particle_code)
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
    """Plot source-defined CompOSE composition quantities without zero fill."""

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    native = _native_required(view)
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
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(7.0, 4.6))
        density = native.column("baryon_density_fm3")
        source_indices = _source_indices(native)
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
                marker=markers[index % len(markers)] if len(source_indices) else None,
                markevery=_sparse_markevery(
                    source_indices,
                    maximum=36,
                    offset=index % 2,
                ),
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.8,
                label=_composition_label(name),
            )
        if partial:
            ax.text(
                0.02,
                0.03,
                "Missing source coverage is retained as gaps",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize="small",
                color=_VERMILLION,
            )
        _set_positive_xscale(ax, [density])
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Source-defined composition quantity")
        ax.set_title(f"{view.model_name}: CompOSE composition")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            ncols=1,
        )
        _finish(ax)
        return ax


def plot_phase_codes(
    model: EosModel | ThermodynamicView,
    *,
    ax: Axes | None = None,
    curve_points: int = 513,
) -> Axes:
    """Plot uninterpreted model-specific CompOSE phase codes."""

    plt, _log_norm = _require_matplotlib()
    view = _view(model, curve_points=curve_points)
    native = _native_required(view)
    if "phase_code" not in native.column_names:
        raise EosInputError("the selected CompOSE profile has no phase-code data")
    density = native.column("baryon_density_fm3")
    values = native.column("phase_code").copy()
    if "phase_code_available" in native.column_names:
        values[native.column("phase_code_available") <= 0.5] = np.nan
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.8, 3.8))
        ax.plot(
            density,
            values,
            color=_PURPLE,
            linewidth=1.7,
            marker="o",
            markersize=3.0,
            drawstyle="steps-post",
        )
        _set_positive_xscale(ax, [density])
        finite_codes = np.unique(values[np.isfinite(values)])
        if 0 < len(finite_codes) <= 12:
            ax.set_yticks(finite_codes)
        ax.set_xlabel(r"Baryon number density $n_B$ [fm$^{-3}$]")
        ax.set_ylabel("Model-specific phase code")
        ax.set_title(f"{view.model_name}: uninterpreted CompOSE phase codes")
        ax.text(
            0.02,
            0.97,
            "Codes are displayed, not interpreted as physical transitions",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color=_GREY,
        )
        _finish(ax)
        return ax


def plot_mass_profile(
    star: StarResult,
    *,
    ax: Axes | None = None,
    normalized: bool = False,
) -> Axes:
    """Plot an already-retained enclosed-mass profile for one background."""

    if not isinstance(star, StarResult):
        raise TypeError("plot_mass_profile expects StarResult")
    if not star.radius_profile_km or not star.mass_profile_msun:
        raise ValueError("star has no retained profile; solve with retain_profile=True")
    radius = np.asarray(star.radius_profile_km, dtype=float)
    mass = np.asarray(star.mass_profile_msun, dtype=float)
    if (
        radius.ndim != 1
        or mass.ndim != 1
        or radius.shape != mass.shape
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(mass))
    ):
        raise ValueError("retained stellar profile is malformed")
    if normalized:
        if star.radius_km <= 0.0 or star.mass_msun <= 0.0:
            raise ValueError("stellar boundary mass and radius must be positive")
        plot_radius = radius / star.radius_km
        plot_mass = mass / star.mass_msun
    else:
        plot_radius = radius
        plot_mass = mass

    plt, _log_norm = _require_matplotlib()
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.4, 4.3))
        ax.plot(
            plot_radius,
            plot_mass,
            color=_BLUE,
            linewidth=2.0,
            label="enclosed mass",
        )
        ax.scatter(
            [plot_radius[-1]],
            [plot_mass[-1]],
            marker="D",
            s=38,
            color=_VERMILLION,
            label="positive-pressure source boundary",
            zorder=4,
        )
        if normalized:
            ax.set_xlabel(r"Normalized radius $r/R_{\rm b}$")
            ax.set_ylabel(r"Normalized enclosed mass $M(r)/M_{\rm b}$")
        else:
            ax.set_xlabel(r"Radius $r$ [km]")
            ax.set_ylabel(r"Enclosed mass $M(r)$ [$M_\odot$]")
        ax.set_title(star.model_name or "Stellar background")
        ax.text(
            0.02,
            0.98,
            (
                "Truncated at the EoS lower-pressure boundary "
                f"($P={star.boundary_pressure_mev_fm3:.4g}$ MeV fm$^{{-3}}$; not $P=0$)"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color=_GREY,
        )
        ax.legend(loc="lower right")
        _finish(ax)
        return ax


def _solved_attempts(sequence: SequenceResult) -> list[tuple[int, Any, StarResult]]:
    return [
        (index, attempt, attempt.star)
        for index, attempt in enumerate(sequence.attempts)
        if attempt.star is not None
    ]


def plot_mass_radius(
    sequence: SequenceResult,
    *,
    ax: Axes | None = None,
    connect: bool = False,
    color_by_central_pressure: bool = True,
) -> Axes:
    """Plot solved source-boundary points while retaining sequence incompleteness."""

    if not isinstance(sequence, SequenceResult):
        raise TypeError("plot_mass_radius expects SequenceResult")
    if not sequence.attempts:
        raise ValueError("sequence has no requested attempts")
    solved = _solved_attempts(sequence)
    plt, LogNorm = _require_matplotlib()
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(6.5, 4.6))
        if solved:
            radius = np.asarray([star.radius_km for _i, _a, star in solved])
            mass = np.asarray([star.mass_msun for _i, _a, star in solved])
            pressure = np.asarray(
                [attempt.central_pressure_mev_fm3 for _i, attempt, _star in solved]
            )
            if color_by_central_pressure:
                if np.any(pressure <= 0.0):
                    raise ValueError(
                        "central pressures must be positive for log coloring"
                    )
                minimum = float(np.min(pressure))
                maximum = float(np.max(pressure))
                if minimum == maximum:
                    minimum *= 0.9
                    maximum *= 1.1
                points = ax.scatter(
                    radius,
                    mass,
                    c=pressure,
                    cmap="viridis",
                    norm=LogNorm(vmin=minimum, vmax=maximum),
                    marker="o",
                    s=36,
                    edgecolors="white",
                    linewidths=0.5,
                    label="solved backgrounds",
                    zorder=3,
                )
                ax.figure.colorbar(
                    points,
                    ax=ax,
                    label=r"Central pressure $P_c$ [MeV fm$^{-3}$]",
                )
            else:
                ax.scatter(
                    radius,
                    mass,
                    color=_BLUE,
                    marker="o",
                    s=36,
                    label="solved backgrounds",
                    zorder=3,
                )
            if connect:
                run: list[StarResult] = []
                for attempt in sequence.attempts:
                    if attempt.star is None:
                        if len(run) >= 2:
                            ax.plot(
                                [star.radius_km for star in run],
                                [star.mass_msun for star in run],
                                color=_GREY,
                                linewidth=1.0,
                                zorder=1,
                            )
                        run = []
                    else:
                        run.append(attempt.star)
                if len(run) >= 2:
                    ax.plot(
                        [star.radius_km for star in run],
                        [star.mass_msun for star in run],
                        color=_GREY,
                        linewidth=1.0,
                        zorder=1,
                    )
        solved_count = len(solved)
        requested_count = len(sequence.attempts)
        ax.text(
            0.02,
            0.98,
            f"{solved_count}/{requested_count} requested backgrounds solved",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color=_BLACK if solved_count == requested_count else _VERMILLION,
        )
        if not solved:
            ax.text(
                0.5,
                0.5,
                "No mass-radius point is available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=_VERMILLION,
            )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Mass at source boundary [$M_\odot$]")
        ax.set_title(sequence.model_name)
        if solved:
            ax.legend(loc="best")
        _finish(ax)
        return ax


def plot_sequence_status(
    sequence: SequenceResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot one visible outcome marker for every requested central pressure."""

    if not isinstance(sequence, SequenceResult):
        raise TypeError("plot_sequence_status expects SequenceResult")
    if not sequence.attempts:
        raise ValueError("sequence has no requested attempts")
    pressure = np.asarray(
        [attempt.central_pressure_mev_fm3 for attempt in sequence.attempts],
        dtype=float,
    )
    solved_mask = np.asarray(
        [attempt.star is not None for attempt in sequence.attempts], dtype=bool
    )
    plt, _log_norm = _require_matplotlib()
    with _style_context(plt):
        ax = _axes(plt, ax, figsize=(7.0, 3.6))
        if np.any(solved_mask):
            ax.scatter(
                pressure[solved_mask],
                np.ones(np.count_nonzero(solved_mask)),
                color=_GREEN,
                marker="o",
                s=38,
                label="solved",
                zorder=3,
            )
        if np.any(~solved_mask):
            ax.scatter(
                pressure[~solved_mask],
                np.zeros(np.count_nonzero(~solved_mask)),
                color=_VERMILLION,
                marker="x",
                s=48,
                linewidths=1.5,
                label="unavailable",
                zorder=3,
            )
        _set_positive_xscale(ax, [pressure])
        ax.set_yticks((0.0, 1.0), labels=("unavailable", "solved"))
        ax.set_ylim(-0.35, 1.35)
        ax.set_xlabel(r"Requested central pressure $P_c$ [MeV fm$^{-3}$]")
        ax.set_ylabel("Outcome")
        ax.set_title(f"{sequence.model_name}: sequence attempt status")
        failures = Counter(
            attempt.reason_code or "unclassified_failure"
            for attempt in sequence.attempts
            if attempt.star is None
        )
        if failures:
            summary = "Unavailable reason codes: " + ", ".join(
                f"{code} ({count})" for code, count in sorted(failures.items())
            )
            ax.text(
                0.02,
                0.04,
                summary,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize="small",
                color=_VERMILLION,
                wrap=True,
            )
        ax.legend(loc="upper right")
        _finish(ax)
        return ax


__all__ = [
    "plot_compose_closure_residuals",
    "plot_compose_cold_residuals",
    "plot_compose_free_energy_closure_residuals",
    "plot_composition",
    "plot_mass_profile",
    "plot_mass_radius",
    "plot_phase_codes",
    "plot_pressure_energy",
    "plot_sequence_status",
    "plot_sound_speed_squared",
]
