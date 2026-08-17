"""Shared, optional Matplotlib support for all plot families."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from neutron_star_eos.model import EosModel
from neutron_star_eos.thermodynamics import ThermodynamicSeries, ThermodynamicView

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any


STYLE_PATH = (
    Path(__file__).resolve().parents[1] / "styles" / "neutron_star_eos.mplstyle"
)

# Okabe-Ito-inspired colors.  Markers and line styles reinforce their meaning.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#1A1A1A"
GREY = "#6B7280"


def require_matplotlib() -> tuple[Any, Any]:
    """Import Matplotlib only when a plotting function is called."""

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
def style_context(plt: Any) -> Iterator[None]:
    """Apply the packaged style without mutating global Matplotlib settings."""

    if STYLE_PATH.is_file():
        with plt.style.context(str(STYLE_PATH)):
            yield
    else:
        with plt.rc_context():
            yield


def axes(plt: Any, ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    """Reuse a supplied axes or create one atomic plot axes."""

    if ax is not None:
        return ax
    _figure, created = plt.subplots(figsize=figsize, constrained_layout=True)
    return created


def thermodynamic_view(
    model_or_view: EosModel | ThermodynamicView,
    *,
    curve_points: int,
) -> ThermodynamicView:
    """Resolve the uniform read-only thermodynamic plotting view."""

    if isinstance(model_or_view, ThermodynamicView):
        return model_or_view
    thermodynamics = getattr(model_or_view, "thermodynamics", None)
    if not callable(thermodynamics):
        raise TypeError("expected an EosModel or ThermodynamicView")
    result = thermodynamics(curve_points=curve_points)
    if not isinstance(result, ThermodynamicView):
        raise TypeError("model.thermodynamics() did not return ThermodynamicView")
    return result


def series(view: ThermodynamicView, role: str) -> ThermodynamicSeries | None:
    """Return one named thermodynamic representation when available."""

    try:
        return view.series_for(role)
    except KeyError:
        return None


def source_indices(series_data: ThermodynamicSeries) -> np.ndarray:
    """Locate samples tied to original source-node positions."""

    if "source_node_position" not in series_data.column_names:
        return np.asarray([], dtype=int)
    positions = series_data.column("source_node_position")
    return np.flatnonzero(np.isfinite(positions) & (positions >= 0.0))


def sparse_markevery(
    indices: np.ndarray,
    *,
    maximum: int = 36,
    offset: int = 0,
) -> list[int] | None:
    """Select representative node markers without covering a curve."""

    source_nodes = np.asarray(indices, dtype=int)
    if not len(source_nodes):
        return None
    if len(source_nodes) <= maximum:
        selected = source_nodes[offset::2] if offset else source_nodes
        return (
            [int(item) for item in selected]
            if len(selected)
            else [int(source_nodes[-1])]
        )
    positions = np.linspace(0, len(source_nodes) - 1, maximum, dtype=int)
    selected = source_nodes[np.unique(positions)]
    if offset:
        selected = selected[offset::2]
    return (
        [int(item) for item in selected] if len(selected) else [int(source_nodes[-1])]
    )


def set_positive_xscale(ax: Axes, values: Sequence[np.ndarray]) -> None:
    """Use a log x-axis when possible and a symmetric log otherwise."""

    finite = np.concatenate(
        [np.asarray(item, dtype=float)[np.isfinite(item)] for item in values]
    )
    if finite.size and np.all(finite > 0.0):
        ax.set_xscale("log")
        return
    nonzero = np.abs(finite[finite != 0.0])
    linthresh = float(np.min(nonzero) * 0.1) if nonzero.size else 1.0
    ax.set_xscale("symlog", linthresh=linthresh)


def set_pressure_yscale(ax: Axes, values: Sequence[np.ndarray]) -> bool:
    """Set a pressure scale and report whether non-positive data are visible."""

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


def finish_axes(ax: Axes) -> None:
    """Apply the common major-grid presentation."""

    ax.grid(True, which="major")
    ax.grid(False, which="minor")


# Historical private aliases make the mechanical split safe for downstream use.
_require_matplotlib = require_matplotlib
_style_context = style_context
_axes = axes
_view = thermodynamic_view
_series = series
_source_indices = source_indices
_sparse_markevery = sparse_markevery
_set_positive_xscale = set_positive_xscale
_set_pressure_yscale = set_pressure_yscale
_finish = finish_axes
