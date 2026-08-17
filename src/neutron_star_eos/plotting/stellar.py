"""Plots of calculated stellar profiles and mass-radius sequences."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from neutron_star_eos.plotting.common import (
    BLACK,
    BLUE,
    GREEN,
    GREY,
    VERMILLION,
    Axes,
    axes,
    finish_axes,
    require_matplotlib,
    set_positive_xscale,
    style_context,
)
from neutron_star_eos.stellar import SequenceResult, StarResult


def plot_mass_profile(
    star: StarResult,
    *,
    ax: Axes | None = None,
    normalized: bool = False,
) -> Axes:
    """Plot an already-retained enclosed-mass profile."""

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

    plt, _log_norm = require_matplotlib()
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.4, 4.3))
        ax.plot(
            plot_radius,
            plot_mass,
            color=BLUE,
            linewidth=2.0,
            label="enclosed mass",
        )
        ax.scatter(
            [plot_radius[-1]],
            [plot_mass[-1]],
            marker="D",
            s=38,
            color=VERMILLION,
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
                f"($P={star.boundary_pressure_mev_fm3:.4g}$ MeV fm$^{{-3}}$; "
                "not $P=0$)"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            color=GREY,
        )
        ax.legend(loc="lower right")
        finish_axes(ax)
        return ax


def solved_attempts(sequence: SequenceResult) -> list[tuple[int, Any, StarResult]]:
    """Return solved attempts while retaining their original positions."""

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
    """Plot solved source-boundary points and disclose missing attempts."""

    if not isinstance(sequence, SequenceResult):
        raise TypeError("plot_mass_radius expects SequenceResult")
    if not sequence.attempts:
        raise ValueError("sequence has no requested attempts")
    solved = solved_attempts(sequence)
    plt, LogNorm = require_matplotlib()
    with style_context(plt):
        ax = axes(plt, ax, figsize=(6.5, 4.6))
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
                    color=BLUE,
                    marker="o",
                    s=36,
                    label="solved backgrounds",
                    zorder=3,
                )
            if connect:
                contiguous_run: list[StarResult] = []
                for attempt in sequence.attempts:
                    if attempt.star is None:
                        if len(contiguous_run) >= 2:
                            ax.plot(
                                [item.radius_km for item in contiguous_run],
                                [item.mass_msun for item in contiguous_run],
                                color=GREY,
                                linewidth=1.0,
                                zorder=1,
                            )
                        contiguous_run = []
                    else:
                        contiguous_run.append(attempt.star)
                if len(contiguous_run) >= 2:
                    ax.plot(
                        [item.radius_km for item in contiguous_run],
                        [item.mass_msun for item in contiguous_run],
                        color=GREY,
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
            color=BLACK if solved_count == requested_count else VERMILLION,
        )
        if not solved:
            ax.text(
                0.5,
                0.5,
                "No mass-radius point is available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=VERMILLION,
            )
        ax.set_xlabel("Source-boundary radius [km]")
        ax.set_ylabel(r"Mass at source boundary [$M_\odot$]")
        ax.set_title(sequence.model_name)
        if solved:
            ax.legend(loc="best")
        finish_axes(ax)
        return ax


def plot_sequence_status(
    sequence: SequenceResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot one outcome marker for every requested central pressure."""

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
    plt, _log_norm = require_matplotlib()
    with style_context(plt):
        ax = axes(plt, ax, figsize=(7.0, 3.6))
        if np.any(solved_mask):
            ax.scatter(
                pressure[solved_mask],
                np.ones(np.count_nonzero(solved_mask)),
                color=GREEN,
                marker="o",
                s=38,
                label="solved",
                zorder=3,
            )
        if np.any(~solved_mask):
            ax.scatter(
                pressure[~solved_mask],
                np.zeros(np.count_nonzero(~solved_mask)),
                color=VERMILLION,
                marker="x",
                s=48,
                linewidths=1.5,
                label="unavailable",
                zorder=3,
            )
        set_positive_xscale(ax, [pressure])
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
                color=VERMILLION,
                wrap=True,
            )
        ax.legend(loc="upper right")
        finish_axes(ax)
        return ax


_solved_attempts = solved_attempts
