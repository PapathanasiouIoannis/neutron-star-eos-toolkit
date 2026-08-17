"""Select and diagnose one cold beta-equilibrium path from CompOSE data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from neutron_star_eos.compose.records import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
    ComposeDiagnostic,
    ComposeOrderingIssue,
    ComposeSliceReport,
    ComposeThermodynamicRow,
)
from neutron_star_eos.eos import EosInputError

if TYPE_CHECKING:
    from neutron_star_eos.compose.reader import ComposeDataset


def ordering_issues(
    values: np.ndarray,
    baryon_density_fm3: np.ndarray,
) -> tuple[ComposeOrderingIssue, ...]:
    """Return every adjacent non-increasing source-row pair."""

    return tuple(
        ComposeOrderingIssue(
            int(index),
            int(index) + 1,
            float(baryon_density_fm3[int(index)]),
            float(baryon_density_fm3[int(index) + 1]),
            float(values[int(index)]),
            float(values[int(index) + 1]),
        )
        for index in np.flatnonzero(np.diff(values) <= 0.0)
    )


class ComposeColdSlice:
    """One explicit cold beta-equilibrium path before interpolation."""

    def __init__(
        self,
        *,
        dataset: ComposeDataset,
        rows: tuple[ComposeThermodynamicRow, ...],
        phase_codes: tuple[int | None, ...] | None,
        matter_declaration: str,
        source_positions: tuple[int, ...] | None = None,
    ) -> None:
        self.dataset = dataset
        self.matter_declaration = matter_declaration
        self.rows = rows
        self.phase_codes = phase_codes
        resolved_positions = (
            tuple(range(len(rows))) if source_positions is None else source_positions
        )
        if (
            len(resolved_positions) != len(rows)
            or any(value < 0 for value in resolved_positions)
            or any(
                right <= left
                for left, right in zip(resolved_positions, resolved_positions[1:])
            )
        ):
            raise EosInputError(
                "CompOSE source positions must align with rows and increase strictly"
            )
        self.source_positions = resolved_positions
        density_by_index = dict(
            zip(dataset.baryon_density.indices, dataset.baryon_density.values)
        )
        self._baryon_density = np.asarray(
            [density_by_index[row.baryon_density_index] for row in rows], dtype=float
        )
        self._q = np.asarray([row.q_values for row in rows], dtype=float)
        self._additional_values = tuple(row.additional_values for row in rows)
        self._baryon_density.setflags(write=False)
        self._q.setflags(write=False)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ComposeColdSlice is immutable")
        object.__setattr__(self, name, value)

    @staticmethod
    def _view(values: np.ndarray) -> np.ndarray:
        result = values.view()
        result.setflags(write=False)
        return result

    @property
    def baryon_density_fm3(self) -> np.ndarray:
        return self._view(self._baryon_density)

    @property
    def q_values(self) -> np.ndarray:
        return self._view(self._q)

    @property
    def additional_values(self) -> tuple[tuple[float, ...], ...]:
        return self._additional_values

    @property
    def pressure_mev_fm3(self) -> np.ndarray:
        return self._view(self._baryon_density * self._q[:, 0])

    @property
    def baryon_chemical_potential_mev(self) -> np.ndarray:
        return self._view(self.dataset.neutron_mass_mev * (1.0 + self._q[:, 2]))

    @property
    def energy_density_mev_fm3(self) -> np.ndarray:
        return self._view(
            self._baryon_density * self.dataset.neutron_mass_mev * (1.0 + self._q[:, 6])
        )

    def report(self) -> ComposeSliceReport:
        """Diagnose closure, cold conditions, and adjacent ordering."""

        pressure = np.asarray(self.pressure_mev_fm3)
        epsilon = np.asarray(self.energy_density_mev_fm3)
        chemical_potential = np.asarray(self.baryon_chemical_potential_mev)
        if (
            np.any(~np.isfinite(pressure))
            or np.any(~np.isfinite(epsilon))
            or np.any(~np.isfinite(chemical_potential))
            or np.any(pressure < 0.0)
            or np.any(epsilon <= 0.0)
            or np.any(chemical_potential <= 0.0)
        ):
            raise EosInputError(
                "selected CompOSE path has nonfinite, negative-pressure, or "
                "nonpositive energy/chemical-potential values"
            )
        euler_residual = pressure - (
            self._baryon_density * chemical_potential - epsilon
        )
        scale = np.maximum.reduce(
            (
                np.abs(pressure),
                np.abs(self._baryon_density * chemical_potential),
                np.abs(epsilon),
            )
        )
        if np.any(scale <= 0.0) or np.any(~np.isfinite(scale)):
            raise EosInputError(
                "selected CompOSE path has an invalid Euler normalization"
            )
        euler_max = float(np.max(np.abs(euler_residual) / scale))
        q5_max = float(np.max(np.abs(self._q[:, 4])))
        q67_max = float(np.max(np.abs(self._q[:, 5] - self._q[:, 6])))
        pressure_issues = ordering_issues(pressure, self._baryon_density)
        energy_issues = ordering_issues(epsilon, self._baryon_density)
        diagnostics: list[ComposeDiagnostic] = []
        if np.any(pressure == 0.0):
            diagnostics.append(
                ComposeDiagnostic(
                    "zero_pressure_source_node",
                    "barotrope_blocker",
                    "one or more source nodes have P=0; native thermodynamics "
                    "remain available but the positive-pressure logarithmic "
                    "barotrope does not",
                )
            )
        checks = (
            (
                euler_max > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
                "cold_euler_closure_residual",
                f"maximum normalized residual {euler_max:.12g} exceeds the "
                "diagnostic threshold",
            ),
            (
                q5_max > COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
                "beta_equilibrium_Q5_residual",
                f"maximum |Q5| {q5_max:.12g} exceeds the diagnostic threshold",
            ),
            (
                q67_max > COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
                "zero_temperature_Q6_minus_Q7_residual",
                f"maximum |Q6-Q7| {q67_max:.12g} exceeds the diagnostic threshold",
            ),
        )
        for condition, code, message in checks:
            if condition:
                diagnostics.append(ComposeDiagnostic(code, "warning", message))
        if pressure_issues:
            diagnostics.append(
                ComposeDiagnostic(
                    "pressure_not_strictly_increasing",
                    "barotrope_blocker",
                    f"{len(pressure_issues)} adjacent source-row pair(s) require "
                    "an explicit seam or transition policy",
                )
            )
        if energy_issues:
            diagnostics.append(
                ComposeDiagnostic(
                    "energy_density_not_strictly_increasing",
                    "barotrope_blocker",
                    f"{len(energy_issues)} adjacent source-row pair(s) prevent "
                    "continuous inversion",
                )
            )
        missing_phase_codes = 0
        phase_changes = 0
        if self.phase_codes is not None:
            missing_phase_codes = sum(value is None for value in self.phase_codes)
            phase_changes = sum(
                left is not None and right is not None and left != right
                for left, right in zip(self.phase_codes, self.phase_codes[1:])
            )
        return ComposeSliceReport(
            rows=len(self.rows),
            euler_maximum_normalized_residual=euler_max,
            q5_maximum_absolute_residual=q5_max,
            q6_minus_q7_maximum_absolute_residual=q67_max,
            pressure_ordering_issues=pressure_issues,
            energy_density_ordering_issues=energy_issues,
            phase_code_changes=phase_changes,
            missing_phase_codes=missing_phase_codes,
            diagnostics=tuple(diagnostics),
        )

    def selected_domain(
        self,
        *,
        baryon_density_min_fm3: float | None = None,
        baryon_density_max_fm3: float | None = None,
        minimum_rows: int = 4,
    ) -> ComposeColdSlice:
        """Keep one contiguous baryon-density interval without interpolation."""

        if isinstance(minimum_rows, bool) or int(minimum_rows) != minimum_rows:
            raise EosInputError("minimum_rows must be an integer")
        minimum_rows = int(minimum_rows)
        if minimum_rows < 2:
            raise EosInputError("minimum_rows must be at least two")
        lower = (
            -np.inf if baryon_density_min_fm3 is None else float(baryon_density_min_fm3)
        )
        upper = (
            np.inf if baryon_density_max_fm3 is None else float(baryon_density_max_fm3)
        )
        if not np.isfinite(lower) and lower != -np.inf:
            raise EosInputError("baryon_density_min_fm3 must be finite when supplied")
        if not np.isfinite(upper) and upper != np.inf:
            raise EosInputError("baryon_density_max_fm3 must be finite when supplied")
        if lower <= 0.0 and lower != -np.inf:
            raise EosInputError("baryon_density_min_fm3 must be positive")
        if upper <= 0.0 or upper <= lower:
            raise EosInputError(
                "selected baryon-density limits must satisfy 0 < min < max"
            )
        keep = (self._baryon_density >= lower) & (self._baryon_density <= upper)
        positions = np.flatnonzero(keep)
        if len(positions) < minimum_rows:
            raise EosInputError(
                "selected CompOSE density domain retains fewer than "
                f"{minimum_rows} rows"
            )
        if np.any(np.diff(positions) != 1):
            raise EosInputError(
                "selected CompOSE density domain must be one contiguous source block"
            )
        start = int(positions[0])
        stop = int(positions[-1]) + 1
        phase_codes = None if self.phase_codes is None else self.phase_codes[start:stop]
        return ComposeColdSlice(
            dataset=self.dataset,
            rows=self.rows[start:stop],
            phase_codes=phase_codes,
            matter_declaration=self.matter_declaration,
            source_positions=self.source_positions[start:stop],
        )

    def diagnostic_monotone_subsequence(
        self,
        *,
        conflict_policy: str = "keep_first",
    ) -> ComposeColdSlice:
        """Return an explicit, value-preserving monotone diagnostic reduction.

        ``keep_first`` omits a conflicting later row.  ``keep_later`` replaces
        the last retained non-boundary row when its predecessor stays monotone.
        These policies bracket a local source seam; neither repairs a physical
        transition.
        """

        if conflict_policy not in {"keep_first", "keep_later"}:
            raise EosInputError(
                "diagnostic conflict_policy must be keep_first or keep_later"
            )
        pressure = np.asarray(self.pressure_mev_fm3, dtype=float)
        epsilon = np.asarray(self.energy_density_mev_fm3, dtype=float)
        retained = [0]
        for position in range(1, len(self.rows)):
            previous = retained[-1]
            if (
                pressure[position] > pressure[previous]
                and epsilon[position] > epsilon[previous]
            ):
                retained.append(position)
            elif conflict_policy == "keep_later" and len(retained) > 1:
                predecessor = retained[-2]
                if (
                    pressure[position] > pressure[predecessor]
                    and epsilon[position] > epsilon[predecessor]
                ):
                    retained[-1] = position
        if len(retained) < 4:
            raise EosInputError(
                "diagnostic monotone reduction retains fewer than four source rows"
            )
        phase_codes = (
            None
            if self.phase_codes is None
            else tuple(self.phase_codes[position] for position in retained)
        )
        return ComposeColdSlice(
            dataset=self.dataset,
            rows=tuple(self.rows[position] for position in retained),
            phase_codes=phase_codes,
            matter_declaration=self.matter_declaration,
            source_positions=tuple(
                self.source_positions[position] for position in retained
            ),
        )

    def continuous_segments(self) -> tuple[tuple[int, int], ...]:
        """Return maximal structurally monotone ``[start, stop)`` row blocks."""

        report = self.report()
        split_after = sorted(
            {
                item.left_position
                for item in (
                    *report.pressure_ordering_issues,
                    *report.energy_density_ordering_issues,
                )
            }
        )
        starts = [0, *(index + 1 for index in split_after)]
        stops = [*(index + 1 for index in split_after), len(self.rows)]
        return tuple(
            (start, stop) for start, stop in zip(starts, stops) if stop > start
        )


_ordering_issues = ordering_issues
