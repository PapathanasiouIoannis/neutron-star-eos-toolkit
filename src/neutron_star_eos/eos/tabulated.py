"""Continuous cold-barotrope adapter for ordinary user CSV tables."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from neutron_star_eos.eos import (
    CANONICAL_ENERGY_DENSITY_CONVENTION,
    CANONICAL_UNITS,
    EOS_INPUT_SCHEMA_VERSION,
    EosDomainError,
    EosInputError,
    EosValidationReport,
    _domain_values,
    _scalar_or_array,
    _validate_eos_grid,
)

TABULATED_INTERPOLATION_POLICY = "log_log_pchip_v1"


def _deduplicate_derived_validation_grid(values: Any) -> np.ndarray:
    """Collapse only ULP-near points introduced by validation arithmetic.

    Source table nodes are validated separately and are never passed through
    this helper.  Critical points are found in log space and then exponentiated;
    a root that is mathematically an interval endpoint can consequently differ
    from that endpoint by a handful of floating-point ULPs.  Keeping both points
    creates a meaningless pressure-ordering comparison at effectively the same
    energy density.
    """

    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.ndim != 1 or not np.all(np.isfinite(ordered)):
        raise ValueError("derived validation grid must be finite and one-dimensional")
    retained: list[float] = []
    relative_tolerance = 16.0 * np.finfo(float).eps
    for raw_value in ordered:
        value = float(raw_value)
        if retained and math.isclose(
            value,
            retained[-1],
            rel_tol=relative_tolerance,
            abs_tol=0.0,
        ):
            continue
        retained.append(value)
    return np.asarray(retained, dtype=float)


def _validation_grid_with_exact_endpoints(
    interior_values: Any,
    *,
    lower: float,
    upper: float,
) -> np.ndarray:
    grid = _deduplicate_derived_validation_grid(
        np.concatenate(
            (
                np.asarray([lower], dtype=float),
                np.asarray(interior_values, dtype=float),
                np.asarray([upper], dtype=float),
            )
        )
    )
    grid[0] = lower
    grid[-1] = upper
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("derived validation grid could not preserve exact endpoints")
    return grid


def _one_dimensional(name: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise EosInputError(f"{name} must be one-dimensional")
    if len(array) < 4:
        raise EosInputError("a tabulated EoS requires at least four rows")
    if not np.all(np.isfinite(array)):
        raise EosInputError(f"{name} must contain only finite values")
    return array


class TabulatedEos:
    """One strictly ordered, continuous cold barotrope.

    The table is never sorted, clipped, repaired, or extrapolated.  Constant
    pressure plateaus and density jumps are intentionally rejected in v1;
    they need an explicit segmented representation and discontinuity audit.
    """

    surface_boundary_kind = "finite_source_pressure_not_vacuum"
    tidal_capability_status = "unavailable_positive_pressure_source_boundary"

    def __init__(
        self,
        *,
        name: str,
        energy_density_mev_fm3: Any,
        pressure_mev_fm3: Any,
        source: str,
        baryon_density_fm3: Any | None = None,
        source_metadata: dict[str, Any] | None = None,
        energy_density_convention: str = CANONICAL_ENERGY_DENSITY_CONVENTION,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise EosInputError("tabulated EoS name must be non-empty")
        if not isinstance(source, str) or not source.strip():
            raise EosInputError("tabulated EoS source must be non-empty")
        if energy_density_convention != CANONICAL_ENERGY_DENSITY_CONVENTION:
            raise EosInputError(
                "energy density must be total energy density including rest mass"
            )
        if source_metadata is not None and not isinstance(source_metadata, dict):
            raise EosInputError("source_metadata must be a JSON-safe dictionary")
        try:
            json.dumps(source_metadata or {}, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EosInputError("source_metadata must be JSON-safe and finite") from exc
        epsilon = _one_dimensional("energy_density_mev_fm3", energy_density_mev_fm3)
        pressure = _one_dimensional("pressure_mev_fm3", pressure_mev_fm3)
        if epsilon.shape != pressure.shape:
            raise EosInputError(
                "energy-density and pressure columns must have equal length"
            )
        if np.any(epsilon <= 0.0) or np.any(pressure <= 0.0):
            raise EosInputError(
                "v1 tabulated energy density and pressure must be positive"
            )
        if np.any(np.diff(epsilon) <= 0.0):
            raise EosInputError(
                "energy density must be supplied in strictly increasing order"
            )
        if np.any(np.diff(pressure) <= 0.0):
            raise EosInputError(
                "pressure must increase strictly; plateaus or jumps need explicit future support"
            )

        baryon_density = None
        if baryon_density_fm3 is not None:
            baryon_density = _one_dimensional("baryon_density_fm3", baryon_density_fm3)
            if baryon_density.shape != epsilon.shape:
                raise EosInputError("baryon-density column must match the table length")
            if np.any(baryon_density <= 0.0) or np.any(np.diff(baryon_density) <= 0.0):
                raise EosInputError(
                    "baryon density must be positive and strictly increasing"
                )

        self.model_name = name.strip()
        self.source = source.strip()
        self.energy_density_convention = energy_density_convention
        self._energy_density_mev_fm3 = epsilon.copy()
        self._pressure_mev_fm3 = pressure.copy()
        self._baryon_density_fm3 = (
            None if baryon_density is None else baryon_density.copy()
        )
        self._energy_density_mev_fm3.setflags(write=False)
        self._pressure_mev_fm3.setflags(write=False)
        if self._baryon_density_fm3 is not None:
            self._baryon_density_fm3.setflags(write=False)
        self._source_metadata = copy.deepcopy(source_metadata or {})
        self.energy_density_min_mev_fm3 = float(epsilon[0])
        self.energy_density_max_mev_fm3 = float(epsilon[-1])
        self.pressure_min_mev_fm3 = float(pressure[0])
        self.pressure_max_mev_fm3 = float(pressure[-1])

        log_epsilon = np.log(epsilon)
        log_pressure = np.log(pressure)
        self._log_pressure_from_log_epsilon = PchipInterpolator(
            log_epsilon, log_pressure, extrapolate=False
        )
        self._dlog_pressure_dlog_epsilon = (
            self._log_pressure_from_log_epsilon.derivative()
        )

    @staticmethod
    def _readonly_view(values: np.ndarray) -> np.ndarray:
        view = values.view()
        view.setflags(write=False)
        return view

    @property
    def energy_density_mev_fm3(self) -> np.ndarray:
        return self._readonly_view(self._energy_density_mev_fm3)

    @property
    def pressure_mev_fm3(self) -> np.ndarray:
        return self._readonly_view(self._pressure_mev_fm3)

    @property
    def baryon_density_fm3(self) -> np.ndarray | None:
        if self._baryon_density_fm3 is None:
            return None
        return self._readonly_view(self._baryon_density_fm3)

    @property
    def source_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._source_metadata)

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> Any:
        values, scalar = _domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        result = np.exp(self._log_pressure_from_log_epsilon(np.log(values)))
        result = np.where(
            values == self.energy_density_min_mev_fm3,
            self.pressure_min_mev_fm3,
            result,
        )
        result = np.where(
            values == self.energy_density_max_mev_fm3,
            self.pressure_max_mev_fm3,
            result,
        )
        if not np.all(np.isfinite(result)):
            raise EosDomainError(
                "pressure interpolation left the declared table domain"
            )
        return _scalar_or_array(result, scalar)

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> Any:
        values, scalar = _domain_values(
            pressure_mev_fm3,
            name="pressure_mev_fm3",
            lower=self.pressure_min_mev_fm3,
            upper=self.pressure_max_mev_fm3,
        )
        result = np.empty_like(values, dtype=float)
        flat_result = result.reshape(-1)
        log_energy_nodes = np.log(self._energy_density_mev_fm3)
        for output_index, pressure in enumerate(values.reshape(-1)):
            if pressure == self.pressure_min_mev_fm3:
                flat_result[output_index] = self.energy_density_min_mev_fm3
                continue
            if pressure == self.pressure_max_mev_fm3:
                flat_result[output_index] = self.energy_density_max_mev_fm3
                continue
            interval = int(
                np.searchsorted(self._pressure_mev_fm3, pressure, side="right") - 1
            )
            interval = min(max(interval, 0), len(self._pressure_mev_fm3) - 2)
            target = math.log(float(pressure))
            root = brentq(
                lambda log_epsilon, target=target: float(
                    self._log_pressure_from_log_epsilon(log_epsilon) - target
                ),
                float(log_energy_nodes[interval]),
                float(log_energy_nodes[interval + 1]),
                xtol=5.0e-15,
                rtol=4.0 * np.finfo(float).eps,
            )
            flat_result[output_index] = math.exp(root)
        if not np.all(np.isfinite(result)):
            raise EosDomainError(
                "energy-density interpolation left the declared table domain"
            )
        return _scalar_or_array(result, scalar)

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> Any:
        values, scalar = _domain_values(
            energy_density_mev_fm3,
            name="energy_density_mev_fm3",
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        pressure = np.asarray(self.pressure_from_energy_density(values), dtype=float)
        slope = np.asarray(
            self._dlog_pressure_dlog_epsilon(np.log(values)), dtype=float
        )
        result = pressure * slope / values
        if not np.all(np.isfinite(result)):
            raise EosInputError("sound-speed derivative is nonfinite")
        return _scalar_or_array(result, scalar)

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]:
        epsilon = float(self.energy_density_from_pressure(pressure_mev_fm3))
        return epsilon, float(self.sound_speed_squared_from_energy_density(epsilon))

    def validate(self, *, points: int = 2049) -> EosValidationReport:
        if int(points) < 17:
            raise ValueError("validation points must be at least 17")
        log_epsilon = np.log(self._energy_density_mev_fm3)
        interval_count = len(log_epsilon) - 1
        points_per_interval = max(5, int(np.ceil((int(points) - 1) / interval_count)))
        segments = [
            np.linspace(
                log_epsilon[index],
                log_epsilon[index + 1],
                points_per_interval,
                endpoint=False,
            )
            for index in range(interval_count)
        ]
        critical_logs: list[float] = []
        coefficients = self._log_pressure_from_log_epsilon.c
        for index in range(interval_count):
            width = float(log_epsilon[index + 1] - log_epsilon[index])
            cubic, quadratic, linear, _constant = coefficients[:, index]
            first_derivative = Polynomial((linear, 2.0 * quadratic, 3.0 * cubic))
            sound_speed_stationary = (
                first_derivative.deriv()
                + first_derivative * first_derivative
                - first_derivative
            )
            for polynomial in (first_derivative, sound_speed_stationary):
                for root in polynomial.roots():
                    if abs(float(np.imag(root))) > 1.0e-11:
                        continue
                    local = float(np.real(root))
                    if -1.0e-12 <= local <= width + 1.0e-12:
                        critical_logs.append(
                            float(log_epsilon[index] + min(max(local, 0.0), width))
                        )
        validation_logs = np.unique(
            np.concatenate(
                (
                    *segments,
                    np.asarray([log_epsilon[-1]], dtype=float),
                    np.asarray(critical_logs, dtype=float),
                )
            )
        )
        raw_validation_grid = np.exp(validation_logs)
        interior_grid = raw_validation_grid[
            (raw_validation_grid > self.energy_density_min_mev_fm3)
            & (raw_validation_grid < self.energy_density_max_mev_fm3)
        ]
        validation_grid = _validation_grid_with_exact_endpoints(
            interior_grid,
            lower=self.energy_density_min_mev_fm3,
            upper=self.energy_density_max_mev_fm3,
        )
        return _validate_eos_grid(self, validation_grid)

    def provenance(self) -> dict[str, Any]:
        payload = {
            "schema_version": EOS_INPUT_SCHEMA_VERSION,
            "adapter": "tabulated_eos_v1",
            "model_name": self.model_name,
            "source": self.source,
            "units": CANONICAL_UNITS,
            "energy_density_convention": self.energy_density_convention,
            "rows": int(len(self._energy_density_mev_fm3)),
            "baryon_density_available": self._baryon_density_fm3 is not None,
            "interpolation": TABULATED_INTERPOLATION_POLICY,
            "extrapolation": "forbidden",
            "discontinuous_barotropes": "unsupported_fail_closed",
            "validation_policy": "every_log_pchip_sound_speed_extremum",
            "stellar_surface": {
                "kind": self.surface_boundary_kind,
                "pressure_MeV_fm3": self.pressure_min_mev_fm3,
                "tidal_capability": self.tidal_capability_status,
            },
            "domain": {
                "energy_density_min_MeV_fm3": self.energy_density_min_mev_fm3,
                "energy_density_max_MeV_fm3": self.energy_density_max_mev_fm3,
                "pressure_min_MeV_fm3": self.pressure_min_mev_fm3,
                "pressure_max_MeV_fm3": self.pressure_max_mev_fm3,
            },
        }
        payload["source_metadata"] = copy.deepcopy(self._source_metadata)
        return payload


def load_csv_eos(
    path: str | Path,
    *,
    epsilon_column: str = "epsilon_mev_fm3",
    pressure_column: str = "pressure_mev_fm3",
    baryon_density_column: str | None = None,
    units: str = CANONICAL_UNITS,
    name: str | None = None,
    source: str | None = None,
    energy_density_convention: str = CANONICAL_ENERGY_DENSITY_CONVENTION,
) -> TabulatedEos:
    """Load an ordinary CSV without guessing columns, units, or ordering."""
    if units != CANONICAL_UNITS:
        raise EosInputError(f"v1 accepts only canonical units {CANONICAL_UNITS!r}")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EosInputError(f"CSV EoS file does not exist: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise EosInputError("CSV EoS is missing a header row")
            if len(fieldnames) != len(set(fieldnames)):
                raise EosInputError("CSV EoS contains duplicate column names")
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise EosInputError("CSV EoS must be UTF-8 encoded") from exc
    requested = [epsilon_column, pressure_column]
    if baryon_density_column is not None:
        requested.append(baryon_density_column)
    missing = [column for column in requested if column not in fieldnames]
    if missing:
        raise EosInputError(f"CSV EoS is missing columns: {missing}")
    if not rows:
        raise EosInputError("CSV EoS contains no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise EosInputError(
            "CSV EoS rows must contain exactly the number of fields declared by the header"
        )

    def values(column: str) -> np.ndarray:
        try:
            return np.asarray([float(row[column]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise EosInputError(
                f"CSV EoS column {column!r} must contain only numeric values"
            ) from exc

    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return TabulatedEos(
        name=name or resolved.stem,
        energy_density_mev_fm3=values(epsilon_column),
        pressure_mev_fm3=values(pressure_column),
        baryon_density_fm3=(
            None if baryon_density_column is None else values(baryon_density_column)
        ),
        source=source or resolved.name,
        source_metadata={
            "source_file": {
                "name": resolved.name,
                "bytes": resolved.stat().st_size,
                "sha256": digest,
            },
            "columns": {
                "energy_density": epsilon_column,
                "pressure": pressure_column,
                "baryon_density": baryon_density_column,
            },
        },
        energy_density_convention=energy_density_convention,
    )


__all__ = ["TABULATED_INTERPOLATION_POLICY", "TabulatedEos", "load_csv_eos"]
