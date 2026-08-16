"""Read-only, source-aware thermodynamic data views for presentation layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

THERMODYNAMIC_SERIES_ROLES = (
    "source_nodes",
    "native_thermodynamics",
    "continuous_barotrope",
)


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))


@dataclass(frozen=True, slots=True)
class ThermodynamicSeries:
    """One aligned collection of named thermodynamic columns.

    Arrays are copied and made read-only.  ``role`` distinguishes authoritative
    source nodes, a native thermodynamic assessment, and an evaluated
    continuous stellar barotrope so plots never imply that they are identical.
    """

    role: str
    label: str
    columns: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    descriptions: Mapping[str, str]
    diagnostic_codes: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.role not in THERMODYNAMIC_SERIES_ROLES:
            raise ValueError(
                f"role must be one of {THERMODYNAMIC_SERIES_ROLES}, got {self.role!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("thermodynamic series label must be non-empty")
        immutable: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for name, values in self.columns.items():
            if not isinstance(name, str) or not name:
                raise ValueError("thermodynamic column names must be non-empty strings")
            array = np.asarray(values, dtype=float)
            if array.ndim != 1:
                raise ValueError(
                    f"thermodynamic column {name!r} must be one-dimensional"
                )
            copied = array.copy()
            copied.setflags(write=False)
            immutable[name] = copied
            lengths.add(len(copied))
        if not immutable:
            raise ValueError("a thermodynamic series requires at least one column")
        if len(lengths) != 1:
            raise ValueError("thermodynamic series columns must have equal lengths")
        missing_units = set(immutable) - set(self.units)
        missing_descriptions = set(immutable) - set(self.descriptions)
        if missing_units:
            raise ValueError(f"missing units for columns: {sorted(missing_units)}")
        if missing_descriptions:
            raise ValueError(
                f"missing descriptions for columns: {sorted(missing_descriptions)}"
            )
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "columns", MappingProxyType(immutable))
        object.__setattr__(
            self,
            "units",
            MappingProxyType({name: str(self.units[name]) for name in immutable}),
        )
        object.__setattr__(
            self,
            "descriptions",
            MappingProxyType(
                {name: str(self.descriptions[name]) for name in immutable}
            ),
        )
        object.__setattr__(
            self, "diagnostic_codes", tuple(dict.fromkeys(self.diagnostic_codes))
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(_json_copy(self.metadata or {})),
        )

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    @property
    def rows(self) -> int:
        return len(next(iter(self.columns.values())))

    def column(self, name: str) -> np.ndarray:
        """Return a read-only view of one named column."""

        values = self.columns[name].view()
        values.setflags(write=False)
        return values


@dataclass(frozen=True, slots=True)
class ThermodynamicView:
    """All thermodynamic representations currently available for one model."""

    model_name: str
    input_kind: str
    series: tuple[ThermodynamicSeries, ...]

    def __post_init__(self) -> None:
        roles = tuple(item.role for item in self.series)
        if not self.series:
            raise ValueError("a thermodynamic view requires at least one series")
        if len(set(roles)) != len(roles):
            raise ValueError("thermodynamic series roles must be unique")

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(item.role for item in self.series)

    def series_for(self, role: str) -> ThermodynamicSeries:
        for item in self.series:
            if item.role == role:
                return item
        raise KeyError(role)


__all__ = [
    "THERMODYNAMIC_SERIES_ROLES",
    "ThermodynamicSeries",
    "ThermodynamicView",
]
