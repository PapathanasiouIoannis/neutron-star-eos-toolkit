"""Read optional CompOSE ``eos.mr`` data as an independent reference.

The CompOSE format guarantees that radius in km and gravitational mass in
solar masses occupy the first two columns.  Any later columns are
model-dependent and therefore remain explicitly uninterpreted here.  This
module does not sort a curve, infer a stable branch, calculate a maximum mass,
or provide data to the toolkit's TOV solver.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from neutron_star_eos.compose.dataset import ComposeDataset
from neutron_star_eos.eos import EosInputError

COMPOSE_MASS_RADIUS_SCHEMA_VERSION = "compose_mass_radius_reference_v1"
COMPOSE_MASS_RADIUS_FORMAT_AUTHORITY = "CompOSE Reference Manual v3.01, section 4.2.6"


def _readonly(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ComposeMassRadiusReference:
    """Immutable source-order view of one optional ``eos.mr`` table."""

    model_id: str
    source_url: str
    radius_km: np.ndarray
    mass_msun: np.ndarray
    additional_columns: tuple[np.ndarray, ...]
    header_lines: tuple[str, ...]
    source_bytes: int
    source_sha256: str
    archive_sha256: str | None = None

    def __post_init__(self) -> None:
        radius = _readonly(self.radius_km)
        mass = _readonly(self.mass_msun)
        additional = tuple(_readonly(values) for values in self.additional_columns)
        if radius.ndim != 1 or mass.ndim != 1 or radius.shape != mass.shape:
            raise ValueError("CompOSE mass-radius columns must be aligned 1D arrays")
        if len(radius) == 0:
            raise ValueError("CompOSE mass-radius reference must contain a data row")
        if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
            raise ValueError("CompOSE reference radii must be finite and positive")
        if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("CompOSE reference masses must be finite and positive")
        for values in additional:
            if values.ndim != 1 or values.shape != radius.shape:
                raise ValueError(
                    "CompOSE additional mass-radius columns must align with radius"
                )
            if np.any(~np.isfinite(values)):
                raise ValueError(
                    "CompOSE additional mass-radius columns must be finite"
                )
        if not self.model_id.strip() or not self.source_url.strip():
            raise ValueError("CompOSE mass-radius source identity must be non-empty")
        if self.source_bytes <= 0:
            raise ValueError("CompOSE mass-radius source byte count must be positive")
        if len(self.source_sha256) != 64:
            raise ValueError("CompOSE mass-radius source SHA-256 is malformed")
        if self.archive_sha256 is not None and len(self.archive_sha256) != 64:
            raise ValueError("CompOSE archive SHA-256 is malformed")
        object.__setattr__(self, "radius_km", radius)
        object.__setattr__(self, "mass_msun", mass)
        object.__setattr__(self, "additional_columns", additional)
        object.__setattr__(self, "header_lines", tuple(self.header_lines))

    @property
    def rows(self) -> int:
        return len(self.radius_km)

    @property
    def column_names(self) -> tuple[str, ...]:
        return (
            "radius_km",
            "mass_msun",
            *(
                f"source_column_{index}"
                for index in range(3, 3 + len(self.additional_columns))
            ),
        )

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        values = (self.radius_km, self.mass_msun, *self.additional_columns)
        return MappingProxyType(dict(zip(self.column_names, values)))

    def column(self, name: str) -> np.ndarray:
        """Return a read-only column without assigning semantics to extras."""

        try:
            return self.columns[name]
        except KeyError as exc:
            raise KeyError(f"unknown CompOSE mass-radius column: {name}") from exc

    def provenance(self) -> dict[str, Any]:
        """Return the source identity and the deliberately narrow interpretation."""

        return {
            "schema_version": COMPOSE_MASS_RADIUS_SCHEMA_VERSION,
            "format_authority": COMPOSE_MASS_RADIUS_FORMAT_AUTHORITY,
            "role": "independent_reference_not_solver_input",
            "model_id": self.model_id,
            "source_url": self.source_url,
            "rows": self.rows,
            "columns": list(self.column_names),
            "source_order_preserved": True,
            "stable_branch_inferred": False,
            "maximum_mass_inferred": False,
            "additional_columns_interpreted": False,
            "header_lines": list(self.header_lines),
            "source_file": {
                "name": "eos.mr",
                "bytes": self.source_bytes,
                "sha256": self.source_sha256,
            },
            "archive_sha256": self.archive_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready columnar representation plus provenance."""

        return {
            "provenance": self.provenance(),
            "data": {
                name: [float(value) for value in self.column(name)]
                for name in self.column_names
            },
        }


def _numeric_row(line: str, *, line_number: int) -> tuple[float, ...]:
    try:
        values = tuple(
            float(token.replace("D", "E").replace("d", "e")) for token in line.split()
        )
    except ValueError as exc:
        raise EosInputError(
            f"CompOSE eos.mr line {line_number} contains invalid numeric fields"
        ) from exc
    if len(values) < 2:
        raise EosInputError(
            f"CompOSE eos.mr line {line_number} must contain at least radius and mass"
        )
    if not np.all(np.isfinite(values)):
        raise EosInputError(f"CompOSE eos.mr line {line_number} is nonfinite")
    return values


def load_compose_mass_radius_reference(
    dataset: ComposeDataset,
) -> ComposeMassRadiusReference:
    """Parse a dataset's optional ``eos.mr`` without using it in a TOV solve.

    Initial comment or textual header rows are retained.  Once numerical data
    begin, every non-comment row must be numeric and have a consistent width.
    Only the first two columns receive standard meanings; consult the model's
    CompOSE data sheet before interpreting any later source column.
    """

    if not isinstance(dataset, ComposeDataset):
        raise TypeError("load_compose_mass_radius_reference expects ComposeDataset")
    if "eos.mr" not in dataset.available_files:
        raise EosInputError("the CompOSE source does not provide eos.mr")
    raw = dataset.source_file_bytes("eos.mr")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise EosInputError("CompOSE eos.mr must be ASCII") from exc

    header_lines: list[str] = []
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header_lines.append(line)
            continue
        first_token = line.split(maxsplit=1)[0]
        try:
            float(first_token.replace("D", "E").replace("d", "e"))
        except ValueError:
            if not rows:
                header_lines.append(line)
                continue
        values = _numeric_row(line, line_number=line_number)
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise EosInputError(
                f"CompOSE eos.mr line {line_number} has {len(values)} columns; "
                f"expected {width}"
            )
        rows.append(values)
    if not rows:
        raise EosInputError("CompOSE eos.mr contains no numerical data rows")

    table = np.asarray(rows, dtype=float)
    provenance = dataset.provenance()
    source_identity = provenance["source_identity"]
    archive_sha256 = source_identity.get("archive_sha256")
    try:
        return ComposeMassRadiusReference(
            model_id=dataset.model_id,
            source_url=dataset.source_url,
            radius_km=table[:, 0],
            mass_msun=table[:, 1],
            additional_columns=tuple(
                table[:, index] for index in range(2, table.shape[1])
            ),
            header_lines=tuple(header_lines),
            source_bytes=len(raw),
            source_sha256=hashlib.sha256(raw).hexdigest(),
            archive_sha256=(None if archive_sha256 is None else str(archive_sha256)),
        )
    except ValueError as exc:
        raise EosInputError(f"invalid CompOSE eos.mr reference: {exc}") from exc


__all__ = [
    "COMPOSE_MASS_RADIUS_FORMAT_AUTHORITY",
    "COMPOSE_MASS_RADIUS_SCHEMA_VERSION",
    "ComposeMassRadiusReference",
    "load_compose_mass_radius_reference",
]
