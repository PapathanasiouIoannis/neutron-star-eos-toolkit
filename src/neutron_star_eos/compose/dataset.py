"""Lossless source-layer parsing for CompOSE thermodynamic datasets.

This module deliberately stops before interpolation or stellar use.  A valid
CompOSE archive can therefore be inspected even when it cannot yet be reduced
to one continuous cold barotrope.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from neutron_star_eos.eos import EosInputError

COMPOSE_DATASET_SCHEMA_VERSION = "compose_dataset_v2"
COMPOSE_FORMAT_AUTHORITY = (
    "CompOSE Reference Manual v3.01, sections 4.2.1-4.2.9 and appendix A"
)
COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE = 1.0e-7
COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE = 1.0e-7
REQUIRED_FILES = ("eos.t", "eos.nb", "eos.yq", "eos.thermo")
OPTIONAL_FILES = ("eos.compo", "eos.micro", "eos.init", "eos.mr")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ascii(name: str, data: bytes) -> list[str]:
    try:
        return data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise EosInputError(f"CompOSE file {name} is not ASCII") from exc


@dataclass(frozen=True, slots=True)
class ComposeAxis:
    """One indexed CompOSE coordinate axis."""

    minimum_index: int
    maximum_index: int
    values: tuple[float, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.minimum_index, self.maximum_index + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_index": self.minimum_index,
            "maximum_index": self.maximum_index,
            "points": len(self.values),
            "minimum": min(self.values),
            "maximum": max(self.values),
        }


@dataclass(frozen=True, slots=True)
class ComposeThermodynamicRow:
    """All thermodynamic fields from one ``eos.thermo`` record."""

    temperature_index: int
    baryon_density_index: int
    charge_fraction_index: int
    q_values: tuple[float, float, float, float, float, float, float]
    additional_values: tuple[float, ...]

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            self.temperature_index,
            self.baryon_density_index,
            self.charge_fraction_index,
        )


@dataclass(frozen=True, slots=True)
class ComposeCompositionRow:
    """A preserved ``eos.compo`` record without interpreting model-specific fields."""

    temperature_index: int
    baryon_density_index: int
    charge_fraction_index: int
    phase_code: int
    raw_payload_tokens: tuple[str, ...]

    @property
    def key(self) -> tuple[int, int, int]:
        return (
            self.temperature_index,
            self.baryon_density_index,
            self.charge_fraction_index,
        )


@dataclass(frozen=True, slots=True)
class ComposeDiagnostic:
    """One source or reduction diagnostic with explicit severity."""

    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass(frozen=True, slots=True)
class ComposeOrderingIssue:
    """One adjacent source-row ordering failure, preserved without repair."""

    left_position: int
    right_position: int
    left_baryon_density_fm3: float
    right_baryon_density_fm3: float
    left_value: float
    right_value: float

    @property
    def relative_change(self) -> float:
        return (self.right_value - self.left_value) / abs(self.left_value)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "left_position": self.left_position,
            "right_position": self.right_position,
            "left_baryon_density_fm3": self.left_baryon_density_fm3,
            "right_baryon_density_fm3": self.right_baryon_density_fm3,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "relative_change": self.relative_change,
        }


@dataclass(frozen=True, slots=True)
class ComposeSliceReport:
    """Independent diagnostics for one declared cold beta-equilibrium path."""

    rows: int
    euler_maximum_normalized_residual: float
    q5_maximum_absolute_residual: float
    q6_minus_q7_maximum_absolute_residual: float
    pressure_ordering_issues: tuple[ComposeOrderingIssue, ...]
    energy_density_ordering_issues: tuple[ComposeOrderingIssue, ...]
    phase_code_changes: int
    missing_phase_codes: int
    diagnostics: tuple[ComposeDiagnostic, ...]

    @property
    def continuous_barotrope_available(self) -> bool:
        return (
            not self.pressure_ordering_issues
            and not self.energy_density_ordering_issues
            and not any(
                item.severity == "barotrope_blocker" for item in self.diagnostics
            )
        )

    @property
    def status(self) -> str:
        if not self.continuous_barotrope_available:
            return "parsed_but_continuous_barotrope_unavailable"
        if any(item.severity == "warning" for item in self.diagnostics):
            return "continuous_barotrope_available_with_source_diagnostics"
        return "continuous_barotrope_available"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPOSE_DATASET_SCHEMA_VERSION,
            "status": self.status,
            "rows": self.rows,
            "continuous_barotrope_available": self.continuous_barotrope_available,
            "cold_euler_closure": {
                "maximum_normalized_residual": self.euler_maximum_normalized_residual,
                "diagnostic_tolerance": COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
            },
            "beta_equilibrium_Q5": {
                "maximum_absolute_residual": self.q5_maximum_absolute_residual,
                "diagnostic_tolerance": COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
            },
            "zero_temperature_Q6_minus_Q7": {
                "maximum_absolute_residual": self.q6_minus_q7_maximum_absolute_residual,
                "diagnostic_tolerance": COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
            },
            "pressure_ordering_issues": [
                item.to_dict() for item in self.pressure_ordering_issues
            ],
            "energy_density_ordering_issues": [
                item.to_dict() for item in self.energy_density_ordering_issues
            ],
            "phase_code_changes": self.phase_code_changes,
            "missing_phase_codes": self.missing_phase_codes,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _parse_axis(name: str, data: bytes) -> ComposeAxis:
    lines = [line.strip() for line in _ascii(name, data) if line.strip()]
    if len(lines) < 3:
        raise EosInputError(f"CompOSE axis {name} is incomplete")
    try:
        minimum = int(lines[0])
        maximum = int(lines[1])
        values = tuple(float(token) for line in lines[2:] for token in line.split())
    except ValueError as exc:
        raise EosInputError(f"CompOSE axis {name} contains invalid values") from exc
    expected = maximum - minimum + 1
    array = np.asarray(values, dtype=float)
    if expected <= 0 or len(values) != expected:
        raise EosInputError(
            f"CompOSE axis {name} declares {expected} values but contains {len(values)}"
        )
    if not np.all(np.isfinite(array)):
        raise EosInputError(f"CompOSE axis {name} must contain finite values")
    if len(array) > 1 and np.any(np.diff(array) <= 0.0):
        raise EosInputError(f"CompOSE axis {name} must be strictly increasing")
    return ComposeAxis(minimum, maximum, values)


def _read_bundle(path_or_zip: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    path = Path(path_or_zip).expanduser().resolve()
    names = (*REQUIRED_FILES, *OPTIONAL_FILES)
    files: dict[str, bytes] = {}
    if path.is_dir():
        for name in names:
            candidate = path / name
            if candidate.is_file():
                files[name] = candidate.read_bytes()
        identity: dict[str, Any] = {"kind": "directory", "path_name": path.name}
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            for name in names:
                matches = [item for item in members if Path(item.filename).name == name]
                if len(matches) > 1:
                    raise EosInputError(
                        f"CompOSE archive contains multiple {name} files"
                    )
                if matches:
                    files[name] = archive.read(matches[0])
        raw = path.read_bytes()
        identity = {
            "kind": "zip",
            "path_name": path.name,
            "archive_bytes": len(raw),
            "archive_sha256": _sha256(raw),
        }
    else:
        raise EosInputError(f"CompOSE input must be a directory or ZIP archive: {path}")
    missing = [name for name in REQUIRED_FILES if name not in files]
    if missing:
        raise EosInputError(f"CompOSE input is missing required files: {missing}")
    identity["files"] = {
        name: {"bytes": len(data), "sha256": _sha256(data)}
        for name, data in sorted(files.items())
    }
    return files, identity


def _parse_thermodynamics(
    data: bytes,
    *,
    temperature: ComposeAxis,
    density: ComposeAxis,
    charge_fraction: ComposeAxis,
) -> tuple[float, float, int, tuple[ComposeThermodynamicRow, ...], int]:
    lines = [line.strip() for line in _ascii("eos.thermo", data) if line.strip()]
    if len(lines) < 2:
        raise EosInputError("CompOSE eos.thermo is incomplete")
    header = lines[0].split()
    if len(header) != 3:
        raise EosInputError("CompOSE eos.thermo mass header is invalid")
    try:
        neutron_mass = float(header[0])
        proton_mass = float(header[1])
        lepton_indicator = int(header[2])
    except ValueError as exc:
        raise EosInputError("CompOSE mass/lepton header is invalid") from exc
    if (
        not np.isfinite(neutron_mass)
        or neutron_mass <= 0.0
        or not np.isfinite(proton_mass)
        or proton_mass <= 0.0
    ):
        raise EosInputError(
            "CompOSE neutron and proton masses must be finite and positive"
        )

    allowed_t = set(temperature.indices)
    allowed_n = set(density.indices)
    allowed_y = set(charge_fraction.indices)
    rows: dict[tuple[int, int, int], ComposeThermodynamicRow] = {}
    duplicates = 0
    for line_number, line in enumerate(lines[1:], start=2):
        tokens = line.split()
        if len(tokens) < 11:
            raise EosInputError(f"CompOSE eos.thermo line {line_number} is too short")
        try:
            key = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            quantities = tuple(float(value) for value in tokens[3:10])
            additional_count = int(tokens[10])
        except ValueError as exc:
            raise EosInputError(
                f"CompOSE eos.thermo line {line_number} contains invalid fields"
            ) from exc
        if (
            key[0] not in allowed_t
            or key[1] not in allowed_n
            or key[2] not in allowed_y
        ):
            raise EosInputError(
                f"CompOSE eos.thermo line {line_number} has an out-of-grid index"
            )
        if additional_count < 0 or len(tokens) != 11 + additional_count:
            raise EosInputError(
                f"CompOSE eos.thermo line {line_number} has an invalid Nadd payload"
            )
        try:
            additional = tuple(float(value) for value in tokens[11:])
        except ValueError as exc:
            raise EosInputError(
                f"CompOSE eos.thermo line {line_number} has invalid additional values"
            ) from exc
        if not np.all(np.isfinite(quantities)) or not np.all(np.isfinite(additional)):
            raise EosInputError(f"CompOSE eos.thermo line {line_number} is nonfinite")
        duplicates += int(key in rows)
        rows[key] = ComposeThermodynamicRow(
            key[0],
            key[1],
            key[2],
            quantities,  # type: ignore[arg-type]
            additional,
        )
    return neutron_mass, proton_mass, lepton_indicator, tuple(rows.values()), duplicates


def _parse_composition(
    data: bytes | None,
    *,
    temperature: ComposeAxis,
    density: ComposeAxis,
    charge_fraction: ComposeAxis,
) -> tuple[ComposeCompositionRow, ...]:
    if data is None:
        return ()
    allowed_t = set(temperature.indices)
    allowed_n = set(density.indices)
    allowed_y = set(charge_fraction.indices)
    rows: dict[tuple[int, int, int], ComposeCompositionRow] = {}
    for line_number, line in enumerate(_ascii("eos.compo", data), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < 6:
            raise EosInputError(f"CompOSE eos.compo line {line_number} is malformed")
        try:
            key = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            phase_code = int(tokens[3])
        except ValueError as exc:
            raise EosInputError(
                f"CompOSE eos.compo line {line_number} is malformed"
            ) from exc
        if (
            key[0] not in allowed_t
            or key[1] not in allowed_n
            or key[2] not in allowed_y
        ):
            raise EosInputError(
                f"CompOSE eos.compo line {line_number} has an out-of-grid index"
            )
        rows[key] = ComposeCompositionRow(
            key[0], key[1], key[2], phase_code, tuple(tokens[4:])
        )
    return tuple(rows.values())


def _ordering_issues(
    values: np.ndarray,
    baryon_density: np.ndarray,
) -> tuple[ComposeOrderingIssue, ...]:
    return tuple(
        ComposeOrderingIssue(
            int(index),
            int(index) + 1,
            float(baryon_density[int(index)]),
            float(baryon_density[int(index) + 1]),
            float(values[int(index)]),
            float(values[int(index) + 1]),
        )
        for index in np.flatnonzero(np.diff(values) <= 0.0)
    )


class ComposeColdSlice:
    """One explicit cold beta-equilibrium path, before interpolation."""

    def __init__(
        self,
        *,
        dataset: "ComposeDataset",
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
        pressure_issues = _ordering_issues(pressure, self._baryon_density)
        energy_issues = _ordering_issues(epsilon, self._baryon_density)
        diagnostics: list[ComposeDiagnostic] = []
        if np.any(pressure == 0.0):
            diagnostics.append(
                ComposeDiagnostic(
                    "zero_pressure_source_node",
                    "barotrope_blocker",
                    "one or more source nodes have P=0; native thermodynamics remain "
                    "available but the positive-pressure logarithmic barotrope does not",
                )
            )
        for condition, code, message in (
            (
                euler_max > COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
                "cold_euler_closure_residual",
                f"maximum normalized residual {euler_max:.12g} exceeds the diagnostic threshold",
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
        ):
            if condition:
                diagnostics.append(ComposeDiagnostic(code, "warning", message))
        if pressure_issues:
            diagnostics.append(
                ComposeDiagnostic(
                    "pressure_not_strictly_increasing",
                    "barotrope_blocker",
                    f"{len(pressure_issues)} adjacent source-row pair(s) require an explicit seam or transition policy",
                )
            )
        if energy_issues:
            diagnostics.append(
                ComposeDiagnostic(
                    "energy_density_not_strictly_increasing",
                    "barotrope_blocker",
                    f"{len(energy_issues)} adjacent source-row pair(s) prevent continuous inversion",
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
    ) -> "ComposeColdSlice":
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
    ) -> "ComposeColdSlice":
        """Return an explicit monotone diagnostic reduction.

        This operation never changes a source value and always preserves the
        first boundary row. ``keep_first`` omits a conflicting later row.
        ``keep_later`` replaces the last retained non-boundary row when the
        replacement remains monotone relative to its predecessor. Omitted
        source-row positions remain recoverable through ``source_positions``
        and must be reported by callers. The two policies bracket a local seam
        sensitivity; neither resolves a physical transition.
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


class ComposeDataset:
    """Immutable parsed CompOSE source material, independent of interpolation."""

    def __init__(
        self,
        *,
        model_id: str,
        source_url: str,
        source_identity: dict[str, Any],
        source_files: dict[str, bytes],
        temperature: ComposeAxis,
        baryon_density: ComposeAxis,
        charge_fraction: ComposeAxis,
        neutron_mass_mev: float,
        proton_mass_mev: float,
        lepton_indicator: int,
        thermodynamic_rows: tuple[ComposeThermodynamicRow, ...],
        thermodynamic_duplicate_indices: int,
        composition_rows: tuple[ComposeCompositionRow, ...],
    ) -> None:
        self.model_id = model_id
        self.source_url = source_url
        self.temperature = temperature
        self.baryon_density = baryon_density
        self.charge_fraction = charge_fraction
        self.neutron_mass_mev = neutron_mass_mev
        self.proton_mass_mev = proton_mass_mev
        self.lepton_indicator = lepton_indicator
        self.thermodynamic_rows = thermodynamic_rows
        self.thermodynamic_duplicate_indices = thermodynamic_duplicate_indices
        self.composition_rows = composition_rows
        self._source_identity_json = json.dumps(
            source_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        self._source_files = MappingProxyType(dict(source_files))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ComposeDataset is immutable")
        object.__setattr__(self, name, value)

    @property
    def available_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._source_files))

    def source_file_bytes(self, name: str) -> bytes:
        try:
            return bytes(self._source_files[name])
        except KeyError as exc:
            raise KeyError(f"CompOSE source file is not available: {name}") from exc

    def provenance(self) -> dict[str, Any]:
        expected_rows = (
            len(self.temperature.values)
            * len(self.baryon_density.values)
            * len(self.charge_fraction.values)
        )
        return {
            "schema_version": COMPOSE_DATASET_SCHEMA_VERSION,
            "format_authority": COMPOSE_FORMAT_AUTHORITY,
            "model_id": self.model_id,
            "source_url": self.source_url,
            "source_identity": json.loads(self._source_identity_json),
            "axes": {
                "temperature_MeV": self.temperature.to_dict(),
                "baryon_density_fm3": self.baryon_density.to_dict(),
                "charge_fraction": self.charge_fraction.to_dict(),
            },
            "neutron_mass_MeV": self.neutron_mass_mev,
            "proton_mass_MeV": self.proton_mass_mev,
            "lepton_indicator_Il": self.lepton_indicator,
            "thermodynamic_rows": len(self.thermodynamic_rows),
            "expected_rectangular_rows": expected_rows,
            "thermodynamic_duplicate_indices_last_row_wins": (
                self.thermodynamic_duplicate_indices
            ),
            "composition_rows": len(self.composition_rows),
            "additional_thermodynamic_values_preserved": True,
            "optional_source_files_preserved": [
                name for name in OPTIONAL_FILES if name in self._source_files
            ],
        }

    def cold_beta_equilibrium_slice(
        self,
        *,
        matter: str,
        includes_leptons: bool,
    ) -> ComposeColdSlice:
        if matter != "cold_beta_equilibrated":
            raise EosInputError(
                "cold stellar reduction requires an explicit cold beta-equilibrated matter declaration"
            )
        if includes_leptons is not True:
            raise EosInputError(
                "cold stellar reduction must explicitly include leptons"
            )
        if self.lepton_indicator != 1:
            raise EosInputError(
                "CompOSE eos.thermo declares that leptons are absent (Il != 1)"
            )
        zero_indices = [
            index
            for index, value in zip(self.temperature.indices, self.temperature.values)
            if value == 0.0
        ]
        if len(zero_indices) != 1:
            raise EosInputError(
                "cold stellar reduction requires exactly one explicit T=0 CompOSE slice"
            )
        if len(self.charge_fraction.values) != 1:
            raise EosInputError(
                "cold stellar reduction requires one declared beta-equilibrium density path"
            )
        if float(self.charge_fraction.values[0]) != 0.0:
            raise EosInputError(
                "cold beta-equilibrium reduction requires the CompOSE Yq=0 sentinel"
            )
        temperature_index = zero_indices[0]
        charge_index = self.charge_fraction.indices[0]
        row_map = {row.key: row for row in self.thermodynamic_rows}
        keys = tuple(
            (temperature_index, density_index, charge_index)
            for density_index in self.baryon_density.indices
        )
        missing = [key for key in keys if key not in row_map]
        if missing:
            raise EosInputError(
                f"selected CompOSE thermodynamic path is incomplete: {len(missing)} rows missing"
            )
        rows = tuple(row_map[key] for key in keys)
        phase_codes: tuple[int | None, ...] | None = None
        if self.composition_rows:
            composition_map = {row.key: row.phase_code for row in self.composition_rows}
            phase_codes = tuple(composition_map.get(key) for key in keys)
        return ComposeColdSlice(
            dataset=self,
            rows=rows,
            phase_codes=phase_codes,
            matter_declaration=matter,
        )


def load_compose_dataset(
    path_or_zip: str | Path,
    *,
    model_id: str,
    source_url: str,
) -> ComposeDataset:
    """Parse and preserve a CompOSE source without constructing a barotrope."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise EosInputError("CompOSE model_id must be non-empty")
    if not isinstance(source_url, str) or not source_url.strip():
        raise EosInputError("CompOSE source_url must be non-empty")
    files, identity = _read_bundle(path_or_zip)
    temperature = _parse_axis("eos.t", files["eos.t"])
    density = _parse_axis("eos.nb", files["eos.nb"])
    if any(value <= 0.0 for value in density.values):
        raise EosInputError(
            "CompOSE baryon-density axis eos.nb must be strictly positive"
        )
    charge = _parse_axis("eos.yq", files["eos.yq"])
    neutron_mass, proton_mass, lepton_indicator, rows, duplicates = (
        _parse_thermodynamics(
            files["eos.thermo"],
            temperature=temperature,
            density=density,
            charge_fraction=charge,
        )
    )
    composition = _parse_composition(
        files.get("eos.compo"),
        temperature=temperature,
        density=density,
        charge_fraction=charge,
    )
    return ComposeDataset(
        model_id=model_id.strip(),
        source_url=source_url.strip(),
        source_identity=identity,
        source_files=files,
        temperature=temperature,
        baryon_density=density,
        charge_fraction=charge,
        neutron_mass_mev=neutron_mass,
        proton_mass_mev=proton_mass,
        lepton_indicator=lepton_indicator,
        thermodynamic_rows=rows,
        thermodynamic_duplicate_indices=duplicates,
        composition_rows=composition,
    )


__all__ = [
    "COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE",
    "COMPOSE_DATASET_SCHEMA_VERSION",
    "COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE",
    "COMPOSE_FORMAT_AUTHORITY",
    "ComposeColdSlice",
    "ComposeCompositionRow",
    "ComposeDataset",
    "ComposeDiagnostic",
    "ComposeOrderingIssue",
    "ComposeSliceReport",
    "ComposeThermodynamicRow",
    "load_compose_dataset",
]
