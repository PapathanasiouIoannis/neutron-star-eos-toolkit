"""Fail-closed reader for cold one-dimensional CompOSE barotropes."""

from __future__ import annotations

import copy
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neutron_star_eos.eos import EosInputError
from neutron_star_eos.tabulated import TabulatedEos


COMPOSE_PARSER_VERSION = "compose_cold_1d_v1"
COMPOSE_FORMAT_AUTHORITY = (
    "CompOSE Reference Manual v3.01, sections 4.2.1-4.2.3 and 4.2.7"
)
# Source-row redundancy check, not a stellar-solver tolerance. The fixed value
# accommodates decimal serialization while remaining tighter than 0.1 ppm.
COMPOSE_EULER_CLOSURE_RELATIVE_TOLERANCE = 1.0e-7
# Q5 and Q6-Q7 are dimensionless. This source-row tolerance accommodates
# ordinary decimal serialization while remaining far below a keV per baryon.
COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE = 1.0e-7
REQUIRED_FILES = ("eos.t", "eos.nb", "eos.yq", "eos.thermo")


@dataclass(frozen=True, slots=True)
class _Axis:
    minimum_index: int
    maximum_index: int
    values: np.ndarray

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(range(self.minimum_index, self.maximum_index + 1))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ascii(name: str, data: bytes) -> list[str]:
    try:
        return data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise EosInputError(f"CompOSE file {name} is not ASCII") from exc


def _parse_axis(name: str, data: bytes) -> _Axis:
    lines = [line.strip() for line in _ascii(name, data) if line.strip()]
    if len(lines) < 3:
        raise EosInputError(f"CompOSE axis {name} is incomplete")
    try:
        minimum = int(lines[0])
        maximum = int(lines[1])
        values = np.asarray(
            [float(token) for line in lines[2:] for token in line.split()], dtype=float
        )
    except ValueError as exc:
        raise EosInputError(f"CompOSE axis {name} contains invalid values") from exc
    expected = maximum - minimum + 1
    if expected <= 0 or len(values) != expected:
        raise EosInputError(
            f"CompOSE axis {name} declares {expected} values but contains {len(values)}"
        )
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        if len(values) != 1 or not np.all(np.isfinite(values)):
            raise EosInputError(f"CompOSE axis {name} must be finite and strictly increasing")
    return _Axis(minimum, maximum, values)


def _bundle_from_directory(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    files: dict[str, bytes] = {}
    for name in (*REQUIRED_FILES, "eos.compo", "eos.init", "eos.mr"):
        candidate = path / name
        if candidate.is_file():
            files[name] = candidate.read_bytes()
    return files, {"kind": "directory", "path_name": path.name}


def _bundle_from_zip(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        for name in (*REQUIRED_FILES, "eos.compo", "eos.init", "eos.mr"):
            matches = [item for item in members if Path(item.filename).name == name]
            if len(matches) > 1:
                raise EosInputError(f"CompOSE archive contains multiple {name} files")
            if matches:
                files[name] = archive.read(matches[0])
    raw = path.read_bytes()
    return files, {
        "kind": "zip",
        "path_name": path.name,
        "archive_bytes": len(raw),
        "archive_sha256": _sha256(raw),
    }


def _read_bundle(path_or_zip: str | Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    path = Path(path_or_zip).expanduser().resolve()
    if path.is_dir():
        files, identity = _bundle_from_directory(path)
    elif path.is_file() and zipfile.is_zipfile(path):
        files, identity = _bundle_from_zip(path)
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
    temperature: _Axis,
    density: _Axis,
    charge_fraction: _Axis,
) -> tuple[float, float, int, dict[tuple[int, int, int], tuple[float, ...]], int]:
    lines = [line.strip() for line in _ascii("eos.thermo", data) if line.strip()]
    if len(lines) < 2:
        raise EosInputError("CompOSE eos.thermo is incomplete")
    header = lines[0].split()
    if len(header) != 3:
        raise EosInputError("CompOSE eos.thermo mass header is invalid")
    try:
        neutron_mass_mev = float(header[0])
        proton_mass_mev = float(header[1])
        lepton_indicator = int(header[2])
    except ValueError as exc:
        raise EosInputError("CompOSE mass/lepton header is invalid") from exc
    if (
        not np.isfinite(neutron_mass_mev)
        or neutron_mass_mev <= 0.0
        or not np.isfinite(proton_mass_mev)
        or proton_mass_mev <= 0.0
    ):
        raise EosInputError("CompOSE neutron and proton masses must be finite and positive")

    allowed_t = set(temperature.indices)
    allowed_n = set(density.indices)
    allowed_y = set(charge_fraction.indices)
    rows: dict[tuple[int, int, int], tuple[float, ...]] = {}
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
        if key[0] not in allowed_t or key[1] not in allowed_n or key[2] not in allowed_y:
            raise EosInputError(f"CompOSE eos.thermo line {line_number} has an out-of-grid index")
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
        rows[key] = quantities
    return neutron_mass_mev, proton_mass_mev, lepton_indicator, rows, duplicates


def _parse_phase_codes(
    data: bytes | None,
    *,
    expected_keys: tuple[tuple[int, int, int], ...],
) -> tuple[int | None, ...] | None:
    if data is None:
        return None
    mapping: dict[tuple[int, int, int], int] = {}
    allowed = set(expected_keys)
    for line_number, line in enumerate(_ascii("eos.compo", data), start=1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            key = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            phase = int(tokens[3])
            pair_count = int(tokens[4])
            if pair_count < 0:
                raise ValueError
            cursor = 5 + 2 * pair_count
            quadruple_count = int(tokens[cursor])
            if quadruple_count < 0 or len(tokens) != cursor + 1 + 4 * quadruple_count:
                raise ValueError
            for pair_index in range(pair_count):
                int(tokens[5 + 2 * pair_index])
                if not np.isfinite(float(tokens[6 + 2 * pair_index])):
                    raise ValueError
            for quadruple_index in range(quadruple_count):
                start = cursor + 1 + 4 * quadruple_index
                int(tokens[start])
                for value in tokens[start + 1 : start + 4]:
                    if not np.isfinite(float(value)):
                        raise ValueError
        except (IndexError, ValueError) as exc:
            raise EosInputError(
                f"CompOSE eos.compo line {line_number} is malformed"
            ) from exc
        if key not in allowed:
            raise EosInputError(
                f"CompOSE eos.compo line {line_number} has an out-of-grid index"
            )
        mapping[key] = phase
    return tuple(mapping.get(key) for key in expected_keys)


class ComposeEos(TabulatedEos):
    """Continuous cold 1D CompOSE table normalized to the common adapter."""

    def __init__(
        self,
        *,
        phase_codes: tuple[int | None, ...] | None,
        baryon_chemical_potential_mev: np.ndarray,
        compose_metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(source_metadata={"compose": compose_metadata}, **kwargs)
        chemical_potential = np.asarray(
            baryon_chemical_potential_mev, dtype=float
        )
        if (
            chemical_potential.shape != self._energy_density_mev_fm3.shape
            or not np.all(np.isfinite(chemical_potential))
            or np.any(chemical_potential <= 0.0)
        ):
            raise EosInputError(
                "CompOSE baryon chemical potential must be finite, positive, and row-aligned"
            )
        if phase_codes is not None and len(phase_codes) != len(chemical_potential):
            raise EosInputError("CompOSE phase codes must align with retained rows")
        self.phase_codes = phase_codes
        self._baryon_chemical_potential_mev = chemical_potential.copy()
        self._baryon_chemical_potential_mev.setflags(write=False)
        self._compose_metadata = copy.deepcopy(compose_metadata)

    @property
    def baryon_chemical_potential_mev(self) -> np.ndarray:
        return self._readonly_view(self._baryon_chemical_potential_mev)

    @property
    def compose_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._compose_metadata)


def load_compose_eos(
    path_or_zip: str | Path,
    *,
    model_id: str,
    source_url: str,
    matter: str,
    includes_leptons: bool,
    baryon_density_max_fm3: float | None = None,
) -> ComposeEos:
    """Read a declared cold beta-equilibrated continuous CompOSE table.

    CompOSE's file format alone cannot prove the physical matter choice.
    Therefore the caller must explicitly supply the catalogue model identity,
    source URL, beta-equilibrium declaration, and lepton inclusion.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise EosInputError("CompOSE model_id must be non-empty")
    if not isinstance(source_url, str) or not source_url.strip():
        raise EosInputError("CompOSE source_url must be non-empty")
    if matter != "cold_beta_equilibrated":
        raise EosInputError("v1 supports only explicitly declared cold beta-equilibrated matter")
    if includes_leptons is not True:
        raise EosInputError("v1 stellar CompOSE input must explicitly include leptons")

    files, bundle_identity = _read_bundle(path_or_zip)
    temperature = _parse_axis("eos.t", files["eos.t"])
    density = _parse_axis("eos.nb", files["eos.nb"])
    charge = _parse_axis("eos.yq", files["eos.yq"])
    if len(temperature.values) != 1 or float(temperature.values[0]) != 0.0:
        raise EosInputError("v1 requires exactly one T=0 CompOSE slice")
    if len(charge.values) != 1:
        raise EosInputError("v1 requires exactly one charge-fraction grid point")

    (
        neutron_mass,
        proton_mass,
        lepton_indicator,
        rows,
        duplicate_count,
    ) = _parse_thermodynamics(
        files["eos.thermo"],
        temperature=temperature,
        density=density,
        charge_fraction=charge,
    )
    if lepton_indicator != 1:
        raise EosInputError(
            "CompOSE eos.thermo declares that leptons are absent (Il != 1)"
        )
    t_index = temperature.indices[0]
    y_index = charge.indices[0]
    keys = tuple((t_index, n_index, y_index) for n_index in density.indices)
    missing = [key for key in keys if key not in rows]
    if missing:
        raise EosInputError(
            f"CompOSE thermodynamic path is incomplete: {len(missing)} rows missing"
        )
    q1 = np.asarray([rows[key][0] for key in keys], dtype=float)
    q3 = np.asarray([rows[key][2] for key in keys], dtype=float)
    q5 = np.asarray([rows[key][4] for key in keys], dtype=float)
    q6 = np.asarray([rows[key][5] for key in keys], dtype=float)
    q7 = np.asarray([rows[key][6] for key in keys], dtype=float)
    pressure = density.values * q1
    baryon_chemical_potential = neutron_mass * (1.0 + q3)
    total_energy_density = density.values * neutron_mass * (1.0 + q7)
    euler_residual = pressure - (
        density.values * baryon_chemical_potential - total_energy_density
    )
    euler_scale = np.maximum.reduce(
        (
            np.abs(pressure),
            np.abs(density.values * baryon_chemical_potential),
            np.abs(total_energy_density),
        )
    )
    if np.any(~np.isfinite(euler_scale)) or np.any(euler_scale <= 0.0):
        raise EosInputError("CompOSE rows have a nonpositive cold-Euler normalization")
    normalized_euler_residual = np.abs(euler_residual) / euler_scale
    source_maximum_euler_residual = float(np.max(normalized_euler_residual))
    beta_equilibrium_residual = np.abs(q5)
    cold_free_energy_residual = np.abs(q6 - q7)
    source_maximum_beta_residual = float(np.max(beta_equilibrium_residual))
    source_maximum_cold_free_energy_residual = float(
        np.max(cold_free_energy_residual)
    )
    phase_codes = _parse_phase_codes(files.get("eos.compo"), expected_keys=keys)
    retained = np.ones(len(density.values), dtype=bool)
    declared_upper_density = None
    if baryon_density_max_fm3 is not None:
        declared_upper_density = float(baryon_density_max_fm3)
        if not np.isfinite(declared_upper_density) or declared_upper_density <= 0.0:
            raise EosInputError("baryon_density_max_fm3 must be finite and positive")
        retained = density.values <= declared_upper_density
        if int(np.count_nonzero(retained)) < 4:
            raise EosInputError("declared CompOSE density limit retains fewer than four rows")
        density_values = density.values[retained]
        pressure = pressure[retained]
        total_energy_density = total_energy_density[retained]
        baryon_chemical_potential = baryon_chemical_potential[retained]
        if phase_codes is not None:
            phase_codes = tuple(value for value, keep in zip(phase_codes, retained) if keep)
    else:
        density_values = density.values
    retained_maximum_euler_residual = float(np.max(normalized_euler_residual[retained]))
    retained_maximum_beta_residual = float(
        np.max(beta_equilibrium_residual[retained])
    )
    retained_maximum_cold_free_energy_residual = float(
        np.max(cold_free_energy_residual[retained])
    )
    if (
        retained_maximum_euler_residual
        > COMPOSE_EULER_CLOSURE_RELATIVE_TOLERANCE
    ):
        raise EosInputError(
            "retained CompOSE rows fail cold Euler closure: "
            f"maximum normalized residual {retained_maximum_euler_residual:.12g} "
            f"exceeds {COMPOSE_EULER_CLOSURE_RELATIVE_TOLERANCE:.12g}"
        )
    if retained_maximum_beta_residual > COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE:
        raise EosInputError(
            "retained CompOSE rows fail the beta-equilibrium condition Q5 = 0: "
            f"maximum absolute residual {retained_maximum_beta_residual:.12g} "
            f"exceeds {COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE:.12g}"
        )
    if (
        retained_maximum_cold_free_energy_residual
        > COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE
    ):
        raise EosInputError(
            "retained CompOSE rows fail the T=0 identity Q6 = Q7: "
            "maximum absolute residual "
            f"{retained_maximum_cold_free_energy_residual:.12g} exceeds "
            f"{COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE:.12g}"
        )
    phase_changes = 0
    missing_phase_codes = 0
    if phase_codes is not None:
        missing_phase_codes = sum(value is None for value in phase_codes)
        phase_changes = sum(
            left is not None and right is not None and left != right
            for left, right in zip(phase_codes, phase_codes[1:])
        )

    compose_metadata = {
        "parser_version": COMPOSE_PARSER_VERSION,
        "format_authority": COMPOSE_FORMAT_AUTHORITY,
        "model_id": model_id.strip(),
        "source_url": source_url.strip(),
        "matter_declaration": matter,
        "includes_leptons": True,
        "temperature_MeV": float(temperature.values[0]),
        "charge_fraction_grid_value": float(charge.values[0]),
        "neutron_mass_MeV": float(neutron_mass),
        "proton_mass_MeV": float(proton_mass),
        "lepton_indicator_Il": int(lepton_indicator),
        "cold_euler_closure_relative_tolerance": COMPOSE_EULER_CLOSURE_RELATIVE_TOLERANCE,
        "cold_euler_closure_source_maximum_normalized_residual": (
            source_maximum_euler_residual
        ),
        "cold_euler_closure_retained_maximum_normalized_residual": (
            retained_maximum_euler_residual
        ),
        "cold_condition_absolute_tolerance": COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE,
        "beta_equilibrium_Q5_source_maximum_absolute_residual": (
            source_maximum_beta_residual
        ),
        "beta_equilibrium_Q5_retained_maximum_absolute_residual": (
            retained_maximum_beta_residual
        ),
        "zero_temperature_Q6_minus_Q7_source_maximum_absolute_residual": (
            source_maximum_cold_free_energy_residual
        ),
        "zero_temperature_Q6_minus_Q7_retained_maximum_absolute_residual": (
            retained_maximum_cold_free_energy_residual
        ),
        "thermodynamic_duplicate_indices_last_row_wins": duplicate_count,
        "phase_codes_available": phase_codes is not None,
        "phase_code_rows_missing": missing_phase_codes,
        "phase_code_changes": phase_changes,
        "phase_codes_interpreted_as_discontinuities": False,
        "source_rows": int(len(density.values)),
        "retained_rows": int(np.count_nonzero(retained)),
        "source_baryon_density_min_fm3": float(density.values[0]),
        "source_baryon_density_max_fm3": float(density.values[-1]),
        "retained_baryon_density_min_fm3": float(density_values[0]),
        "retained_baryon_density_max_fm3": float(density_values[-1]),
        "explicit_baryon_density_max_fm3": declared_upper_density,
        "domain_selection": (
            "complete_source_table"
            if declared_upper_density is None
            else "explicit_user_declared_upper_density_no_root_inference"
        ),
        "bundle": bundle_identity,
    }
    return ComposeEos(
        name=model_id.strip(),
        energy_density_mev_fm3=total_energy_density,
        pressure_mev_fm3=pressure,
        baryon_density_fm3=density_values,
        source=source_url.strip(),
        phase_codes=phase_codes,
        baryon_chemical_potential_mev=baryon_chemical_potential,
        compose_metadata=compose_metadata,
    )


__all__ = [
    "COMPOSE_FORMAT_AUTHORITY",
    "COMPOSE_COLD_CONDITION_ABSOLUTE_TOLERANCE",
    "COMPOSE_EULER_CLOSURE_RELATIVE_TOLERANCE",
    "COMPOSE_PARSER_VERSION",
    "ComposeEos",
    "load_compose_eos",
]
