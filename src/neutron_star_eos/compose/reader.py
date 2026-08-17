"""Read and preserve the source layer of a CompOSE dataset."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from neutron_star_eos.compose.cold_slice import ComposeColdSlice
from neutron_star_eos.compose.records import (
    COMPOSE_DATASET_SCHEMA_VERSION,
    COMPOSE_FORMAT_AUTHORITY,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    ComposeAxis,
    ComposeCompositionRow,
    ComposeThermodynamicRow,
)
from neutron_star_eos.eos import EosInputError


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ascii(name: str, data: bytes) -> list[str]:
    try:
        return data.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise EosInputError(f"CompOSE file {name} is not ASCII") from exc


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
    """Read required and optional source files from a directory or ZIP."""

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
    """Parse indexed Q quantities and preserve every additional value."""

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
    """Preserve phase codes and model-defined composition payload tokens."""

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


class ComposeDataset:
    """Immutable source data, independent of interpolation and stellar use."""

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
        """Select the explicit T=0, Yq=0 density path for stellar matter."""

        if matter != "cold_beta_equilibrated":
            raise EosInputError(
                "cold stellar reduction requires an explicit cold "
                "beta-equilibrated matter declaration"
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
                "cold stellar reduction requires one declared beta-equilibrium "
                "density path"
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
                "selected CompOSE thermodynamic path is incomplete: "
                f"{len(missing)} rows missing"
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


__all__ = ["ComposeDataset", "load_compose_dataset"]
