"""Publish complete result directories atomically and without overwriting."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from neutron_star_eos.eos import EosInputError
from neutron_star_eos.output.metadata import write_json, write_model_files
from neutron_star_eos.output.tables import write_sequence_table, write_thermodynamics
from neutron_star_eos.stellar import SequenceResult, StarResult

if TYPE_CHECKING:
    from neutron_star_eos.model import EosModel


def _write_new_bundle(destination: str | Path, writer: Callable[[Path], None]) -> Path:
    """Build in a temporary directory, then atomically publish the bundle."""

    resolved = Path(destination).expanduser().resolve()
    if resolved.exists():
        raise EosInputError(f"output directory already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{resolved.name}-", dir=resolved.parent))
    try:
        writer(temporary)
        os.replace(temporary, resolved)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return resolved


def write_inspection(model: EosModel, output_directory: str | Path) -> Path:
    """Write a deterministic inspection bundle to a new directory."""

    def writer(temporary: Path) -> None:
        write_model_files(temporary, model)
        if model.report().capability("thermodynamics").available:
            write_thermodynamics(temporary / "thermodynamics.csv", model)

    return _write_new_bundle(output_directory, writer)


def write_star(
    model: EosModel, output_directory: str | Path, result: StarResult
) -> Path:
    """Write one already-computed stellar background and its model report."""

    if not isinstance(result, StarResult):
        raise EosInputError("write_star requires a StarResult")
    model._require_matching_result(result.model_name, result.eos_provenance_sha256)

    def writer(temporary: Path) -> None:
        write_model_files(temporary, model)
        write_json(temporary / "star.json", result.to_dict())

    return _write_new_bundle(output_directory, writer)


def write_sequence(
    model: EosModel, output_directory: str | Path, result: SequenceResult
) -> Path:
    """Write all sequence attempts, including failures, to a new directory."""

    if not isinstance(result, SequenceResult):
        raise EosInputError("write_sequence requires a SequenceResult")
    model._require_matching_result(result.model_name, result.eos_provenance_sha256)

    def writer(temporary: Path) -> None:
        write_model_files(temporary, model)
        write_json(temporary / "sequence.json", result.to_dict())
        write_sequence_table(temporary / "sequence.csv", result)

    return _write_new_bundle(output_directory, writer)


__all__ = ["write_inspection", "write_sequence", "write_star"]
