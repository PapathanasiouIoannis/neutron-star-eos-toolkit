"""Prepare output directories and verify all pinned raw inputs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from acquire import ACQUISITION_SCHEMA_VERSION, AcquisitionError, verify_archive
from settings import DERIVED_ROOT, FIGURE_ROOT, MANIFEST_PATH, RESULTS_ROOT


def _clear_generated_tree(
    directory: Path,
    *,
    root: Path,
    allowed_suffixes: frozenset[str],
) -> None:
    """Remove one generated subtree only after validating its exact scope."""

    if not directory.exists():
        return
    resolved_root = root.resolve()
    resolved_directory = directory.resolve()
    if resolved_directory == resolved_root or not resolved_directory.is_relative_to(
        resolved_root
    ):
        raise RuntimeError(f"refusing unsafe generated-output cleanup: {directory}")

    def expected_generated_file(path: Path) -> bool:
        if path.suffix.lower() in allowed_suffixes:
            return True
        return (
            path.name.startswith(".")
            and path.name.endswith(".tmp")
            and Path(path.name[:-4]).suffix.lower() in allowed_suffixes
        )

    unexpected = [
        path
        for path in directory.rglob("*")
        if path.is_file() and not expected_generated_file(path)
    ]
    if unexpected:
        listed = ", ".join(str(path) for path in unexpected[:3])
        raise RuntimeError(
            f"refusing to clean {directory}; unexpected file type(s): {listed}"
        )
    shutil.rmtree(resolved_directory)


def _prepare_selected_outputs(models: Sequence[Mapping[str, Any]]) -> None:
    """Clear only selected generated outputs so interrupted runs cannot leak stale data."""

    for spec in models:
        slug = str(spec["slug"])
        _clear_generated_tree(
            DERIVED_ROOT / slug,
            root=DERIVED_ROOT,
            allowed_suffixes=frozenset({".csv", ".json"}),
        )
        _clear_generated_tree(
            FIGURE_ROOT / slug,
            root=FIGURE_ROOT,
            allowed_suffixes=frozenset({".png"}),
        )
    _clear_generated_tree(
        FIGURE_ROOT / "comparison",
        root=FIGURE_ROOT,
        allowed_suffixes=frozenset({".png"}),
    )
    for path in (
        RESULTS_ROOT / "all_models_summary.csv",
        RESULTS_ROOT / "acceptance.json",
        RESULTS_ROOT / "report.md",
        RESULTS_ROOT / "failure.json",
        MANIFEST_PATH,
    ):
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"refusing to replace non-file output: {path}")
            path.unlink()


def _preflight_raw_inputs(
    models: Sequence[Mapping[str, Any]],
    *,
    raw_root: Path,
    required_members: Sequence[str],
) -> None:
    """Verify all selected archives and sidecars before replacing prior outputs."""

    for spec in models:
        slug = str(spec["slug"])
        archive = raw_root / slug / str(spec["archive"]["filename"])
        verification = verify_archive(archive, spec, required_members=required_members)
        sidecar = archive.parent / "download.json"
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_schema = payload["schema_version"]
            sidecar_hash = payload["verification"]["sha256"]
            sidecar_bytes = payload["verification"]["bytes"]
            sidecar_filename = payload["archive"]["local_filename"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AcquisitionError(
                f"{slug} acquisition sidecar is missing or malformed; run acquire.py"
            ) from exc
        if (
            sidecar_schema != ACQUISITION_SCHEMA_VERSION
            or sidecar_hash != verification["sha256"]
            or sidecar_bytes != verification["bytes"]
            or sidecar_filename != archive.name
        ):
            raise AcquisitionError(
                f"{slug} acquisition sidecar does not describe the pinned archive; "
                "rerun acquire.py"
            )
