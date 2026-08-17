"""Record exact code, data, software, and physical-constant provenance."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from acquire import sha256_file
from campaign_io import write_json
from scipy import __version__ as scipy_version
from settings import (
    DERIVED_ROOT,
    EXPERIMENT_ROOT,
    FIGURE_ROOT,
    MANIFEST_PATH,
    RESULTS_ROOT,
    RUN_SCHEMA_VERSION,
)

from neutron_star_eos import __version__ as toolkit_version
from neutron_star_eos.stellar import (
    FM3_M3,
    GRAVITY_CONVERSION,
    MEV_J,
    NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2,
    SOLAR_MASS_KG,
    SOLAR_MASS_LENGTH_KM,
    SPEED_OF_LIGHT_M_S,
    STELLAR_CONSTANT_AUTHORITY,
    STELLAR_CONSTANT_REFERENCE_URL,
)


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=EXPERIMENT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def _path_label(path: Path, *, base: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _hashed_file(path: Path, *, base: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"manifest input is missing: {path}")
    return {
        "path": _path_label(path, base=base),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _manifest(
    summaries: Sequence[Mapping[str, Any]],
    config_path: Path,
    *,
    raw_root: Path,
) -> None:
    repository_root = EXPERIMENT_ROOT.parents[1]
    raw_files: list[dict[str, Any]] = []
    generated_files: list[dict[str, Any]] = []
    selected_slugs = [str(item["slug"]) for item in summaries]

    for summary in summaries:
        slug = str(summary["slug"])
        archive = raw_root / slug / str(summary["archive"]["archive_filename"])
        sidecar = archive.parent / "download.json"
        raw_files.extend(
            (
                _hashed_file(archive, base=EXPERIMENT_ROOT, role="pinned_raw_archive"),
                _hashed_file(
                    sidecar,
                    base=EXPERIMENT_ROOT,
                    role="acquisition_verification_sidecar",
                ),
            )
        )
        for root, role in (
            (DERIVED_ROOT / slug, "selected_model_derived_data"),
            (FIGURE_ROOT / slug, "selected_model_figure"),
        ):
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                generated_files.append(
                    _hashed_file(path, base=EXPERIMENT_ROOT, role=role)
                )

    comparison = FIGURE_ROOT / "comparison"
    if comparison.exists():
        for path in sorted(item for item in comparison.rglob("*") if item.is_file()):
            generated_files.append(
                _hashed_file(
                    path, base=EXPERIMENT_ROOT, role="selected_campaign_figure"
                )
            )
    for path in (
        RESULTS_ROOT / "all_models_summary.csv",
        RESULTS_ROOT / "acceptance.json",
        RESULTS_ROOT / "report.md",
    ):
        generated_files.append(
            _hashed_file(path, base=EXPERIMENT_ROOT, role="selected_campaign_result")
        )

    code_paths = [
        *sorted(EXPERIMENT_ROOT.glob("*.py")),
        config_path,
        repository_root / "pyproject.toml",
    ]
    source_root = repository_root / "src" / "neutron_star_eos"
    code_paths.extend(
        sorted(
            path for path in source_root.rglob("*") if _is_canonical_code_input(path)
        )
    )
    unique_code_paths = tuple(dict.fromkeys(path.resolve() for path in code_paths))
    code_inputs = [
        _hashed_file(path, base=repository_root, role="exact_code_input")
        for path in unique_code_paths
    ]
    files = [*raw_files, *generated_files]
    write_json(
        MANIFEST_PATH,
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "config": {
                "path": _path_label(config_path, base=EXPERIMENT_ROOT),
                "sha256": sha256_file(config_path),
            },
            "software": {
                "neutron_star_eos_toolkit": toolkit_version,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy_version,
                "matplotlib": matplotlib.__version__,
                "platform": platform.platform(),
                "git": _git_state(),
                "stellar_physical_constants": {
                    "authority": STELLAR_CONSTANT_AUTHORITY,
                    "authority_url": STELLAR_CONSTANT_REFERENCE_URL,
                    "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
                    "newtonian_gravitational_constant_m3_kg_s2": (
                        NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2
                    ),
                    "solar_mass_kg": SOLAR_MASS_KG,
                    "MeV_J": MEV_J,
                    "fm3_m3": FM3_M3,
                    "gravity_conversion_Msun_per_km3_per_MeV_fm3": (GRAVITY_CONVERSION),
                    "solar_mass_length_km": SOLAR_MASS_LENGTH_KM,
                },
            },
            "models": selected_slugs,
            "enumeration_policy": {
                "raw": (
                    "selected canonical archive and download.json only; legacy live "
                    "downloads and unselected model files are excluded"
                ),
                "generated": (
                    "selected model outputs plus freshly cleaned campaign comparison "
                    "and result artifacts only"
                ),
            },
            "canonical_raw_files": raw_files,
            "generated_artifacts": generated_files,
            "exact_code_inputs": code_inputs,
            "files": files,
            "raw_archives_relicensed_under_mit": False,
        },
    )


def _is_canonical_code_input(path: Path) -> bool:
    """Return whether *path* is a canonical, executable package input."""

    return (
        path.is_file()
        and ".ipynb_checkpoints" not in path.parts
        and not path.match("*-checkpoint.*")
        and (path.suffix in {".py", ".mplstyle"} or path.name == "py.typed")
    )
