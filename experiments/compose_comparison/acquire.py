"""Acquire the pinned CompOSE campaign archives with fail-closed verification.

This script downloads source archives only.  It does not parse an EoS, run a
stellar solver, infer a stable branch, or create scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from config import (
    _SLUG,
    ACQUISITION_SCHEMA_VERSION,
    DEFAULT_CONFIG,
    DEFAULT_RAW_ROOT,
    AcquisitionError,
    _mapping,
    load_config,
)
from config import (
    ORDERING_ACCEPTANCE_POLICY as ORDERING_ACCEPTANCE_POLICY,
)
from config import (
    REGISTRY_SCHEMA_VERSION as REGISTRY_SCHEMA_VERSION,
)
from config import (
    validate_config as validate_config,
)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest."""

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
        raise TypeError("chunk_bytes must be an integer")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_path(raw_root: str | Path, model: Mapping[str, Any]) -> Path:
    """Return a traversal-safe local path for a validated model record."""

    slug = str(model["slug"])
    if _SLUG.fullmatch(slug) is None:
        raise AcquisitionError(f"unsafe model slug: {slug!r}")
    filename = str(_mapping(model["archive"], f"{slug}.archive")["filename"])
    if PurePosixPath(filename).name != filename:
        raise AcquisitionError(f"unsafe archive filename: {filename!r}")
    return Path(raw_root).expanduser().resolve() / slug / filename


def verify_archive(
    path: str | Path,
    model: Mapping[str, Any],
    *,
    required_members: Sequence[str],
) -> dict[str, Any]:
    """Verify bytes, digest, ZIP integrity, and declared member presence."""

    resolved = Path(path).expanduser().resolve()
    slug = str(model["slug"])
    archive = _mapping(model["archive"], f"{slug}.archive")
    if not resolved.is_file():
        raise AcquisitionError(f"archive is missing: {resolved}")
    actual_bytes = resolved.stat().st_size
    expected_bytes = int(archive["bytes"])
    if actual_bytes != expected_bytes:
        raise AcquisitionError(
            f"{slug} archive byte count mismatch: expected {expected_bytes}, "
            f"found {actual_bytes}"
        )
    actual_hash = sha256_file(resolved)
    expected_hash = str(archive["sha256"])
    if actual_hash != expected_hash:
        raise AcquisitionError(
            f"{slug} archive SHA-256 mismatch: expected {expected_hash}, "
            f"found {actual_hash}"
        )
    if not zipfile.is_zipfile(resolved):
        raise AcquisitionError(f"{slug} archive is not a ZIP file")
    try:
        with zipfile.ZipFile(resolved) as source:
            corrupt = source.testzip()
            if corrupt is not None:
                raise AcquisitionError(
                    f"{slug} archive has a corrupt member: {corrupt}"
                )
            members = tuple(
                sorted(info.filename for info in source.infolist() if not info.is_dir())
            )
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"{slug} archive is malformed") from exc
    basenames = [PurePosixPath(name).name for name in members]
    for name in required_members:
        count = basenames.count(name)
        if count != 1:
            raise AcquisitionError(
                f"{slug} archive must contain exactly one {name}; found {count}"
            )
    optional = _mapping(
        model["expected_optional_files"], f"{slug}.expected_optional_files"
    )
    optional_presence: dict[str, bool] = {}
    for name, expected_value in sorted(optional.items()):
        count = basenames.count(name)
        expected = bool(expected_value)
        if count > 1 or (count == 1) != expected:
            raise AcquisitionError(
                f"{slug} optional-member expectation failed for {name}: "
                f"expected {expected}, found {count}"
            )
        optional_presence[name] = count == 1
    return {
        "bytes": actual_bytes,
        "sha256": actual_hash,
        "members": list(members),
        "required_members": list(required_members),
        "optional_members": optional_presence,
        "zip_crc_check": "pass",
    }


def _download_to_temporary_file(
    url: str,
    directory: Path,
    *,
    opener: Callable[..., Any],
    timeout_seconds: float,
) -> Path:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "neutron-star-eos-toolkit-compose-acquisition/1"},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".compose-download-", suffix=".part", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        response = opener(request, timeout=timeout_seconds)
        with response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _provenance(
    model: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    archive = _mapping(model["archive"], f"{model['slug']}.archive")
    return {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "model": {
            "slug": model["slug"],
            "role": model["role"],
            "model_id": model["model_id"],
            "compose_eos_id": model["compose_eos_id"],
            "compose_page_url": model["compose_page_url"],
        },
        "archive": {
            "local_filename": archive["filename"],
            "source_url": archive["url"],
            "zenodo_record_url": archive["zenodo_record_url"],
            "expected_bytes": archive["bytes"],
            "expected_sha256": archive["sha256"],
        },
        "verification": dict(verification),
        "determinism": {
            "retrieval_timestamp_recorded": False,
            "offline_reuse_requires_identical_bytes": True,
            "existing_archives_overwritten": False,
        },
    }


def acquire_model(
    model: Mapping[str, Any],
    *,
    raw_root: str | Path,
    required_members: Sequence[str],
    offline: bool,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Download a missing archive or deterministically reuse an exact local copy."""

    target = archive_path(raw_root, model)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verification = verify_archive(target, model, required_members=required_members)
        status = "verified_existing"
    else:
        if offline:
            raise AcquisitionError(
                f"offline mode cannot acquire missing archive: {target}"
            )
        archive = _mapping(model["archive"], f"{model['slug']}.archive")
        try:
            temporary = _download_to_temporary_file(
                str(archive["url"]),
                target.parent,
                opener=opener,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            raise AcquisitionError(
                f"download failed for {model['slug']}: {archive['url']}"
            ) from exc
        try:
            verification = verify_archive(
                temporary, model, required_members=required_members
            )
            if target.exists():
                raise AcquisitionError(
                    f"refusing to overwrite archive created concurrently: {target}"
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        status = "downloaded"
    sidecar = target.parent / "download.json"
    _write_json_atomic(sidecar, _provenance(model, verification))
    return {
        "slug": model["slug"],
        "status": status,
        "archive": str(target),
        "sidecar": str(sidecar),
        "sha256": verification["sha256"],
        "bytes": verification["bytes"],
    }


def selected_models(
    config: Mapping[str, Any], selected_slugs: Iterable[str] | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Resolve an ordered optional selection and reject unknown or duplicate slugs."""

    models = tuple(_mapping(value, "model") for value in config["models"])
    if selected_slugs is None:
        return models
    requested = tuple(selected_slugs)
    if len(requested) != len(set(requested)):
        raise AcquisitionError("model selection contains duplicate slugs")
    by_slug = {str(model["slug"]): model for model in models}
    unknown = [slug for slug in requested if slug not in by_slug]
    if unknown:
        raise AcquisitionError("unknown model slug(s): " + ", ".join(unknown))
    return tuple(by_slug[slug] for slug in requested)


def acquire_models(
    config: Mapping[str, Any],
    *,
    raw_root: str | Path,
    selected_slugs: Iterable[str] | None = None,
    offline: bool,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[dict[str, Any], ...]:
    campaign = _mapping(config["campaign"], "campaign")
    required = tuple(str(name) for name in campaign["required_archive_members"])
    return tuple(
        acquire_model(
            model,
            raw_root=raw_root,
            required_members=required,
            offline=offline,
            opener=opener,
        )
        for model in selected_models(config, selected_slugs)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire and verify pinned CompOSE comparison archives."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Acquire one registry slug; repeat to preserve a chosen order.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access and verify only existing exact archives.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        models = selected_models(config, arguments.models)
    except AcquisitionError as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True))
        return 2

    campaign = _mapping(config["campaign"], "campaign")
    required = tuple(str(name) for name in campaign["required_archive_members"])
    results: list[dict[str, Any]] = []
    failed = False
    for model in models:
        try:
            result = acquire_model(
                model,
                raw_root=arguments.raw_root,
                required_members=required,
                offline=arguments.offline,
            )
        except AcquisitionError as exc:
            failed = True
            result = {
                "slug": model["slug"],
                "status": "error",
                "reason": str(exc),
            }
        results.append(result)
    print(
        json.dumps(
            {
                "schema_version": ACQUISITION_SCHEMA_VERSION,
                "offline": arguments.offline,
                "results": results,
                "status": "error" if failed else "complete",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
