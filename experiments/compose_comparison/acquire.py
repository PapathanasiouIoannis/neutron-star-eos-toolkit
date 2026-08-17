"""Acquire the pinned CompOSE campaign archives with fail-closed verification.

This script downloads source archives only.  It does not parse an EoS, run a
stellar solver, infer a stable branch, or create scientific results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "models.json"
DEFAULT_RAW_ROOT = EXPERIMENT_ROOT / "data" / "raw"
REGISTRY_SCHEMA_VERSION = "compose-comparison-model-registry-v1"
ACQUISITION_SCHEMA_VERSION = "compose-comparison-acquisition-v1"
ORDERING_ACCEPTANCE_POLICY = (
    "both_diagnostic_reductions_complete_and_compose_catalogue_consistent"
)
_SLUG = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AcquisitionError(RuntimeError):
    """A configuration, download, or archive-integrity failure."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcquisitionError(f"{name} must be an object")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcquisitionError(f"{name} must be numeric")
    result = float(value)
    if not result > 0.0:
        raise AcquisitionError(f"{name} must be positive")
    return result


def _https(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise AcquisitionError(f"{name} must be an https URL")
    return value


def validate_config(payload: object) -> dict[str, Any]:
    """Validate and return one campaign registry without mutating it."""

    root = _mapping(payload, "registry")
    if root.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise AcquisitionError(
            f"registry schema_version must be {REGISTRY_SCHEMA_VERSION!r}"
        )
    campaign = _mapping(root.get("campaign"), "campaign")
    required = campaign.get("required_archive_members")
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(name, str) or not name for name in required)
        or len(set(required)) != len(required)
    ):
        raise AcquisitionError(
            "campaign.required_archive_members must be unique non-empty names"
        )
    if any(PurePosixPath(name).name != name for name in required):
        raise AcquisitionError("required archive members must be basenames")
    models = root.get("models")
    if not isinstance(models, list) or not models:
        raise AcquisitionError("registry.models must be a non-empty list")

    slugs: set[str] = set()
    role_counts = {"core": 0, "stress": 0}
    for index, item in enumerate(models):
        model = _mapping(item, f"models[{index}]")
        slug = model.get("slug")
        if not isinstance(slug, str) or _SLUG.fullmatch(slug) is None:
            raise AcquisitionError(f"models[{index}].slug is invalid")
        if slug in slugs:
            raise AcquisitionError(f"duplicate model slug: {slug}")
        slugs.add(slug)
        role = model.get("role")
        if role not in role_counts:
            raise AcquisitionError(f"{slug}.role must be core or stress")
        role_counts[str(role)] += 1
        if not isinstance(model.get("model_id"), str) or not model["model_id"].strip():
            raise AcquisitionError(f"{slug}.model_id must be non-empty")
        compose_id = model.get("compose_eos_id")
        if isinstance(compose_id, bool) or not isinstance(compose_id, int):
            raise AcquisitionError(f"{slug}.compose_eos_id must be an integer")
        _https(model.get("compose_page_url"), f"{slug}.compose_page_url")
        if model.get("matter") != "cold_beta_equilibrated":
            raise AcquisitionError(f"{slug}.matter must be cold_beta_equilibrated")
        if model.get("includes_leptons") is not True:
            raise AcquisitionError(f"{slug}.includes_leptons must be true")

        archive = _mapping(model.get("archive"), f"{slug}.archive")
        if archive.get("filename") != campaign.get("archive_filename"):
            raise AcquisitionError(
                f"{slug}.archive.filename must match campaign.archive_filename"
            )
        _https(archive.get("url"), f"{slug}.archive.url")
        _https(archive.get("zenodo_record_url"), f"{slug}.archive.zenodo_record_url")
        expected_bytes = archive.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise AcquisitionError(f"{slug}.archive.bytes must be a positive integer")
        expected_hash = archive.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or _SHA256.fullmatch(expected_hash) is None
        ):
            raise AcquisitionError(f"{slug}.archive.sha256 must be a lowercase SHA-256")

        optional = _mapping(
            model.get("expected_optional_files"), f"{slug}.expected_optional_files"
        )
        if any(
            not isinstance(name, str) or not isinstance(flag, bool)
            for name, flag in optional.items()
        ):
            raise AcquisitionError(
                f"{slug}.expected_optional_files must map names to booleans"
            )
        benchmark = _mapping(
            model.get("compose_benchmark"), f"{slug}.compose_benchmark"
        )
        for name in (
            "maximum_mass_msun",
            "radius_at_maximum_mass_km",
            "radius_at_1_4_msun_km",
        ):
            _positive_number(benchmark.get(name), f"{slug}.compose_benchmark.{name}")
        _https(benchmark.get("source_url"), f"{slug}.compose_benchmark.source_url")
        citations = model.get("primary_citations")
        if not isinstance(citations, list) or not citations:
            raise AcquisitionError(f"{slug}.primary_citations must be non-empty")
        for citation_index, citation_value in enumerate(citations):
            citation = _mapping(
                citation_value, f"{slug}.primary_citations[{citation_index}]"
            )
            if (
                not isinstance(citation.get("label"), str)
                or not citation["label"].strip()
            ):
                raise AcquisitionError(
                    f"{slug}.primary_citations[{citation_index}].label is empty"
                )
            _https(
                citation.get("url"),
                f"{slug}.primary_citations[{citation_index}].url",
            )
        literature = _mapping(
            model.get("literature_assessment"), f"{slug}.literature_assessment"
        )
        if (
            not isinstance(literature.get("comparability"), str)
            or not literature["comparability"].strip()
        ):
            raise AcquisitionError(
                f"{slug}.literature_assessment.comparability must be non-empty"
            )
        if (
            not isinstance(literature.get("source_label"), str)
            or not literature["source_label"].strip()
        ):
            raise AcquisitionError(
                f"{slug}.literature_assessment.source_label must be non-empty"
            )
        _https(
            literature.get("source_url"),
            f"{slug}.literature_assessment.source_url",
        )
        for name in (
            "maximum_mass_msun",
            "radius_at_maximum_mass_km",
            "radius_at_1_4_msun_km",
        ):
            value = literature.get(name)
            if value is not None:
                _positive_number(value, f"{slug}.literature_assessment.{name}")
        notes = literature.get("notes")
        if (
            not isinstance(notes, list)
            or not notes
            or any(not isinstance(note, str) or not note.strip() for note in notes)
        ):
            raise AcquisitionError(
                f"{slug}.literature_assessment.notes must be non-empty strings"
            )

        ordering_value = model.get("ordering_analysis")
        if ordering_value is not None:
            ordering = _mapping(ordering_value, f"{slug}.ordering_analysis")
            policies = (
                "diagnostic_monotone_subsequence",
                "diagnostic_keep_later_monotone_subsequence",
            )
            if ordering.get("analysis_policy") not in policies:
                raise AcquisitionError(
                    f"{slug}.ordering_analysis.analysis_policy is unsupported"
                )
            if ordering.get("acceptance_policy") != ORDERING_ACCEPTANCE_POLICY:
                raise AcquisitionError(
                    f"{slug}.ordering_analysis.acceptance_policy must be "
                    f"{ORDERING_ACCEPTANCE_POLICY!r}"
                )
            rationale = ordering.get("analysis_policy_rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise AcquisitionError(
                    f"{slug}.ordering_analysis.analysis_policy_rationale "
                    "must be non-empty"
                )
            sensitivity = ordering.get("sensitivity_policies")
            if (
                not isinstance(sensitivity, list)
                or set(sensitivity) != set(policies)
                or len(sensitivity) != len(policies)
            ):
                raise AcquisitionError(
                    f"{slug}.ordering_analysis.sensitivity_policies must contain "
                    "both diagnostic policies exactly once"
                )
            issues = ordering.get("expected_pressure_issues")
            if not isinstance(issues, list) or not issues:
                raise AcquisitionError(
                    f"{slug}.ordering_analysis.expected_pressure_issues must be non-empty"
                )
            for issue_index, issue_value in enumerate(issues):
                issue = _mapping(
                    issue_value,
                    f"{slug}.ordering_analysis.expected_pressure_issues[{issue_index}]",
                )
                for position_name in ("left_position", "right_position"):
                    position = issue.get(position_name)
                    if (
                        isinstance(position, bool)
                        or not isinstance(position, int)
                        or position < 0
                    ):
                        raise AcquisitionError(
                            f"{slug}.ordering_analysis issue {position_name} is invalid"
                        )
                for numeric_name in (
                    "left_baryon_density_fm3",
                    "right_baryon_density_fm3",
                ):
                    _positive_number(
                        issue.get(numeric_name),
                        f"{slug}.ordering_analysis issue {numeric_name}",
                    )
                relative_change = issue.get("relative_change")
                if (
                    isinstance(relative_change, bool)
                    or not isinstance(relative_change, (int, float))
                    or not float(relative_change) < 0.0
                ):
                    raise AcquisitionError(
                        f"{slug}.ordering_analysis issue relative_change must be negative"
                    )

    for role, campaign_key in (("core", "core_models"), ("stress", "stress_models")):
        declared = campaign.get(campaign_key)
        if declared != role_counts[role]:
            raise AcquisitionError(
                f"campaign.{campaign_key}={declared!r} but found {role_counts[role]}"
            )
    return dict(root)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a UTF-8 JSON registry and enforce its complete acquisition contract."""

    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AcquisitionError(f"cannot read registry: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise AcquisitionError(f"registry is not valid JSON: {resolved}") from exc
    return validate_config(payload)


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
