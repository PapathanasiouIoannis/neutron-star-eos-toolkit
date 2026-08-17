"""Validate the pinned CompOSE model registry and its scientific policies."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "models.json"
DEFAULT_RAW_ROOT = EXPERIMENT_ROOT / "data" / "raw"
REGISTRY_SCHEMA_VERSION = "compose-comparison-model-registry-v2"
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
        metadata_findings = model.get("archive_metadata_findings", [])
        if not isinstance(metadata_findings, list):
            raise AcquisitionError(f"{slug}.archive_metadata_findings must be a list")
        metadata_members: set[str] = set()
        for finding_index, finding_value in enumerate(metadata_findings):
            finding_name = f"{slug}.archive_metadata_findings[{finding_index}]"
            finding = _mapping(finding_value, finding_name)
            if finding.get("classification") != "stale_embedded_data_sheet_metadata":
                raise AcquisitionError(
                    f"{finding_name}.classification must be "
                    "'stale_embedded_data_sheet_metadata'"
                )
            member = finding.get("source_member")
            if (
                not isinstance(member, str)
                or not member
                or PurePosixPath(member).name != member
            ):
                raise AcquisitionError(f"{finding_name}.source_member is invalid")
            if member in metadata_members:
                raise AcquisitionError(
                    f"{slug}.archive_metadata_findings repeats member {member}"
                )
            metadata_members.add(member)
            member_bytes = finding.get("source_member_bytes")
            if (
                isinstance(member_bytes, bool)
                or not isinstance(member_bytes, int)
                or member_bytes <= 0
            ):
                raise AcquisitionError(
                    f"{finding_name}.source_member_bytes must be a positive integer"
                )
            member_hash = finding.get("source_member_sha256")
            if (
                not isinstance(member_hash, str)
                or _SHA256.fullmatch(member_hash) is None
            ):
                raise AcquisitionError(
                    f"{finding_name}.source_member_sha256 must be a lowercase SHA-256"
                )
            embedded_benchmark = _mapping(
                finding.get("declared_embedded_benchmark"),
                f"{finding_name}.declared_embedded_benchmark",
            )
            for name in (
                "maximum_mass_msun",
                "radius_at_maximum_mass_km",
                "radius_at_1_4_msun_km",
            ):
                _positive_number(
                    embedded_benchmark.get(name),
                    f"{finding_name}.declared_embedded_benchmark.{name}",
                )
            for name in ("cause", "interpretation"):
                value = finding.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise AcquisitionError(f"{finding_name}.{name} must be non-empty")
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
