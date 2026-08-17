"""Validation rules applied before solving the TOV equations."""

from __future__ import annotations

from dataclasses import replace

from neutron_star_eos.eos import ColdBarotrope, EosInputError
from neutron_star_eos.stellar.results import StarResult

STELLAR_VALIDATION_MODES = ("strict", "background_diagnostic")
BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES = frozenset({"acausal", "mechanical_instability"})


def validation_for_stellar_background(
    eos: ColdBarotrope,
    validation_mode: str,
) -> tuple[object, tuple[str, ...], str]:
    """Validate an EoS for strict or explicitly diagnostic TOV use."""

    if validation_mode not in STELLAR_VALIDATION_MODES:
        raise ValueError(f"validation_mode must be one of {STELLAR_VALIDATION_MODES}")
    report = eos.validate()
    issue_codes = tuple(item.code for item in report.issues)
    if validation_mode == "strict":
        report.require_pass()
        return report, issue_codes, "pass"
    blockers = tuple(
        code for code in issue_codes if code not in BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES
    )
    if blockers:
        raise EosInputError(
            "background diagnostic cannot bypass EoS issue(s): " + ", ".join(blockers)
        )
    status = "pass" if not issue_codes else "diagnostic_with_issues"
    return report, issue_codes, status


def annotate_validation(
    star: StarResult,
    *,
    validation_mode: str,
    validation_status: str,
    issue_codes: tuple[str, ...],
    model_name: str,
    eos_provenance_sha256: str,
) -> StarResult:
    """Attach EoS validation and provenance to a solved background."""

    return replace(
        star,
        eos_validation_mode=validation_mode,
        eos_validation_status=validation_status,
        eos_validation_issues=issue_codes,
        model_name=model_name,
        eos_provenance_sha256=eos_provenance_sha256,
    )


_validation_for_stellar_background = validation_for_stellar_background
_annotate_validation = annotate_validation
