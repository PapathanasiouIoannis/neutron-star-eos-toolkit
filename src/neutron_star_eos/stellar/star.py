"""Calculate one continuous neutron-star background from an EoS."""

from __future__ import annotations

from neutron_star_eos.eos import ColdBarotrope, _eos_provenance_sha256
from neutron_star_eos.stellar.configuration import (
    StellarConfig,
    checked_config,
    finite_float,
)
from neutron_star_eos.stellar.results import StarResult
from neutron_star_eos.stellar.tov import solve_validated_star
from neutron_star_eos.stellar.validation import (
    annotate_validation,
    validation_for_stellar_background,
)


def solve_star(
    eos: ColdBarotrope,
    central_pressure_mev_fm3: float,
    *,
    retain_profile: bool = False,
    config: StellarConfig | None = None,
    validation_mode: str = "strict",
) -> StarResult:
    """Validate an EoS and calculate one truncated TOV background.

    ``central_pressure_mev_fm3`` is the pressure at the center in MeV fm^-3.
    ``background_diagnostic`` permits only declared causality or sound-speed
    findings and records them on the result; it never turns them into a pass.
    """

    _report, issue_codes, validation_status = validation_for_stellar_background(
        eos, validation_mode
    )
    resolved_config = checked_config(config)
    model_name = str(eos.model_name)
    provenance_sha256 = _eos_provenance_sha256(eos)
    central_pressure = finite_float(
        "central_pressure_mev_fm3", central_pressure_mev_fm3
    )
    star = solve_validated_star(
        eos,
        central_pressure,
        config=resolved_config,
        retain_profile=bool(retain_profile),
        validation_mode=validation_mode,
    )
    return annotate_validation(
        star,
        validation_mode=validation_mode,
        validation_status=validation_status,
        issue_codes=issue_codes,
        model_name=model_name,
        eos_provenance_sha256=provenance_sha256,
    )
