"""Readable public entry points for neutron-star background calculations."""

from neutron_star_eos.stellar.configuration import (
    DEFAULT_STELLAR_CONFIG,
    StellarConfig,
)
from neutron_star_eos.stellar.constants import (
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
from neutron_star_eos.stellar.results import (
    SequenceAttempt,
    SequenceResult,
    StarResult,
    StellarSolveError,
)
from neutron_star_eos.stellar.sequence import solve_sequence
from neutron_star_eos.stellar.star import solve_star
from neutron_star_eos.stellar.validation import (
    BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES,
    STELLAR_VALIDATION_MODES,
)

__all__ = [
    "DEFAULT_STELLAR_CONFIG",
    "GRAVITY_CONVERSION",
    "MEV_J",
    "FM3_M3",
    "NEWTONIAN_GRAVITATIONAL_CONSTANT_M3_KG_S2",
    "SOLAR_MASS_KG",
    "SOLAR_MASS_LENGTH_KM",
    "SPEED_OF_LIGHT_M_S",
    "STELLAR_CONSTANT_AUTHORITY",
    "STELLAR_CONSTANT_REFERENCE_URL",
    "BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES",
    "SequenceAttempt",
    "SequenceResult",
    "StarResult",
    "StellarSolveError",
    "StellarConfig",
    "STELLAR_VALIDATION_MODES",
    "solve_sequence",
    "solve_star",
]
