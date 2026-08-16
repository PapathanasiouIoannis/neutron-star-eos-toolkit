"""Small, fail-closed tools for continuous cold neutron-star barotropes."""

from neutron_star_eos.compose import (
    COMPOSE_ORDERING_POLICIES,
    ComposeColdSlice,
    ComposeDataset,
    ComposeEos,
    ComposeSliceReport,
    build_compose_eos,
    load_compose_dataset,
    load_compose_eos,
)
from neutron_star_eos.compose_thermodynamics import (
    COMPOSE_NATIVE_INTERPOLATION_POLICY,
    ComposeProfileDiagnostic,
    ComposeThermodynamicProfile,
    interpolate_compose_thermodynamics,
)
from neutron_star_eos.eos import (
    AnalyticalEos,
    ColdBarotrope,
    EosDomainError,
    EosInputError,
    EosValidationIssue,
    EosValidationReport,
    validate_eos,
)
from neutron_star_eos.model import (
    Capability,
    CapabilityReport,
    EosModel,
    open_eos,
)
from neutron_star_eos.stellar import (
    BACKGROUND_DIAGNOSTIC_ALLOWED_ISSUES,
    DEFAULT_STELLAR_CONFIG,
    SequenceAttempt,
    SequenceResult,
    StarResult,
    StellarConfig,
    STELLAR_VALIDATION_MODES,
    solve_sequence,
    solve_star,
)
from neutron_star_eos.tabulated import TabulatedEos, load_csv_eos

# Keep the wildcard/documentation surface small.  The advanced compatibility
# names imported above remain available for explicit imports used by existing
# scripts, while newcomers see the facade and its two common configuration
# types first.
__all__ = [
    "Capability",
    "CapabilityReport",
    "EosInputError",
    "EosModel",
    "StellarConfig",
    "open_eos",
]
