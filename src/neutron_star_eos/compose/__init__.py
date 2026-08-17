"""CompOSE parsing, native thermodynamics, and stellar-barotrope adapters.

The three layers are kept separate so a source can retain useful native
thermodynamics even when a continuous stellar reduction is unavailable.
"""

from neutron_star_eos.compose.barotrope import (
    COMPOSE_BAROTROPE_SCHEMA_VERSION,
    COMPOSE_INTERPOLATION_POLICY,
    COMPOSE_ORDERING_POLICIES,
    ComposeEos,
    build_compose_eos,
    load_compose_eos,
)
from neutron_star_eos.compose.dataset import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_DATASET_SCHEMA_VERSION,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
    COMPOSE_FORMAT_AUTHORITY,
    ComposeColdSlice,
    ComposeCompositionRow,
    ComposeDataset,
    ComposeDiagnostic,
    ComposeOrderingIssue,
    ComposeSliceReport,
    ComposeThermodynamicRow,
    load_compose_dataset,
)
from neutron_star_eos.compose.mass_radius import (
    COMPOSE_MASS_RADIUS_FORMAT_AUTHORITY,
    COMPOSE_MASS_RADIUS_SCHEMA_VERSION,
    ComposeMassRadiusReference,
    load_compose_mass_radius_reference,
)
from neutron_star_eos.compose.thermodynamics import (
    COMPOSE_NATIVE_INTERPOLATION_POLICY,
    COMPOSE_NATIVE_THERMODYNAMIC_SCHEMA_VERSION,
    ComposeProfileDiagnostic,
    ComposeThermodynamicProfile,
    interpolate_compose_thermodynamics,
)

__all__ = [
    "COMPOSE_BAROTROPE_SCHEMA_VERSION",
    "COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE",
    "COMPOSE_DATASET_SCHEMA_VERSION",
    "COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE",
    "COMPOSE_FORMAT_AUTHORITY",
    "COMPOSE_INTERPOLATION_POLICY",
    "COMPOSE_MASS_RADIUS_FORMAT_AUTHORITY",
    "COMPOSE_MASS_RADIUS_SCHEMA_VERSION",
    "COMPOSE_NATIVE_INTERPOLATION_POLICY",
    "COMPOSE_NATIVE_THERMODYNAMIC_SCHEMA_VERSION",
    "COMPOSE_ORDERING_POLICIES",
    "ComposeColdSlice",
    "ComposeCompositionRow",
    "ComposeDataset",
    "ComposeDiagnostic",
    "ComposeEos",
    "ComposeMassRadiusReference",
    "ComposeOrderingIssue",
    "ComposeProfileDiagnostic",
    "ComposeSliceReport",
    "ComposeThermodynamicProfile",
    "ComposeThermodynamicRow",
    "build_compose_eos",
    "interpolate_compose_thermodynamics",
    "load_compose_dataset",
    "load_compose_eos",
    "load_compose_mass_radius_reference",
]
