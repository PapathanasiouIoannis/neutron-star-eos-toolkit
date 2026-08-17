"""Compatibility surface for the split CompOSE source layer.

New code can read :mod:`reader`, inspect immutable :mod:`records`, and select a
cold path with :mod:`cold_slice` directly.
"""

from neutron_star_eos.compose.cold_slice import ComposeColdSlice
from neutron_star_eos.compose.reader import ComposeDataset, load_compose_dataset
from neutron_star_eos.compose.records import (
    COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE,
    COMPOSE_DATASET_SCHEMA_VERSION,
    COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE,
    COMPOSE_FORMAT_AUTHORITY,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    ComposeAxis,
    ComposeCompositionRow,
    ComposeDiagnostic,
    ComposeOrderingIssue,
    ComposeSliceReport,
    ComposeThermodynamicRow,
)

__all__ = [
    "COMPOSE_COLD_DIAGNOSTIC_ABSOLUTE_TOLERANCE",
    "COMPOSE_DATASET_SCHEMA_VERSION",
    "COMPOSE_EULER_DIAGNOSTIC_RELATIVE_TOLERANCE",
    "COMPOSE_FORMAT_AUTHORITY",
    "OPTIONAL_FILES",
    "REQUIRED_FILES",
    "ComposeAxis",
    "ComposeColdSlice",
    "ComposeCompositionRow",
    "ComposeDataset",
    "ComposeDiagnostic",
    "ComposeOrderingIssue",
    "ComposeSliceReport",
    "ComposeThermodynamicRow",
    "load_compose_dataset",
]
