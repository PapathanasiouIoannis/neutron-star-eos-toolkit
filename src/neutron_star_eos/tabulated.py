"""Compatibility imports for tabulated EoS inputs.

The implementation lives in :mod:`neutron_star_eos.eos.tabulated`, grouped
with the other cold-barotrope input types.
"""

from neutron_star_eos.eos.tabulated import (
    TabulatedEos,
    load_csv_eos,
)
from neutron_star_eos.eos.tabulated import (
    _deduplicate_derived_validation_grid as _deduplicate_derived_validation_grid,
)
from neutron_star_eos.eos.tabulated import (
    _validation_grid_with_exact_endpoints as _validation_grid_with_exact_endpoints,
)

__all__ = ["TabulatedEos", "load_csv_eos"]
