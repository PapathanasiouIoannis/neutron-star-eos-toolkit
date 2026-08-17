"""User-supplied analytical cold-barotrope adapter.

The implementation remains compatible with the historical
``neutron_star_eos.eos.AnalyticalEos`` import; this focused module is the
preferred home for new source-specific code.
"""

from neutron_star_eos.eos import AnalyticalEos

__all__ = ["AnalyticalEos"]
