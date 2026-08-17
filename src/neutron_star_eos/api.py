"""Small public API for the common neutron-star EoS workflow."""

from neutron_star_eos.capabilities import Capability, CapabilityReport
from neutron_star_eos.model import EosModel, open_eos

__all__ = ["Capability", "CapabilityReport", "EosModel", "open_eos"]
