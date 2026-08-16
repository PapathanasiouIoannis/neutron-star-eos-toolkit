"""Compatibility imports for the CompOSE native-thermodynamics layer.

New code should import from :mod:`neutron_star_eos.compose.thermodynamics`.
"""

from neutron_star_eos.compose import thermodynamics as _thermodynamics
from neutron_star_eos.compose.thermodynamics import *  # noqa: F403

__all__ = _thermodynamics.__all__


def __getattr__(name: str):
    """Delegate historical explicit imports to the relocated module."""

    return getattr(_thermodynamics, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_thermodynamics)))
