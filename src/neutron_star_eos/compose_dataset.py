"""Compatibility imports for the CompOSE dataset layer.

New code should import from :mod:`neutron_star_eos.compose.dataset`.
"""

from neutron_star_eos.compose import dataset as _dataset
from neutron_star_eos.compose.dataset import *  # noqa: F403

__all__ = _dataset.__all__


def __getattr__(name: str):
    """Delegate historical explicit imports, including private constants."""

    return getattr(_dataset, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_dataset)))
