"""Atomic plotting functions grouped by their physical subject."""

from neutron_star_eos.plotting.compose import (
    plot_compose_closure_residuals,
    plot_compose_cold_residuals,
    plot_compose_free_energy_closure_residuals,
    plot_composition,
    plot_phase_codes,
)
from neutron_star_eos.plotting.eos import (
    plot_pressure_energy,
    plot_sound_speed_squared,
)
from neutron_star_eos.plotting.stellar import (
    plot_mass_profile,
    plot_mass_radius,
    plot_sequence_status,
)

__all__ = [
    "plot_compose_closure_residuals",
    "plot_compose_cold_residuals",
    "plot_compose_free_energy_closure_residuals",
    "plot_composition",
    "plot_mass_profile",
    "plot_mass_radius",
    "plot_phase_codes",
    "plot_pressure_energy",
    "plot_sequence_status",
    "plot_sound_speed_squared",
]
