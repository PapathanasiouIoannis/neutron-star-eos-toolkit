"""Thin public facade for inspecting an EoS and solving stellar models.

Source-specific loading lives in :mod:`neutron_star_eos.loading`; physical TOV
work lives in :mod:`neutron_star_eos.stellar`.  This class only connects those
objects into the small workflow a user sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from neutron_star_eos.analytical import AnalyticalEos
from neutron_star_eos.capabilities import (
    CAPABILITY_NAMES,
    CAPABILITY_STATUSES,
    Capability,
    CapabilityReport,
)
from neutron_star_eos.compose import ComposeColdSlice, ComposeDataset
from neutron_star_eos.compose.thermodynamics import ComposeThermodynamicProfile
from neutron_star_eos.eos import ColdBarotrope, EosInputError, _eos_provenance_sha256
from neutron_star_eos.stellar import (
    STELLAR_VALIDATION_MODES,
    SequenceResult,
    StarResult,
    StellarConfig,
    solve_sequence,
    solve_star,
)
from neutron_star_eos.thermodynamic_view import build_thermodynamic_view
from neutron_star_eos.thermodynamics import ThermodynamicView

MODEL_KINDS = ("analytical", "csv", "compose")


@dataclass(frozen=True, slots=True)
class EosModel:
    """High-level handle for one analytical, CSV, or CompOSE input."""

    kind: str
    model_name: str
    _report: CapabilityReport
    barotrope: ColdBarotrope | None = None
    dataset: ComposeDataset | None = None
    cold_slice: ComposeColdSlice | None = None
    native_thermodynamics: ComposeThermodynamicProfile | None = None

    def report(self) -> CapabilityReport:
        return self._report

    def summary(self) -> str:
        return self._report.format_text()

    def thermodynamics(self, *, curve_points: int = 513) -> ThermodynamicView:
        """Return source-aware, read-only columns for plots and experiments."""

        return build_thermodynamic_view(self, curve_points=curve_points)

    def require_barotrope(self, *, validation_mode: str = "strict") -> ColdBarotrope:
        """Return the stellar EoS, or explain why stellar work is unavailable."""

        if validation_mode not in STELLAR_VALIDATION_MODES:
            raise ValueError(
                f"validation_mode must be one of {STELLAR_VALIDATION_MODES}"
            )
        capability = self._report.capability("continuous_barotrope")
        if self.barotrope is None:
            reason = capability.reason or "continuous stellar barotrope is unavailable"
            raise EosInputError(reason)
        if not capability.available and validation_mode != "background_diagnostic":
            reason = capability.reason or "continuous stellar barotrope is unavailable"
            raise EosInputError(reason)
        return self.barotrope

    def _require_matching_result(self, model_name: str, provenance_sha256: str) -> None:
        eos = self.barotrope
        if eos is None:
            raise EosInputError("cannot write a stellar result without a barotrope")
        expected = _eos_provenance_sha256(eos)
        if model_name != str(eos.model_name) or provenance_sha256 != expected:
            raise EosInputError(
                "stellar result does not belong to this model and EoS provenance"
            )

    def solve_star(
        self,
        central_pressure_mev_fm3: float,
        *,
        config: StellarConfig | None = None,
        retain_profile: bool = False,
        validation_mode: str = "strict",
    ) -> StarResult:
        """Integrate one TOV model at the chosen central pressure."""

        return solve_star(
            self.require_barotrope(validation_mode=validation_mode),
            central_pressure_mev_fm3,
            config=config,
            retain_profile=retain_profile,
            validation_mode=validation_mode,
        )

    def solve_sequence(
        self,
        central_pressures_mev_fm3: Iterable[float] | None = None,
        *,
        points: int = 50,
        config: StellarConfig | None = None,
        validation_mode: str = "strict",
    ) -> SequenceResult:
        """Integrate a sequence of TOV models over central pressure."""

        return solve_sequence(
            self.require_barotrope(validation_mode=validation_mode),
            central_pressures_mev_fm3,
            points=points,
            config=config,
            validation_mode=validation_mode,
        )

    def write_inspection(self, output_directory: str | Path) -> Path:
        """Write a deterministic inspection bundle to a new directory."""

        from neutron_star_eos.output import write_inspection

        return write_inspection(self, output_directory)

    def write_star(self, output_directory: str | Path, result: StarResult) -> Path:
        """Write one already-computed star and its model report."""

        self._require_matching_result(result.model_name, result.eos_provenance_sha256)
        from neutron_star_eos.output import write_star

        return write_star(self, output_directory, result)

    def write_sequence(
        self, output_directory: str | Path, result: SequenceResult
    ) -> Path:
        """Write every requested sequence attempt, including failures."""

        self._require_matching_result(result.model_name, result.eos_provenance_sha256)
        from neutron_star_eos.output import write_sequence

        return write_sequence(self, output_directory, result)

    @classmethod
    def from_analytical(
        cls,
        *,
        name: str,
        pressure_from_energy_density: Callable[[Any], Any],
        sound_speed_squared_from_energy_density: Callable[[Any], Any],
        energy_density_domain_mev_fm3: tuple[float, float],
        source: str,
        energy_density_from_pressure: Callable[[Any], Any] | None = None,
    ) -> EosModel:
        """Construct a model from user-supplied analytical ``P(epsilon)``."""

        eos = AnalyticalEos(
            name=name,
            pressure_from_energy_density=pressure_from_energy_density,
            sound_speed_squared_from_energy_density=(
                sound_speed_squared_from_energy_density
            ),
            energy_density_domain_mev_fm3=energy_density_domain_mev_fm3,
            source=source,
            energy_density_from_pressure=energy_density_from_pressure,
        )
        from neutron_star_eos.loading import _model_from_barotrope

        return _model_from_barotrope("analytical", eos)


def open_eos(
    source: str | Path | AnalyticalEos,
    *,
    kind: str | None = None,
    name: str | None = None,
    source_description: str | None = None,
    epsilon_column: str = "epsilon_mev_fm3",
    pressure_column: str = "pressure_mev_fm3",
    baryon_density_column: str | None = None,
    model_id: str | None = None,
    source_url: str | None = None,
    matter: str = "cold_beta_equilibrated",
    includes_leptons: bool = False,
    baryon_density_min_fm3: float | None = None,
    baryon_density_max_fm3: float | None = None,
    native_points: int = 2001,
    ordering_policy: str = "strict",
) -> EosModel:
    """Open one EoS through an explicit source-specific workflow."""

    from neutron_star_eos.loading import open_eos as _open_eos

    return _open_eos(
        source,
        kind=kind,
        name=name,
        source_description=source_description,
        epsilon_column=epsilon_column,
        pressure_column=pressure_column,
        baryon_density_column=baryon_density_column,
        model_id=model_id,
        source_url=source_url,
        matter=matter,
        includes_leptons=includes_leptons,
        baryon_density_min_fm3=baryon_density_min_fm3,
        baryon_density_max_fm3=baryon_density_max_fm3,
        native_points=native_points,
        ordering_policy=ordering_policy,
    )


__all__ = [
    "CAPABILITY_NAMES",
    "CAPABILITY_STATUSES",
    "MODEL_KINDS",
    "Capability",
    "CapabilityReport",
    "EosModel",
    "open_eos",
]
