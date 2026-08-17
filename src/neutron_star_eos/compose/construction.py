"""Select source rows and construct a continuous CompOSE stellar EoS."""

from __future__ import annotations

from pathlib import Path

from neutron_star_eos.compose.dataset import (
    ComposeColdSlice,
    ComposeDataset,
    load_compose_dataset,
)
from neutron_star_eos.eos import EosInputError

from .barotrope import COMPOSE_ORDERING_POLICIES, ComposeEos


def build_compose_eos(
    dataset_or_slice: ComposeDataset | ComposeColdSlice,
    *,
    matter: str = "cold_beta_equilibrated",
    includes_leptons: bool = True,
    baryon_density_min_fm3: float | None = None,
    baryon_density_max_fm3: float | None = None,
    ordering_policy: str = "strict",
) -> ComposeEos:
    """Construct one explicitly selected continuous CompOSE barotrope.

    ``strict`` preserves every selected source row and fails if the result is
    not invertible.  The two diagnostic policies keep either the earlier or
    later row at a local ordering conflict.  They record every omitted source
    position and are sensitivity checks, not physical transition models.
    """

    if isinstance(dataset_or_slice, ComposeDataset):
        cold_slice = dataset_or_slice.cold_beta_equilibrium_slice(
            matter=matter,
            includes_leptons=includes_leptons,
        )
    elif isinstance(dataset_or_slice, ComposeColdSlice):
        cold_slice = dataset_or_slice
    else:
        raise TypeError("build_compose_eos expects ComposeDataset or ComposeColdSlice")
    if ordering_policy not in COMPOSE_ORDERING_POLICIES:
        raise EosInputError(
            f"ordering_policy must be one of {COMPOSE_ORDERING_POLICIES}"
        )
    source_slice_report = cold_slice.report()
    selected_source = cold_slice.selected_domain(
        baryon_density_min_fm3=baryon_density_min_fm3,
        baryon_density_max_fm3=baryon_density_max_fm3,
    )
    selected = selected_source
    if ordering_policy == "diagnostic_monotone_subsequence":
        selected = selected_source.diagnostic_monotone_subsequence(
            conflict_policy="keep_first"
        )
    elif ordering_policy == "diagnostic_keep_later_monotone_subsequence":
        selected = selected_source.diagnostic_monotone_subsequence(
            conflict_policy="keep_later"
        )
    retained_positions = set(selected.source_positions)
    omitted_positions = tuple(
        position
        for position in selected_source.source_positions
        if position not in retained_positions
    )
    return ComposeEos(
        cold_slice=selected,
        source_slice_report=source_slice_report,
        selection={
            "requested_baryon_density_min_fm3": baryon_density_min_fm3,
            "requested_baryon_density_max_fm3": baryon_density_max_fm3,
            "last_retained_source_node_fm3": float(selected.baryon_density_fm3[-1]),
            "ordering_policy": ordering_policy,
            "selected_source_rows_before_ordering_policy": len(selected_source.rows),
            "retained_source_positions": list(selected.source_positions),
            "omitted_source_positions": list(omitted_positions),
            "omitted_source_rows": len(omitted_positions),
            "diagnostic_reduction_is_physical_transition_policy": False,
        },
    )


def load_compose_eos(
    path_or_zip: str | Path,
    *,
    model_id: str,
    source_url: str,
    matter: str,
    includes_leptons: bool,
    baryon_density_min_fm3: float | None = None,
    baryon_density_max_fm3: float | None = None,
    ordering_policy: str = "strict",
) -> ComposeEos:
    """Parse a dataset and build its explicitly selected stellar barotrope."""

    if matter != "cold_beta_equilibrated":
        raise EosInputError(
            "v2 stellar CompOSE input must explicitly be cold beta-equilibrated matter"
        )
    if includes_leptons is not True:
        raise EosInputError("v2 stellar CompOSE input must explicitly include leptons")
    dataset = load_compose_dataset(
        path_or_zip, model_id=model_id, source_url=source_url
    )
    return build_compose_eos(
        dataset,
        matter=matter,
        includes_leptons=includes_leptons,
        baryon_density_min_fm3=baryon_density_min_fm3,
        baryon_density_max_fm3=baryon_density_max_fm3,
        ordering_policy=ordering_policy,
    )


__all__ = ["build_compose_eos", "load_compose_eos"]
