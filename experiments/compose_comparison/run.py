"""Run the pinned CompOSE campaign.

Usage: ``python experiments/compose_comparison/run.py``

The readable top-level workflow is in :func:`campaign_cli.main`; focused
scientific steps live in the neighbouring sampling, calculate, compare,
acceptance, figures, report, and provenance modules.  The imports below keep
historical research-test names available during the structural transition.
"""

from __future__ import annotations

import sys

from acceptance import (  # noqa: F401
    _campaign_interpretive_status,
    _optional_reference_campaign_findings,
    _ordering_analysis_assessment,
)
from acquire import AcquisitionError  # noqa: F401
from calculate import _causal_endpoint  # noqa: F401
from campaign_cli import main
from compare import _literature_check  # noqa: F401
from figures import _save_ax, _save_comparison_plots, plt  # noqa: F401
from files import _preflight_raw_inputs, _prepare_selected_outputs  # noqa: F401
from provenance import (  # noqa: F401
    _is_canonical_code_input,
    _manifest,
    scipy_version,
    toolkit_version,
)
from reference import (  # noqa: F401
    _archive_metadata_findings,
    _comparison_to_reference,
    _eos_mr_comparison_coverage,
    _eos_mr_source_consistency,
)
from report import (  # noqa: F401
    _archive_metadata_report_line,
    _causality_report_text,
    _optional_reference_report_line,
    _ordering_systematic_report_line,
)
from sampling import (  # noqa: F401
    _branch_metrics,
    _cs2_within_causal_threshold,
    _merge_sequences,
)
from settings import (  # noqa: F401
    CAUSALITY_THRESHOLD_TOLERANCE,
    DERIVED_ROOT,
    FIGURE_ROOT,
    MANIFEST_PATH,
    RESULTS_ROOT,
    RUN_SCHEMA_VERSION,
    BranchData,
)
from summary import SUMMARY_FIELDS, _summary_rows  # noqa: F401

from neutron_star_eos import SequenceAttempt, SequenceResult  # noqa: F401
from neutron_star_eos.compose import ComposeMassRadiusReference  # noqa: F401
from neutron_star_eos.stellar import (  # noqa: F401
    GRAVITY_CONVERSION,
    STELLAR_CONSTANT_AUTHORITY,
)

if __name__ == "__main__":
    sys.exit(main())
