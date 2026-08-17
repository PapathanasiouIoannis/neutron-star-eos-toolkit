"""Write model reports and JSON metadata with deterministic formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from neutron_star_eos.model import EosModel


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write sorted, indented UTF-8 JSON with a final newline."""

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_model_files(directory: Path, model: EosModel) -> None:
    """Write the human summary and machine-readable capability report."""

    (directory / "summary.txt").write_text(
        model.summary() + "\n", encoding="utf-8", newline="\n"
    )
    write_json(directory / "report.json", model.report().to_dict())


__all__ = ["write_json", "write_model_files"]
