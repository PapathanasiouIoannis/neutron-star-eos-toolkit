"""Diagnostic records and residual scaling for native CompOSE profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def normalized_residual(residual: np.ndarray, *terms: np.ndarray) -> np.ndarray:
    """Scale an absolute residual by the largest contributing magnitude."""

    scale = np.maximum.reduce(tuple(np.abs(term) for term in terms))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(scale > 0.0, np.abs(residual) / scale, np.nan)


@dataclass(frozen=True, slots=True)
class ComposeProfileDiagnostic:
    """One visible sampled-profile finding; never an automatic repair."""

    code: str
    severity: str
    sampled_points: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "sampled_points": self.sampled_points,
            "message": self.message,
        }


def profile_diagnostic(
    code: str,
    severity: str,
    mask: np.ndarray,
    message: str,
) -> ComposeProfileDiagnostic | None:
    """Create a finding only when at least one sampled point is affected."""

    count = int(np.count_nonzero(mask))
    if not count:
        return None
    return ComposeProfileDiagnostic(code, severity, count, message)


_normalized_residual = normalized_residual
_diagnostic = profile_diagnostic
