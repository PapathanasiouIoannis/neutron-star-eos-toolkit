"""The small physical interface shared by every cold barotrope."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from neutron_star_eos.eos.validation import EosValidationReport

EOS_INPUT_SCHEMA_VERSION = "cold_barotrope_input_v1"
CANONICAL_ENERGY_DENSITY_CONVENTION = "total_including_rest_mass"
CANONICAL_UNITS = "MeV/fm^3"


class EosInputError(ValueError):
    """Base error for invalid user-supplied EoS material."""


class EosDomainError(EosInputError):
    """Raised when an EoS is evaluated outside its declared domain."""


@runtime_checkable
class ColdBarotrope(Protocol):
    """Continuous cold-barotrope interface used by stellar calculations.

    Pressure and total energy density include rest mass and use MeV fm^-3.
    Sound speed squared is dP/d(epsilon) with c=1.
    """

    model_name: str
    pressure_min_mev_fm3: float
    pressure_max_mev_fm3: float
    energy_density_min_mev_fm3: float
    energy_density_max_mev_fm3: float
    surface_boundary_kind: str
    tidal_capability_status: str

    def pressure_from_energy_density(self, energy_density_mev_fm3: Any) -> Any: ...

    def energy_density_from_pressure(self, pressure_mev_fm3: Any) -> Any: ...

    def sound_speed_squared_from_energy_density(
        self, energy_density_mev_fm3: Any
    ) -> Any: ...

    def validate(self, *, points: int = 2049) -> EosValidationReport: ...

    def provenance(self) -> dict[str, Any]: ...

    def __call__(self, pressure_mev_fm3: float) -> tuple[float, float]: ...


def eos_provenance_sha256(eos: ColdBarotrope) -> str:
    """Hash the exact declared EoS provenance using canonical JSON."""

    encoded = json.dumps(
        eos.provenance(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finite_float(name: str, value: Any) -> float:
    """Return a finite float or raise a user-facing EoS input error."""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise EosInputError(f"{name} must be finite")
    return result


def domain_values(
    value: Any,
    *,
    name: str,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, bool]:
    """Convert numeric input and enforce one adapter's declared domain."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise EosInputError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise EosInputError(f"{name} must contain only finite values")
    if np.any(array < lower) or np.any(array > upper):
        raise EosDomainError(
            f"{name} is outside the declared interval [{lower!r}, {upper!r}]"
        )
    return array, array.ndim == 0


def scalar_or_array(value: np.ndarray, scalar: bool) -> float | np.ndarray:
    """Preserve scalar input while returning arrays for array input."""

    return float(np.asarray(value)) if scalar else np.asarray(value, dtype=float)


def evaluate_user_function(
    function: Callable[[Any], Any], values: np.ndarray, *, name: str
) -> np.ndarray:
    """Evaluate NumPy-aware or ordinary scalar analytical functions."""

    try:
        result = np.asarray(function(values), dtype=float)
        if result.shape == values.shape:
            return result
    except (TypeError, ValueError, ArithmeticError):
        pass
    try:
        result = np.asarray(
            [float(function(float(value))) for value in values.reshape(-1)],
            dtype=float,
        ).reshape(values.shape)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise EosInputError(
            f"analytical {name} could not evaluate the declared domain"
        ) from exc
    return result


# Historical private aliases retained for existing adapters and scripts.
_eos_provenance_sha256 = eos_provenance_sha256
_finite_float = finite_float
_domain_values = domain_values
_scalar_or_array = scalar_or_array
_evaluate_user_function = evaluate_user_function
