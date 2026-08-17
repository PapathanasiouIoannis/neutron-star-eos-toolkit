"""Preserve optional CompOSE composition and microphysics quantities."""

from __future__ import annotations

import math

import numpy as np

from neutron_star_eos.compose.cold_slice import ComposeColdSlice
from neutron_star_eos.eos import EosInputError


def parse_composition_payload(
    tokens: tuple[str, ...],
) -> tuple[dict[int, float], dict[int, tuple[float, float, float]]]:
    """Parse pair and quadruple payloads without interpreting their codes."""

    try:
        pair_count = int(tokens[0])
        if pair_count < 0:
            raise ValueError
        cursor = 1
        pairs: dict[int, float] = {}
        for _ in range(pair_count):
            pairs[int(tokens[cursor])] = float(tokens[cursor + 1])
            cursor += 2
        quadruple_count = int(tokens[cursor])
        cursor += 1
        if quadruple_count < 0:
            raise ValueError
        quadruples: dict[int, tuple[float, float, float]] = {}
        for _ in range(quadruple_count):
            code = int(tokens[cursor])
            quadruples[code] = (
                float(tokens[cursor + 1]),
                float(tokens[cursor + 2]),
                float(tokens[cursor + 3]),
            )
            cursor += 4
        if cursor != len(tokens):
            raise ValueError
    except (IndexError, ValueError) as exc:
        raise EosInputError("CompOSE composition payload is malformed") from exc
    return pairs, quadruples


def parse_microphysics(
    cold_slice: ComposeColdSlice,
) -> dict[tuple[int, int, int], dict[int, float]]:
    """Parse optional eos.micro code-value pairs for the selected source."""

    if "eos.micro" not in cold_slice.dataset.available_files:
        return {}
    try:
        lines = (
            cold_slice.dataset.source_file_bytes("eos.micro")
            .decode("ascii")
            .splitlines()
        )
    except UnicodeDecodeError as exc:
        raise EosInputError("CompOSE eos.micro is not ASCII") from exc
    parsed: dict[tuple[int, int, int], dict[int, float]] = {}
    for line_number, line in enumerate(lines, start=1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            key = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
            count = int(tokens[3])
            if count < 0 or len(tokens) != 4 + 2 * count:
                raise ValueError
            values = {
                int(tokens[4 + 2 * index]): float(tokens[5 + 2 * index])
                for index in range(count)
            }
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError
        except (IndexError, ValueError) as exc:
            raise EosInputError(
                f"CompOSE eos.micro line {line_number} is malformed"
            ) from exc
        parsed[key] = values
    return parsed


def optional_source_fields(
    cold_slice: ComposeColdSlice,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    """Return optional source columns, units, and source-defined descriptions."""

    rows = cold_slice.rows
    count = len(rows)
    values: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    descriptions: dict[str, str] = {}

    maximum_width = max((len(row.additional_values) for row in rows), default=0)
    for index in range(maximum_width):
        name = f"additional_{index + 1}"
        values[name] = np.asarray(
            [
                row.additional_values[index]
                if index < len(row.additional_values)
                else np.nan
                for row in rows
            ]
        )
        units[name] = "source-defined"
        descriptions[name] = (
            f"CompOSE additional thermodynamic quantity {index + 1}; "
            "missing source-row values remain NaN"
        )

    composition_map = {row.key: row for row in cold_slice.dataset.composition_rows}
    parsed_composition = []
    for row in rows:
        source = composition_map.get(row.key)
        parsed_composition.append(
            ({}, {})
            if source is None
            else parse_composition_payload(source.raw_payload_tokens)
        )
    pair_codes = sorted(
        {code for pairs, _quadruples in parsed_composition for code in pairs}
    )
    for code in pair_codes:
        name = f"composition_pair_{code}"
        values[name] = np.asarray(
            [pairs.get(code, np.nan) for pairs, _quadruples in parsed_composition]
        )
        units[name] = "dimensionless"
        descriptions[name] = (
            f"CompOSE composition pair quantity for particle code {code}"
        )
    quadruple_codes = sorted(
        {code for _pairs, quadruples in parsed_composition for code in quadruples}
    )
    for code in quadruple_codes:
        # eos.compo stores (Aav, Zav, Yav); Nav is calculated as Aav-Zav.
        for component, label in enumerate(("Aav", "Zav", "Yav")):
            name = f"composition_quadruple_{code}_{label}"
            values[name] = np.asarray(
                [
                    quadruples.get(code, (np.nan, np.nan, np.nan))[component]
                    for _pairs, quadruples in parsed_composition
                ]
            )
            units[name] = "dimensionless"
            descriptions[name] = (
                f"CompOSE composition quadruple {label} for particle-set code {code}"
            )
        name = f"composition_quadruple_{code}_Nav"
        values[name] = np.asarray(
            [
                quadruples[code][0] - quadruples[code][1]
                if code in quadruples
                else np.nan
                for _pairs, quadruples in parsed_composition
            ]
        )
        units[name] = "dimensionless"
        descriptions[name] = (
            f"Calculated neutron number Aav-Zav for particle-set code {code}"
        )

    micro_map = parse_microphysics(cold_slice)
    micro_codes = sorted({code for row in rows for code in micro_map.get(row.key, {})})
    for code in micro_codes:
        name = f"micro_{code}"
        values[name] = np.asarray(
            [micro_map.get(row.key, {}).get(code, np.nan) for row in rows]
        )
        units[name] = "source-defined"
        descriptions[name] = f"CompOSE microscopic quantity code {code}"

    if cold_slice.phase_codes is not None:
        values["phase_code"] = np.asarray(
            [np.nan if item is None else float(item) for item in cold_slice.phase_codes]
        )
        units["phase_code"] = "model-specific code"
        descriptions["phase_code"] = (
            "Model-specific CompOSE phase code; not interpreted physically"
        )
    assert all(len(item) == count for item in values.values())
    return values, units, descriptions


def sample_optional_field(
    source: np.ndarray,
    nodes: np.ndarray,
    query: np.ndarray,
    intervals: np.ndarray,
    *,
    piecewise_constant: bool,
) -> np.ndarray:
    """Sample an optional field while keeping unavailable intervals as NaN."""

    if piecewise_constant:
        result = source[intervals].copy()
        result[query == nodes[-1]] = source[-1]
    else:
        left = source[intervals]
        right = source[intervals + 1]
        usable = np.isfinite(left) & np.isfinite(right)
        fraction = (query - nodes[intervals]) / (
            nodes[intervals + 1] - nodes[intervals]
        )
        result = np.where(usable, left + fraction * (right - left), np.nan)
    positions = np.searchsorted(query, nodes)
    valid = positions < len(query)
    exact = np.zeros(len(nodes), dtype=bool)
    exact[valid] = query[positions[valid]] == nodes[valid]
    result[positions[exact]] = source[exact]
    return result


_parse_composition_payload = parse_composition_payload
_parse_microphysics = parse_microphysics
_optional_source_fields = optional_source_fields
_sample_optional_field = sample_optional_field
