"""Passive command-line validation for reusable cold-EoS inputs."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from neutron_star_eos import EosInputError, load_compose_eos, load_csv_eos


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eos-tool",
        description="Validate a canonical-unit CSV or a cold 1D CompOSE EoS.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser(
        "validate", help="read and validate without running a stellar solver or writing files"
    )
    sources = validate.add_subparsers(dest="input_type", required=True)

    csv_parser = sources.add_parser("csv", help="ordinary canonical-unit CSV")
    csv_parser.add_argument("path")
    csv_parser.add_argument("--name")
    csv_parser.add_argument("--source")
    csv_parser.add_argument("--epsilon-column", default="epsilon_mev_fm3")
    csv_parser.add_argument("--pressure-column", default="pressure_mev_fm3")
    csv_parser.add_argument("--baryon-density-column")
    csv_parser.add_argument("--format", choices=("text", "json"), default="text")

    compose = sources.add_parser("compose", help="cold one-dimensional CompOSE input")
    compose.add_argument("path")
    compose.add_argument("--model-id", required=True)
    compose.add_argument("--source-url", required=True)
    compose.add_argument(
        "--includes-leptons",
        action="store_true",
        help="explicitly declare that the selected catalogue table includes leptons",
    )
    compose.add_argument(
        "--baryon-density-max-fm3",
        type=float,
        help="optional explicit retained upper density; never inferred automatically",
    )
    compose.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _load(arguments: argparse.Namespace):
    if arguments.input_type == "csv":
        return load_csv_eos(
            arguments.path,
            name=arguments.name,
            source=arguments.source,
            epsilon_column=arguments.epsilon_column,
            pressure_column=arguments.pressure_column,
            baryon_density_column=arguments.baryon_density_column,
        )
    return load_compose_eos(
        arguments.path,
        model_id=arguments.model_id,
        source_url=arguments.source_url,
        matter="cold_beta_equilibrated",
        includes_leptons=arguments.includes_leptons,
        baryon_density_max_fm3=arguments.baryon_density_max_fm3,
    )


def _text(eos, report) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"EoS validation: {status}",
        f"Model: {eos.model_name}",
        (
            "Rows/domain samples: "
            f"{len(getattr(eos, 'pressure_mev_fm3', [])) or 'analytical'} / "
            f"{report.assessed_points}"
        ),
        (
            "Pressure domain: "
            f"[{eos.pressure_min_mev_fm3:.12g}, {eos.pressure_max_mev_fm3:.12g}] MeV/fm^3"
        ),
        (
            "Energy-density domain: "
            f"[{eos.energy_density_min_mev_fm3:.12g}, "
            f"{eos.energy_density_max_mev_fm3:.12g}] MeV/fm^3"
        ),
        f"Sound-speed-squared range: [{report.cs2_min:.12g}, {report.cs2_max:.12g}]",
        (
            "Stellar surface policy: finite source boundary at "
            f"P={eos.pressure_min_mev_fm3:.12g} MeV/fm^3; this is not claimed to be P=0"
        ),
        "Tidal capability: unavailable at this positive-pressure source boundary",
        "Extrapolation: forbidden",
    ]
    if report.issues:
        lines.append("Issues:")
        lines.extend(f"- {item.code}: {item.message}" for item in report.issues)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        eos = _load(arguments)
        report = eos.validate()
    except (EosInputError, OSError, ValueError) as exc:
        if arguments.format == "json":
            print(json.dumps({"status": "fail", "error": str(exc)}, indent=2))
        else:
            print(f"EoS validation: FAIL\n{exc}")
        return 2
    if arguments.format == "json":
        print(
            json.dumps(
                {"validation": report.to_dict(), "provenance": eos.provenance()},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_text(eos, report))
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
