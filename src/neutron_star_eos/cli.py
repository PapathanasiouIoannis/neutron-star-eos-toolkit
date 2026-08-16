"""Small command-line front end for reusable cold-EoS inputs."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from neutron_star_eos._version import __version__
from neutron_star_eos.compose import COMPOSE_ORDERING_POLICIES
from neutron_star_eos.eos import EosDomainError, EosInputError
from neutron_star_eos.model import EosModel, open_eos
from neutron_star_eos.stellar import STELLAR_VALIDATION_MODES
from neutron_star_eos.tabulated import load_csv_eos


def _add_open_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path")
    parser.add_argument("--kind", choices=("csv", "compose"), required=True)
    parser.add_argument("--name")
    parser.add_argument("--source", dest="source_description")
    parser.add_argument("--epsilon-column", default="epsilon_mev_fm3")
    parser.add_argument("--pressure-column", default="pressure_mev_fm3")
    parser.add_argument("--baryon-density-column")
    parser.add_argument("--model-id")
    parser.add_argument("--source-url")
    parser.add_argument(
        "--includes-leptons",
        action="store_true",
        help="declare that a CompOSE source includes leptons",
    )
    parser.add_argument("--baryon-density-min-fm3", type=float)
    parser.add_argument("--baryon-density-max-fm3", type=float)
    parser.add_argument(
        "--ordering-policy",
        choices=COMPOSE_ORDERING_POLICIES,
        default="strict",
    )
    parser.add_argument("--native-points", type=int, default=2001)


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--output",
        help="write a new result directory; an existing target is never overwritten",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eos-tool",
        description="Inspect an EoS or calculate continuous stellar backgrounds.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect = subcommands.add_parser(
        "inspect",
        help="report source, thermodynamic, and stellar capabilities without solving stars",
    )
    _add_open_arguments(inspect)
    _add_output_arguments(inspect)
    inspect.add_argument(
        "--require-barotrope",
        action="store_true",
        help="fail unless the optional continuous stellar barotrope is available",
    )

    star = subcommands.add_parser(
        "star", help="solve one source-boundary stellar background"
    )
    _add_open_arguments(star)
    _add_output_arguments(star)
    star.add_argument("--central-pressure-mev-fm3", type=float, required=True)
    star.add_argument(
        "--validation-mode",
        choices=STELLAR_VALIDATION_MODES,
        default="strict",
    )

    sequence = subcommands.add_parser(
        "sequence", help="sample a source-boundary stellar sequence"
    )
    _add_open_arguments(sequence)
    _add_output_arguments(sequence)
    sequence.add_argument("--points", type=int, default=50)
    sequence.add_argument(
        "--validation-mode",
        choices=STELLAR_VALIDATION_MODES,
        default="strict",
    )

    validate = subcommands.add_parser(
        "validate",
        help="compatibility alias for the original passive CSV/CompOSE validation commands",
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
        "--baryon-density-min-fm3",
        type=float,
        help="optional explicit first retained source density",
    )
    compose.add_argument(
        "--baryon-density-max-fm3",
        type=float,
        help="optional explicit retained upper density; never inferred automatically",
    )
    compose.add_argument(
        "--ordering-policy",
        choices=COMPOSE_ORDERING_POLICIES,
        default="strict",
        help=(
            "strict, or an explicit diagnostic monotone source-row subsequence; "
            "the diagnostic policy records omissions and is not a transition model"
        ),
    )
    compose.add_argument(
        "--native-points",
        type=int,
        default=2001,
        help=(
            "base geometric sample count for the native-Q assessment; exact source "
            "nodes are then unioned into the reported query grid"
        ),
    )
    compose.add_argument(
        "--require-barotrope",
        action="store_true",
        help=(
            "return a failing exit status unless the optional continuous stellar "
            "barotrope is also available and passes its physics gate"
        ),
    )
    compose.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _load_csv(arguments: argparse.Namespace):
    return load_csv_eos(
        arguments.path,
        name=arguments.name,
        source=arguments.source,
        epsilon_column=arguments.epsilon_column,
        pressure_column=arguments.pressure_column,
        baryon_density_column=arguments.baryon_density_column,
    )


def _open_model(arguments: argparse.Namespace) -> EosModel:
    return open_eos(
        arguments.path,
        kind=arguments.kind,
        name=arguments.name,
        source_description=arguments.source_description,
        epsilon_column=arguments.epsilon_column,
        pressure_column=arguments.pressure_column,
        baryon_density_column=arguments.baryon_density_column,
        model_id=arguments.model_id,
        source_url=arguments.source_url,
        includes_leptons=arguments.includes_leptons,
        baryon_density_min_fm3=arguments.baryon_density_min_fm3,
        baryon_density_max_fm3=arguments.baryon_density_max_fm3,
        native_points=arguments.native_points,
        ordering_policy=arguments.ordering_policy,
    )


def _inspection_exit_code(model: EosModel, *, require_barotrope: bool) -> int:
    report = model.report()
    if not report.capability("thermodynamics").available:
        return 2
    if require_barotrope and not report.capability("continuous_barotrope").available:
        return 2
    if (
        model.kind != "compose"
        and not report.capability("continuous_barotrope").available
    ):
        return 2
    return 0


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


def _compose_text(payload: dict[str, object]) -> str:
    dataset = payload["dataset"]
    cold_slice = payload.get("cold_slice")
    barotrope = payload.get("barotrope")
    assert isinstance(dataset, dict)
    lines = [
        f"CompOSE dataset: {str(dataset['status']).upper()}",
        f"Model: {dataset.get('model_id', 'unavailable')}",
    ]
    if dataset.get("reason"):
        lines.append(f"- {dataset['reason']}")
    if isinstance(cold_slice, dict):
        lines.append(f"Cold-slice status: {cold_slice['status']}")
        if cold_slice.get("reason"):
            lines.append(f"- {cold_slice['reason']}")
        for item in cold_slice.get("diagnostics", []):
            lines.append(f"- {item['severity']} {item['code']}: {item['message']}")
    density_selection = payload.get("density_selection")
    if isinstance(density_selection, dict):
        lines.append(
            "Selected native density path: "
            f"{density_selection['source_rows']} source rows over "
            f"[{density_selection['baryon_density_min_fm3']:.12g}, "
            f"{density_selection['baryon_density_max_fm3']:.12g}] fm^-3"
        )
        requested_minimum = density_selection.get("requested_baryon_density_min_fm3")
        requested_maximum = density_selection.get("requested_baryon_density_max_fm3")
        if requested_minimum is not None or requested_maximum is not None:
            lines.append(
                "- requested bounds: "
                f"min={requested_minimum if requested_minimum is not None else 'source minimum'}, "
                f"max={requested_maximum if requested_maximum is not None else 'source maximum'} "
                "fm^-3"
            )
    native = payload.get("native_thermodynamics")
    if isinstance(native, dict):
        lines.append(f"Native-Q thermodynamics: {native['status']}")
        if native.get("reason"):
            lines.append(f"- {native['reason']}")
        elif "profile_points" in native:
            lines.append(
                f"- {native['profile_points']} evaluated points; "
                f"{len(native['columns'])} preserved or reconstructed columns"
            )
        for item in native.get("diagnostics", []):
            lines.append(f"- {item['severity']} {item['code']}: {item['message']}")
    if isinstance(barotrope, dict):
        lines.append(f"Continuous barotrope: {barotrope['status']}")
        if barotrope.get("reason"):
            lines.append(f"- {barotrope['reason']}")
        validation = barotrope.get("validation")
        provenance = barotrope.get("provenance")
        if isinstance(provenance, dict):
            selection = provenance.get("selection")
            if isinstance(selection, dict):
                lines.append(
                    f"Ordering policy: {selection.get('ordering_policy', 'strict')}"
                )
                omitted = int(selection.get("omitted_source_rows", 0))
                if omitted:
                    lines.append(
                        f"- DIAGNOSTIC: {omitted} source row(s) omitted; "
                        "not a physical transition treatment"
                    )
        if isinstance(validation, dict):
            lines.append(f"Interpolant validation: {validation['status'].upper()}")
            for item in validation.get("issues", []):
                lines.append(f"- {item['code']}: {item['message']}")
    return "\n".join(lines)


def _assess_compose(arguments: argparse.Namespace) -> tuple[dict[str, object], int]:
    model = open_eos(
        arguments.path,
        kind="compose",
        model_id=arguments.model_id,
        source_url=arguments.source_url,
        includes_leptons=arguments.includes_leptons,
        baryon_density_min_fm3=arguments.baryon_density_min_fm3,
        baryon_density_max_fm3=arguments.baryon_density_max_fm3,
        native_points=arguments.native_points,
        ordering_policy=arguments.ordering_policy,
    )
    payload: dict[str, object] = {
        "input_type": "compose",
        **json.loads(json.dumps(model.report().details, sort_keys=True)),
    }
    barotrope = payload.setdefault(
        "barotrope",
        {"status": "unavailable", "reason": "barotrope was not assessed"},
    )
    assert isinstance(barotrope, dict)
    barotrope["required"] = bool(arguments.require_barotrope)
    return payload, _inspection_exit_code(
        model,
        require_barotrope=bool(arguments.require_barotrope),
    )


def _failure_reason_code(exc: BaseException) -> str:
    """Classify command failures without making callers parse error messages."""

    supplied = getattr(exc, "reason_code", None)
    if isinstance(supplied, str) and supplied:
        return supplied
    if isinstance(exc, EosDomainError):
        return "eos_domain_error"
    if isinstance(exc, EosInputError):
        return "eos_input_error"
    if isinstance(exc, OSError):
        return "io_error"
    if isinstance(exc, ArithmeticError):
        return "arithmetic_error"
    if isinstance(exc, ValueError):
        return "value_error"
    return "runtime_error"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command in {"inspect", "star", "sequence"}:
        try:
            model = _open_model(arguments)
            output_directory = None
            if arguments.command == "inspect":
                payload: dict[str, object] = model.report().to_dict()
                if arguments.output:
                    output_directory = model.write_inspection(arguments.output)
                exit_code = _inspection_exit_code(
                    model,
                    require_barotrope=bool(arguments.require_barotrope),
                )
                text_output = model.summary()
            elif arguments.command == "star":
                star_result = model.solve_star(
                    arguments.central_pressure_mev_fm3,
                    validation_mode=arguments.validation_mode,
                )
                payload = {
                    "model": model.report().to_dict(),
                    "star": star_result.to_dict(),
                }
                if arguments.output:
                    output_directory = model.write_star(arguments.output, star_result)
                exit_code = 0
                text_output = "\n".join(
                    (
                        f"Model: {model.model_name}",
                        f"Mass: {star_result.mass_msun:.12g} Msun",
                        f"Radius: {star_result.radius_km:.12g} km",
                        f"Boundary: {star_result.boundary_status}",
                    )
                )
            else:
                sequence_result = model.solve_sequence(
                    points=arguments.points,
                    validation_mode=arguments.validation_mode,
                )
                payload = {
                    "model": model.report().to_dict(),
                    "sequence": sequence_result.to_dict(),
                }
                if arguments.output:
                    output_directory = model.write_sequence(
                        arguments.output, sequence_result
                    )
                exit_code = 0 if sequence_result.status == "complete" else 1
                text_output = "\n".join(
                    (
                        f"Model: {model.model_name}",
                        f"Sequence: {sequence_result.status}",
                        "Solved: "
                        f"{len(sequence_result.stars)}/{len(sequence_result.attempts)}",
                        f"Boundary: {sequence_result.boundary_status}",
                    )
                )
            if output_directory is not None:
                payload["output_directory"] = str(output_directory)
                text_output += f"\nOutput: {output_directory}"
        except (
            EosInputError,
            OSError,
            ValueError,
            RuntimeError,
            ArithmeticError,
        ) as exc:
            payload = {
                "status": "fail",
                "reason_code": _failure_reason_code(exc),
                "error": str(exc),
            }
            text_output = f"{arguments.command.capitalize()}: FAIL\n{exc}"
            exit_code = 2
        if arguments.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(text_output)
        return exit_code
    if arguments.input_type == "compose":
        try:
            payload, exit_code = _assess_compose(arguments)
        except (EosInputError, OSError, ValueError) as exc:
            payload = {
                "input_type": "compose",
                "dataset": {"status": "fail", "reason": str(exc)},
            }
            exit_code = 2
        if arguments.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_compose_text(payload))
        return exit_code
    try:
        eos = _load_csv(arguments)
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
