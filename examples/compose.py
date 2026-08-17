"""Inspect a user-supplied cold one-dimensional CompOSE source."""

from __future__ import annotations

import argparse

from neutron_star_eos import open_eos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CompOSE directory or eos.zip")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument(
        "--includes-leptons",
        action="store_true",
        help="explicitly declare that this catalogue source includes leptons",
    )
    arguments = parser.parse_args()

    model = open_eos(
        arguments.path,
        kind="compose",
        model_id=arguments.model_id,
        source_url=arguments.source_url,
        includes_leptons=arguments.includes_leptons,
    )
    print(model.summary())

    if model.report().capability("continuous_barotrope").available:
        print("A continuous source-boundary stellar calculation is available.")
    else:
        print("The native thermodynamics remain usable without a stellar reduction.")


if __name__ == "__main__":
    main()
