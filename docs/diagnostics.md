# Diagnostics and availability

Capability status and physical validity are deliberately separate.

- `available`: the operation completed under its declared contract.
- `available_with_diagnostics`: results remain accessible and named findings
  remain visible.
- `unavailable`: the evidence does not authorize the operation.
- `not_applicable`: the input type does not define the quantity.

Every machine-readable report contains stable diagnostic codes alongside human
messages. Code should branch on capability status or diagnostic code, never on
fragments of an English message.

## Continuous-barotrope findings

Common findings include nonfinite or nonpositive values, nonmonotone pressure,
mechanical instability, acausality, an inconsistent supplied analytical
derivative, or an inconsistent supplied analytical inverse. Strict stellar
calculations require the continuous pressure-energy-density contract to pass.

## CompOSE findings

CompOSE assessment is layered:

1. lossless parsing and cold-slice selection;
2. native-Q thermodynamic reconstruction;
3. optional continuous stellar-barotrope construction.

A finding at layer three does not erase useful results from layers one or two.
Native diagnostics cover source ordering, duplicated indices, cold-condition
residuals, thermodynamic closure, derivative signs, causality, missing optional
fields, and partial composition coverage. See [compose.md](compose.md) for the
quantity definitions and interpolation policies.

Diagnostic tolerances are assessment thresholds. Passing one is not a claim of
observational agreement, global thermodynamic consistency, stable stellar
branches, or numerical convergence.

## Stellar attempt reason codes

Sequence attempts retain both a stable `reason_code` and a detailed message.
`radius_limit_reached` means the configured integration ceiling was reached
before the lowest supplied positive pressure. Other numerical or EoS failures
use a stable general code while preserving the exact message for audit.

Command-line failures always include a non-null `reason_code`. Solver-specific
codes take precedence; other failures use `eos_domain_error`,
`eos_input_error`, `io_error`, `arithmetic_error`, `value_error`, or
`runtime_error` according to the exception boundary.

Plots and tables must retain unsuccessful attempts. They must not connect
across a missing result or reinterpret an unavailable attempt as zero.
