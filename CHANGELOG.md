# Changelog

Notable changes to the neutron-star EoS toolkit are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and released
versions will follow [Semantic Versioning](https://semver.org/).

Version `0.2.0` is the active development version; it has not been published or
tagged as a supported release.

## [Unreleased]

### Added

- A unified `open_eos` and `EosModel` workflow for analytical functions, CSV
  tables, and cold one-dimensional CompOSE sources.
- Capability reports that distinguish parsed source data, native
  thermodynamics, continuous stellar barotropes, stellar backgrounds,
  composition, and unavailable tidal operations.
- `inspect`, `star`, and `sequence` command-line workflows with explicit,
  non-overwriting result bundles.
- Native CompOSE thermodynamic reconstruction that remains available when a
  continuous stellar barotrope cannot be constructed.
- Source-boundary stellar results that retain failed sequence attempts and
  their reasons.
- A read-only thermodynamic view that distinguishes source nodes, native
  CompOSE quantities, and the continuous stellar barotrope.
- Optional, source-aware scientific plotting functions and a packaged
  Matplotlib style.
- A guided, headlessly tested experiment notebook plus an editable analytical
  `P(epsilon)` definition with source hashing and an opt-in result manifest.
- A pinned nine-model cold CompOSE campaign that independently calculates TOV
  sequences, writes one plot per PNG, and cross-checks catalogue, optional
  `eos.mr`, and convention-classified literature benchmarks.
- A provenance-preserving reader for optional CompOSE `eos.mr` tables and
  no-extrapolation mappings between native baryon density and pressure.
- Stable command-line and sequence-attempt reason codes.
- Architecture, CompOSE, and runnable input examples for the current public
  interface.
- Contribution, security, diagnostics, parameter, plotting, and
  reproducibility guidance, together with reproducible Python 3.12 constraints.

### Changed

- CompOSE parsing, native thermodynamics, and optional stellar-barotrope
  construction now live in focused submodules while compatibility imports
  remain available.
- Reports now identify the toolkit, Python, NumPy, and SciPy versions used for
  a calculation.
- Result serialization now validates both model provenance and result type,
  and never overwrites an existing output directory.
- The TOV unit conversions now retain the precision implied by the CompOSE
  v3.01 constants table, and every stellar result records those constants and
  their authority for reproducibility.

### Release status

- The verified runtime remains CPython 3.12 with NumPy 1.26.4 and SciPy 1.13.1.
- The source is distributed under the MIT License. No public support promise,
  package-index publication, or archived release is implied by this
  development version.
