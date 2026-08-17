# Contributing

Thank you for improving the neutron-star EoS toolkit. Contributions are
coordinated with the repository owner and are distributed under the project's
MIT License.

## Development environment

The verified runtime is CPython 3.12 with NumPy 1.26.4 and SciPy 1.13.1. Create
an isolated environment from the repository root and apply the checked-in
constraints:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints/verified-py312.txt -e .
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -c constraints/verified-py312.txt -e .
```

The constraints file records the verified environment; it does not broaden
the supported Python range or replace the package metadata.

## Verification

Run the complete unit-test suite and the examples that require no external
data:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe examples/analytical.py
.\.venv\Scripts\python.exe examples/tabulated.py
```

Use the corresponding `.venv/bin/python` path on Linux or macOS. CompOSE tests
must use synthetic fixtures or caller-supplied data; do not commit downloaded
catalogue tables.

Before requesting review:

1. Run the full tests from a clean checkout or isolated environment.
2. Add focused tests for successful behavior and failure paths.
3. Update public documentation and the changelog when behavior changes.
4. Keep notebook cells executable from top to bottom and avoid committing
   bulky generated output.
5. Confirm that examples and plots use the public API rather than duplicating
   scientific calculations.

## Scientific invariants

Changes must preserve the toolkit's fail-closed behavior:

- State units and the total-energy-density convention explicitly.
- Never silently sort, clip, smooth, repair, extrapolate, or splice input data.
- Preserve source ordering, provenance, diagnostics, missing values, and every
  requested sequence attempt.
- Keep native CompOSE thermodynamics separate from the optional continuous
  stellar reduction.
- Do not describe a positive-pressure source boundary as a vacuum surface.
- Do not infer tides, stability, turning points, or a maximum mass from the
  current background solver.
- Treat diagnostic reductions as sensitivity tools, not physical phase-
  transition prescriptions.

Any proposed change to these invariants needs an explicit scientific rationale
and owner review.

## Code and repository hygiene

- Use UTF-8 text and the line endings defined in `.gitattributes` and
  `.editorconfig`.
- Keep public functions typed and documented with parameter units, valid
  domains, and failure behavior.
- Prefer small, responsibility-focused modules and preserve compatibility when
  moving an established import.
- Keep plotting, notebooks, and other optional interfaces out of core runtime
  dependencies.
- Do not commit virtual environments, build artifacts, caches, credentials, or
  private research data.
- Do not mix unrelated changes into the same review.

## Review process

Work on a focused branch, summarize the user-visible and scientific impact,
and open a pull request against `main`. Include the verification commands and
results in the pull-request description. A passing test run is necessary but
does not replace scientific review.
