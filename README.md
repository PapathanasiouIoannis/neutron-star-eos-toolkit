# Neutron-star EoS toolkit

A small Python toolkit for continuous, cold, one-fluid equations of state.
It accepts:

- analytical functions;
- ordinary CSV tables;
- cold one-dimensional CompOSE directories or ZIP files.

Every input declares a finite domain and is checked without sorting, clipping,
repairing, extrapolating, or inventing missing physics.

## Install

Python 3.12 is currently the verified runtime.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/eos-tool validate csv examples/tabulated.csv
```

## First two commands

Validate the supplied CSV without running a stellar calculation:

```powershell
.\.venv\Scripts\eos-tool.exe validate csv .\examples\tabulated.csv
```

Run the toy analytical example end to end (it is an interface demonstration,
not a realistic neutron-star EoS):

```powershell
.\.venv\Scripts\python.exe .\examples\analytical.py
```

## Use your own CSV

Supply total energy density and pressure in MeV/fm^3, ordered from low to high:

```text
epsilon_mev_fm3,pressure_mev_fm3
1.0,0.001
10.0,0.1
100.0,10.0
400.0,160.0
```

```python
from neutron_star_eos import load_csv_eos, solve_star

eos = load_csv_eos("my_eos.csv")
eos.validate().require_pass()

star = solve_star(eos, central_pressure_mev_fm3=100.0)
print(star.mass_msun, star.radius_km)
```

## Use a CompOSE table

Version 1 accepts only continuous, cold, beta-equilibrated, one-dimensional
stellar-matter tables whose CompOSE metadata confirm that leptons are present.

```powershell
.\.venv\Scripts\eos-tool.exe validate compose path\to\eos.zip `
  --model-id "catalogue model" `
  --source-url "https://compose.obspm.fr/eos/..." `
  --includes-leptons
```

See [the CompOSE guide](docs/compose.md) for the exact supported format and
the optional explicit upper-density selection.

## Scientific boundary

The background solver stops at the EoS source's lowest positive pressure.
Consequently, the reported mass and radius are **source-boundary values**, not
silently vacuum `P=0` observables. Tidal observables, density jumps, maximum-mass claims,
finite-temperature tables, automatic crust splicing, and two-fluid dark-matter
stars are outside version 1 and fail closed rather than being approximated.

## Repository map

```text
src/neutron_star_eos/  public library
examples/              one analytical example and one CSV
docs/                  short format-specific guidance
tests/                 compact interface and background-solver tests
```

This clean toolkit deliberately contains no research campaigns, manuscripts,
strict-run packets, publication figures, or private validation archive.

## Development

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This is currently a private pre-release. A software license must be selected
before public redistribution.
