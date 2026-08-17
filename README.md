# Neutron-star EoS toolkit

A small Python toolkit for inspecting cold one-fluid equations of state and,
when the input supports it, calculating continuous source-boundary stellar
backgrounds.

It accepts three deliberately separate input types:

- analytical pressure and sound-speed functions;
- ordinary CSV tables containing total energy density and pressure;
- cold one-dimensional CompOSE directories or ZIP archives.

The package never silently sorts, clips, smooths, repairs, extrapolates, or
splices an input. A CompOSE source can provide useful native thermodynamics
even when it cannot be reduced to one continuous stellar barotrope.

## Install

Python 3.12 is the verified runtime.

The shortest development setup installs the package, plots, and notebook
kernel together:

```powershell
uv sync --all-extras
```

If you do not use `uv`, a conventional virtual environment also works:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
```

On Linux or macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
```

## Interactive experiments

Register the project environment as a named Jupyter kernel, then open the
guided experiment:

```powershell
uv run python -m ipykernel install --user --name neutron-star-eos-toolkit `
  --display-name "Python (neutron-star-eos-toolkit)"
uv run jupyter lab .\notebooks\eos_experiments.ipynb
```

In VS Code, select **Python (neutron-star-eos-toolkit)** in the notebook's
kernel picker. If `import neutron_star_eos` fails, the notebook is using a
different Python environment.

The notebook runs with the supplied CSV by default. It can also load a user
CSV, a CompOSE source, or the editable analytical definition in
[`notebooks/analytical_eos.py`](notebooks/analytical_eos.py). That file holds
the authoritative user function `P(epsilon)` and its consistent
`dP/d(epsilon)`; the notebook handles inspection, plots, stellar calculations,
and provenance.

For a pinned, reproducible research example, the
[cold CompOSE comparison campaign](experiments/compose_comparison/README.md)
downloads and independently TOV-integrates eight core EoSs plus one hybrid
stress test. It writes every plot to a separate PNG and cross-checks calculated
curves against optional `eos.mr` references, current CompOSE catalogue values,
and convention-aware primary literature.

## Beginner workflows

The [`workflows/`](workflows) directory contains six short, editable scripts.
Each has one obvious parameter block and reads from top to bottom:

```text
analytical_eos.py               define P(epsilon)
csv_eos.py                      load a CSV table
compose_eos.py                  load one downloaded CompOSE archive
calculate_one_star.py           integrate one TOV model
calculate_mass_radius.py        calculate and plot a sequence
compare_equations_of_state.py   compare two calculated sequences
```

For example:

```powershell
uv run python workflows\calculate_one_star.py
```

Use these scripts when you want an editable calculation without notebook or
command-line abstraction. Use the notebook when you want explanations and
plots beside each step.

## First command

Inspect the supplied CSV without running a stellar solver or writing files:

```powershell
.\.venv\Scripts\eos-tool.exe inspect .\examples\tabulated.csv --kind csv
```

On Linux or macOS, use:

```bash
.venv/bin/eos-tool inspect examples/tabulated.csv --kind csv
```

The report states independently whether the source, thermodynamics,
continuous barotrope, stellar background, composition, and tidal operations
are available.

## Python API

The beginner workflow has two entry points: `open_eos` for files and
`EosModel.from_analytical` for functions.

### Ordinary CSV

```python
from neutron_star_eos import open_eos

model = open_eos("my_eos.csv", kind="csv")
print(model.summary())

star = model.solve_star(central_pressure_mev_fm3=100.0)
print(star.mass_msun, star.radius_km, star.boundary_status)
```

CSV columns use total energy density and pressure in MeV/fm^3, ordered from
low to high:

```text
epsilon_mev_fm3,pressure_mev_fm3
1.0,0.001
10.0,0.1
100.0,10.0
400.0,160.0
```

### Analytical expression

```python
import numpy as np
from neutron_star_eos import EosModel


def pressure(energy_density):
    epsilon = np.asarray(energy_density)
    return 1.0e-3 * epsilon**2


def sound_speed_squared(energy_density):
    epsilon = np.asarray(energy_density)
    return 2.0e-3 * epsilon


model = EosModel.from_analytical(
    name="example-analytical-eos",
    pressure_from_energy_density=pressure,
    sound_speed_squared_from_energy_density=sound_speed_squared,
    energy_density_domain_mev_fm3=(1.0, 400.0),
    source="educational example",
)
print(model.summary())
```

The quadratic is only a runnable example. The interface accepts the complete
analytical function `P(epsilon)` you provide; it does not require `K`, `gamma`,
or any particular parameterization.

The report fingerprints evaluated pressure and sound-speed values on a
declared 2049-point grid and the recovered energy density on 129 corresponding
pressure points. That identifies the assessed behavior of both the forward and
inverse callables, not their source code, so keep the defining analytical
script under version control alongside any result bundle.

### CompOSE source

```python
from neutron_star_eos import open_eos

model = open_eos(
    "path/to/eos.zip",
    kind="compose",
    model_id="catalogue model",
    source_url="https://compose.obspm.fr/eos/...",
    includes_leptons=True,
)
print(model.summary())
profile = model.native_thermodynamics
```

CompOSE parsing, native-Q thermodynamics, and the optional continuous stellar
barotrope are separate layers. A pressure reversal can make
`continuous_barotrope` unavailable while `thermodynamics` remains available
with diagnostics. See [the CompOSE guide](docs/compose.md) for the native
quantities, density selection, and diagnostic ordering policies.

`ComposeEos.pressure_from_baryon_density(...)` and
`ComposeEos.baryon_density_from_pressure(...)` expose the declared native
density coordinate without extrapolation, which is useful for controlled
central-density experiments.

## Plotting

Plotting is optional and never runs a solver, repairs data, or writes files:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plot]"
```

```python
import matplotlib.pyplot as plt
from neutron_star_eos.plotting import plot_pressure_energy

fig, ax = plt.subplots(constrained_layout=True)
plot_pressure_energy(model, ax=ax)
```

Raw source nodes, native CompOSE thermodynamics, and evaluated continuous
barotropes remain visually distinct. See the [plotting guide](docs/plotting.md).

## Command line

```text
eos-tool inspect  PATH --kind csv|compose
eos-tool star     PATH --kind csv|compose --central-pressure-mev-fm3 P
eos-tool sequence PATH --kind csv|compose [--points N]
```

CompOSE commands additionally require `--model-id`, `--source-url`, and the
explicit `--includes-leptons` declaration when applicable.

Commands print results by default and write nothing. Add `--output NEW_DIR`
to create a new deterministic bundle:

```text
NEW_DIR/
├── summary.txt
├── report.json
├── thermodynamics.csv     inspect only, when thermodynamics are available
├── star.json              one-star calculation
└── sequence.json/.csv     sequence calculation
```

An existing output directory is never overwritten. Sequence tables retain
every requested central pressure, including unavailable attempts and reasons.
The default sequence uses 50 geometric central pressures across the declared
pressure domain; an explicit `--points` value must be at least 9.

Reports record the toolkit, Python, NumPy, and SciPy versions. Stellar JSON
also records the exact ODE settings, validation mode, EoS provenance identity,
source-boundary status, physical conversion constants and their authority, and
any retained radial profile.

The original `eos-tool validate csv|compose` syntax remains as a compatibility
alias for existing scripts.

## Capability meanings

- `available`: the operation completed under its declared contract.
- `available_with_diagnostics`: results are available and the named findings
  remain visible.
- `unavailable`: the operation was not authorized by the available evidence.
- `not_applicable`: the input type does not define that quantity or operation.

Availability is deliberately narrower than a claim of complete physical
validity, numerical convergence, branch stability, or observational agreement.

## Scientific boundary

The current stellar solver handles continuous background models only. It
stops at the input's lowest selected positive pressure, so reported radii are
source-boundary radii rather than silently claimed vacuum surfaces.

Tidal observables, physical density jumps, maximum-mass claims,
finite-temperature stellar reductions, automatic crust splicing, and
two-fluid dark-matter stars are outside this release and remain unavailable.

## Repository map

```text
src/neutron_star_eos/  public package
examples/              one runnable example per input type
workflows/             short task-oriented calculations for beginners
notebooks/             guided experiments and editable analytical definition
experiments/           pinned, reproducible research campaigns
docs/                  concise architecture and CompOSE guidance
tests/                 synthetic interface and solver tests
```

See [the architecture map](docs/architecture.md) for the responsibility of
every source file. Downloaded CompOSE archives and generated campaign data and
figures remain local and ignored; the tracked campaign registry contains only
source URLs, hashes, benchmark metadata, and reproducible code.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The [parameter reference](docs/parameters.md),
[diagnostic reference](docs/diagnostics.md), and
[reproducibility guide](docs/reproducibility.md) describe the public contracts
used by the CLI and notebook. See [CONTRIBUTING.md](CONTRIBUTING.md) before
changing scientific behavior.

This project is distributed under the [MIT License](LICENSE).
