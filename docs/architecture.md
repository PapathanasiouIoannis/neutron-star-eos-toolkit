# Architecture

The package has one beginner-facing workflow and three source-specific input
paths. Source-specific physics stays separate until an input can legitimately
provide the common continuous-barotrope contract.

```text
Analytical functions -> analytical assessment --\
Ordinary CSV table  -> tabulated assessment -----+-> continuous barotrope -> stars
CompOSE source      -> native-Q thermodynamics --/
                                  |
                                  +-> may remain useful without a barotrope
```

## Public workflow

`model.py` provides `open_eos`, `EosModel`, the capability report, and a
read-only thermodynamic presentation view. It coordinates existing scientific
objects; it does not give different source types a shared interpolation policy.

```python
model = open_eos(path, kind="csv")
print(model.summary())
barotrope = model.require_barotrope()
star = model.solve_star(central_pressure_mev_fm3=100.0)
```

Advanced users can still import the lower-level adapters directly from their
modules.

## Source-file responsibilities

| File | Responsibility |
|---|---|
| `model.py` | High-level loading, capabilities, orchestration, and thermodynamic views |
| `output.py` | Deterministic, non-overwriting inspection and stellar bundles |
| `thermodynamics.py` | Read-only source/native/barotrope presentation contracts |
| `eos.py` | Common continuous-barotrope contract, validation reports, and errors |
| `analytical.py` | Focused analytical-adapter import and compatibility boundary |
| `tabulated.py` | Ordinary CSV/array barotropes and their fixed interpolation policy |
| `compose/dataset.py` | Lossless CompOSE parsing, hashes, cold-slice selection, and source diagnostics |
| `compose/thermodynamics.py` | Native-Q interpolation and reconstructed thermodynamic quantities |
| `compose/barotrope.py` | Optional continuous CompOSE barotrope for stellar backgrounds |
| `plotting.py` | Optional presentation of already-loaded data and computed results |
| `stellar.py` | Continuous source-boundary TOV stars and sequences |
| `cli.py` | Argument parsing and display; scientific orchestration delegates to `model.py` |
| `__init__.py` | Stable beginner imports plus retained advanced compatibility imports |

The three CompOSE layers form one subpackage but remain intentionally separate. Combining them would
hide the important case in which source parsing and native thermodynamics
succeed while a continuous pressure-energy-density reduction is unavailable.

## Dependency direction

```text
eos.py
├── analytical.py
├── tabulated.py
├── compose/dataset.py
│   ├── compose/thermodynamics.py
│   └── compose/barotrope.py
└── stellar.py

model.py -> all workflow layers
output.py -> model/result presentation contracts
plotting.py -> thermodynamic views and existing results
cli.py   -> model.py
```

Scientific modules never import the facade or CLI. This keeps the low-level
calculations independently testable and prevents circular dependencies.

## Invariants

- Units and energy-density conventions are explicit.
- No input is silently sorted, clipped, repaired, smoothed, or extrapolated.
- CompOSE native thermodynamics does not require a stellar barotrope.
- A numerical seam is not automatically labelled a physical transition.
- Every sequence request retains a success or an explicit failure reason.
- A positive-pressure source boundary is not called a vacuum surface.
- Unavailable tides, stability, turning points, and maximum masses are never
  inferred from a background sequence.
- Public tests use synthetic fixtures; private external validation data does
  not enter this repository.
