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

`model.py` provides `open_eos`, `EosModel`, and the capability report. It only
coordinates existing scientific objects; it does not give different source
types a shared interpolation policy.

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
| `model.py` | High-level loading, capabilities, orchestration, and explicit result bundles |
| `eos.py` | Common continuous-barotrope contract, validation reports, errors, and analytical inputs |
| `tabulated.py` | Ordinary CSV/array barotropes and their fixed interpolation policy |
| `compose_dataset.py` | Lossless CompOSE parsing, hashes, cold-slice selection, and source diagnostics |
| `compose_thermodynamics.py` | Native-Q interpolation and reconstructed thermodynamic quantities |
| `compose.py` | Optional continuous CompOSE barotrope for stellar backgrounds |
| `stellar.py` | Continuous source-boundary TOV stars and sequences |
| `cli.py` | Argument parsing and display; scientific orchestration delegates to `model.py` |
| `__init__.py` | Stable beginner imports plus retained advanced compatibility imports |

The three CompOSE modules are intentionally separate. Combining them would
hide the important case in which source parsing and native thermodynamics
succeed while a continuous pressure-energy-density reduction is unavailable.

## Dependency direction

```text
eos.py
├── tabulated.py
├── compose_dataset.py
│   ├── compose_thermodynamics.py
│   └── compose.py
└── stellar.py

model.py -> all workflow layers
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
