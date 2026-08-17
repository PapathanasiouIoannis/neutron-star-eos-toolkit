# Physics-first architecture

The repository is organized by scientific task. A physics student should be
able to find an equation, source-selection rule, interpolation, or plot by its
physical name rather than by knowing the software design first.

```text
source data -> source-specific validation -> continuous cold barotrope
                                                   |
                                                   v
                                   TOV equations -> one star -> sequence
```

CompOSE native thermodynamics can remain useful even when the source cannot be
reduced to one continuous stellar barotrope.

## Start here

- `workflows/`: short scripts with one visible parameter section.
- `notebooks/eos_experiments.ipynb`: guided experiment with progressive detail.
- `api.py`: the intentionally small Python entry surface.
- `model.py`: the thin loaded-model handle used by both routes.

The usual workflow is deliberately short:

```python
from neutron_star_eos import open_eos

model = open_eos("my_eos.csv", kind="csv")
star = model.solve_star(central_pressure_mev_fm3=100.0)
sequence = model.solve_sequence((40.0, 60.0, 100.0, 150.0))
```

## Equation-of-state modules

| Module | One scientific responsibility |
|---|---|
| `eos/core.py` | Common cold-barotrope contract, units, domains, and provenance identity |
| `eos/analytical.py` | User-supplied analytical `P(epsilon)` and `dP/d(epsilon)` |
| `eos/tabulated.py` | Ordinary CSV nodes and their declared interpolation policy |
| `eos/validation.py` | Positivity, monotonicity, stability, and causality assessment |
| `loading.py` | Source-specific loading into an `EosModel` |
| `capabilities.py` | Plain-language report of operations that are available |
| `thermodynamic_view.py` | Read-only source and continuous series used by plots |

Top-level `analytical.py`, `tabulated.py`, and the former import names remain
small compatibility boundaries for existing user scripts.

## CompOSE modules

| Module | One scientific responsibility |
|---|---|
| `compose/reader.py` | Lossless parsing, archive members, axes, and row records |
| `compose/cold_slice.py` | Explicit cold beta-equilibrium path and density selection |
| `compose/records.py` | Immutable source records and diagnostic report types |
| `compose/profile.py` | Read-only native thermodynamic profile returned to users |
| `compose/optional_fields.py` | Composition, microscopic, and phase fields when supplied |
| `compose/thermodynamics.py` | Native-Q interpolation and reconstructed quantities |
| `compose/diagnostics.py` | Named residual calculations; never source repair |
| `compose/barotrope.py` | Continuous `P(nB)` and `epsilon(nB)` interpolation |
| `compose/validation.py` | Assessment of the continuous CompOSE interpolant |
| `compose/construction.py` | Explicit ordering/density policy and barotrope construction |
| `compose/mass_radius.py` | Independent `eos.mr` reference reader, never solver input |

`compose/dataset.py` is now only a compatibility re-export. This separation
keeps parsing, native thermodynamics, and the optional stellar reduction visibly
different.

## Stellar modules

| Module | Physical calculation |
|---|---|
| `stellar/constants.py` | Physical constants and unit conversions with authority |
| `stellar/configuration.py` | Solver tolerances and integration-domain settings |
| `stellar/tov.py` | The TOV differential equations and radial integration |
| `stellar/star.py` | Validation plus orchestration of one stellar background |
| `stellar/sequence.py` | Repeated one-star calculations over central pressure |
| `stellar/results.py` | Star, sequence, attempt, and diagnostic result records |

If you want to read the actual relativistic stellar equations, begin with
`stellar/tov.py`. No facade, plotting, output, or campaign code changes those
equations.

## Plotting and output

| Module | Responsibility |
|---|---|
| `plotting/eos.py` | Pressure-energy and sound-speed plots |
| `plotting/compose.py` | CompOSE residual, composition, and phase plots |
| `plotting/stellar.py` | Profiles, sequence status, and mass-radius plots |
| `plotting/common.py` | Shared style and axes helpers only |
| `output/metadata.py` | JSON reports and human model summaries |
| `output/tables.py` | Thermodynamic and sequence CSV tables |
| `output/bundles.py` | Atomic, non-overwriting result directories |

Plotting receives existing data or results. It does not run a solver, repair an
EoS, or save unless the caller explicitly asks.

## CompOSE comparison campaign

The research campaign follows the same physical order:

```text
config.py -> acquire.py -> sampling.py -> calculate.py -> compare.py
                                                |
                                                v
                       acceptance.py -> figures.py -> report.py/provenance.py
```

`experiments/compose_comparison/run.py` is a short command entry point.
`campaign.py` runs one model and `campaign_cli.py` coordinates the selected
models. Optional `eos.mr` data enter only through `reference.py`, after the TOV
calculation.

## Invariants

- Units and total-energy-density conventions are explicit.
- Inputs are never silently sorted, clipped, repaired, smoothed, or extrapolated.
- A numerical seam is not automatically labelled a physical transition.
- Every sequence request retains a solution or an explicit failure reason.
- A positive-pressure source boundary is not called a vacuum surface.
- Tidal observables, stability, turning points, and maximum masses are not
  inferred by the background solver.
