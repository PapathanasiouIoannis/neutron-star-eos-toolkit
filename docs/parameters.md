# Parameter reference

All production-facing pressures and total energy densities use `MeV/fm^3`.
Baryon number density uses `fm^-3`, radius uses km, and mass uses solar masses.

## Input parameters

### CSV

| Parameter | Meaning |
|---|---|
| `epsilon_column` | Total energy density including rest mass |
| `pressure_column` | Pressure in the same row order |
| `baryon_density_column` | Optional positive, strictly increasing baryon density |
| `name` | Optional model label |
| `source_description` | Optional provenance description |

Rows must already be finite, positive, and strictly increasing. The toolkit
does not sort, deduplicate, clip, repair, or extrapolate them.

### Analytical callable

`EosModel.from_analytical` requires an authoritative `P(epsilon)` callable, a
consistent `dP/d(epsilon)` callable, a finite energy-density domain, a model
name, and a source description. An explicit inverse `epsilon(P)` is optional;
otherwise a bracketed inverse is evaluated inside the declared domain.

The callable is the model. Coefficients such as `K` or `gamma` belong only to
particular example functions and are not toolkit controls.

### CompOSE

| Parameter | Meaning |
|---|---|
| `model_id` | Human-readable source identity |
| `source_url` | Catalogue or archive provenance |
| `includes_leptons` | Explicit physical declaration; never inferred |
| `baryon_density_min_fm3` | Optional first retained native density |
| `baryon_density_max_fm3` | Optional last retained native density |
| `native_points` | Base native-Q assessment sample count |
| `ordering_policy` | Strict default or explicitly diagnostic reduction |

`native_points` changes the assessment grid, not source values. A diagnostic
ordering policy records omissions and is not a phase-transition model.

## Stellar configuration

| Field | Default | Meaning |
|---|---:|---|
| `radius_start_km` | `1e-4` | Positive initial radius for the centre expansion |
| `radius_max_km` | `25` | Integration ceiling, not a physical surface |
| `center_expansion_limit_km` | `1e-4` | Upper validity scale for the centre expansion |
| `ode_rtol` | `1e-10` | Relative ODE tolerance |
| `ode_atol` | `1e-12` | Absolute ODE tolerance |
| `profile_points` | `300` | Retained mass-profile points when requested |

All scale and tolerance values must be positive. `profile_points` must be an
integer of at least two. If the pressure boundary is not reached before
`radius_max_km`, the result is unavailable with reason code
`radius_limit_reached`.

## Stars and sequences

`central_pressure_mev_fm3` must be strictly above the EoS lower pressure and no
greater than its upper pressure. An explicit sequence must be finite, strictly
increasing, and remain inside that interval. The automatic sequence uses a
geometric grid and requires at least nine integer points.

`validation_mode="strict"` is the default. The expert
`background_diagnostic` mode retains named causality or mechanical findings;
it never upgrades the EoS to a physical validation pass.
