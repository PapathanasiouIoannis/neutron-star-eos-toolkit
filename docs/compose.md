# Native CompOSE thermodynamics and optional cold barotropes

The toolkit reads a CompOSE directory or ZIP directly. It does not rewrite,
sort, repair, extrapolate, or splice the source. Dataset parsing, native-field
thermodynamics, and construction of an optional stellar barotrope are separate
operations.

The shortest supported workflow is:

```python
from neutron_star_eos import open_eos

model = open_eos(
    "eos.zip",
    kind="compose",
    model_id="catalogue model",
    source_url="https://compose.obspm.fr/eos/...",
    includes_leptons=True,
)
print(model.summary())
profile = model.native_thermodynamics
```

The remainder of this guide documents the advanced lower-level objects behind
that facade.

## Three layers

`load_compose_dataset` preserves:

- all seven standard thermodynamic `Q` quantities;
- every row's `Nadd` values;
- model-specific phase codes and the un-interpreted composition payload;
- optional `eos.micro`, `eos.init`, and `eos.mr` bytes;
- source-file sizes and SHA-256 hashes.

The source can therefore be inspected even when it cannot be represented as
one continuous `P(epsilon)` relation.

`interpolate_compose_thermodynamics` implements the official first-order cold
one-dimensional workflow. It interpolates every native Q1--Q7 field separately
and linearly in raw baryon density. It then reconstructs pressure, energy and
free energy, chemical potentials, enthalpy, Gibbs energy, derivatives,
compressibilities and closure residuals. Source nodes are included by default,
no extrapolation is allowed, and warnings remain visible rather than becoming
automatic row deletion.

Three sound-speed routes are reported side by side:

- the derivative of the reconstructed `P(epsilon)` curve;
- the CompOSE thermodynamic route `(dP/dnB)/H`;
- the cold-beta route `nB/muB * dmuB/dnB`.

Their differences are consistency diagnostics because CompOSE stores redundant
quantities and interpolates them independently. In the official cold branch,
CompOSE sets the quantity named `Gamma=cP/cV` to the convention `1`; a one-point
T=0 axis cannot independently evaluate `cP` or `cV`. This is not the barotropic
adiabatic index, which is reported separately.

`build_compose_eos` is a third, optional reduction for stellar backgrounds. It
constructs pressure and total energy density as separate log-PCHIP functions of
native baryon density. This requires a continuous invertible path and may be
unavailable even though the dataset and native thermodynamic profile are fully
usable.

## Evidence first, capability second

The current cold native-Q path requires:

- one unambiguous temperature point at `T=0` (other temperature slices may be
  present in the parsed dataset);
- one charge-fraction point equal to the CompOSE cold-beta sentinel `Yq=0`;
- a one-dimensional baryon-density path;
- beta-equilibrated stellar matter;
- leptons, confirmed both by the caller and by the CompOSE lepton flag;
- an explicitly selected density path.

The raw path is always preserved and reportable. A finite `P=0` source node is
also retained by the native profile, while it explicitly blocks the optional
positive-pressure logarithmic barotrope. Strict continuous stellar construction
requires positive, increasing baryon density, pressure, and total energy
density; its separate physics gate assesses mechanical stability and causality.
A reversal, plateau, diagnostic threshold crossing, or acausal tail is therefore
an assessment result - not a reason to hide the source from native plots or
reports.

The required files are `eos.t`, `eos.nb`, `eos.yq`, and `eos.thermo`.
`eos.compo` is optional. The source layer follows the CompOSE Reference Manual
v3.01, sections 4.2.1--4.2.9 and Appendix A. The native workflow structurally
parses `eos.compo` and `eos.micro`; `eos.init` remains preserved as opaque source
bytes. Optional `eos.mr` data can be opened explicitly as an independent
reference with `load_compose_mass_radius_reference(dataset)`. Only its standard
first two columns (radius in km and gravitational mass in solar masses) are
interpreted; later columns remain source-defined and must be identified from the
model data sheet. Reference points are never used as input to the TOV solver.

The continuous CompOSE adapter also exposes
`pressure_from_baryon_density(...)` and
`baryon_density_from_pressure(...)`. Both enforce the selected native-density
domain and never extrapolate; this supports explicit central-density grids
without reaching into private interpolation state.

```powershell
eos-tool inspect path\to\eos.zip --kind compose `
  --model-id "catalogue model" `
  --source-url "https://compose.obspm.fr/eos/..." `
  --includes-leptons
```

If a source extends beyond the domain you intend to assess, select its source
nodes explicitly:

```text
--baryon-density-max-fm3 1.088
```

An explicit lower selection is also available as
`--baryon-density-min-fm3`. These are declared source-domain choices, not an
automatically inferred causal root or an automatic repair of a seam.
The original file hashes, source row count, and retained row count remain in
the returned provenance.

The same density selection is applied to the native profile and optional
barotrope. The CLI returns success when parsing, cold-path selection and native
thermodynamics succeed. Add `--require-barotrope` only when a calling workflow
must also require a continuous barotrope and its physics gate.

`--native-points` is the base geometric sample count. Exact source nodes are
unioned into that grid by default, so the final point count can be larger. The
final grid size, bounds, construction policy and float64 SHA-256 are recorded in
the native-profile provenance.

For sensitivity work, a user may explicitly request:

```text
--ordering-policy diagnostic_monotone_subsequence
```

This keep-first reduction does not alter any retained value. The companion
`diagnostic_keep_later_monotone_subsequence` policy preferentially retains the
later conflicting row when doing so remains monotone relative to the preceding
source row. Both record every omitted source position. Their difference brackets
the sensitivity to a local seam; neither is a Maxwell construction,
discontinuity prescription, or validation pass for the original path.

## Status layers

The dataset parser checks the CompOSE index mapping, payload widths, neutron/proton
mass header, lepton flag, `P=n_B Q1`, `mu_B=m_n(1+Q3)`,
and `epsilon=n_B m_n(1+Q7)`.

Cold Euler closure, `Q5=0`, and `Q6=Q7` at zero temperature are retained as
source diagnostics. CompOSE tabulates redundant quantities at finite precision,
so a diagnostic threshold crossing is visible but is not automatically treated
as corruption of the independently supplied pressure--energy-density path.

The CLI reports independently:

1. dataset parsing;
2. cold-slice eligibility and source diagnostics;
3. native-Q thermodynamic availability and diagnostics;
4. continuous-barotrope availability;
5. barotrope mechanical stability and causality.

Phase codes and ordering seams are never converted automatically into physical
density jumps. The strict path remains unavailable when it is nonmonotonic, but
the parsed data, maximal monotone source blocks, and an explicitly requested
diagnostic reduction remain inspectable side by side.

## Advanced Python API

```python
from neutron_star_eos import (
    build_compose_eos,
    interpolate_compose_thermodynamics,
    load_compose_dataset,
    solve_star,
)

dataset = load_compose_dataset(
    "eos.zip",
    model_id="catalogue model",
    source_url="https://compose.obspm.fr/eos/...",
)
cold = dataset.cold_beta_equilibrium_slice(
    matter="cold_beta_equilibrated",
    includes_leptons=True,
)
print(cold.report().status)
print(cold.continuous_segments())

# Native CompOSE order-1 workflow. This remains available even if the
# optional stellar barotrope below is blocked by a pressure reversal.
profile = interpolate_compose_thermodynamics(cold, points=2001)
print(profile.summary())
pressure = profile.column("pressure_mev_fm3")
epsilon = profile.column("energy_density_mev_fm3")
cs2_compose = profile.column("sound_speed_squared_compose_thermodynamic")

# Optional continuous stellar-background adapter.
eos = build_compose_eos(
    cold,
    baryon_density_max_fm3=1.088,
)
```

The profile also exposes all Q fields and Q derivatives, reconstructed
thermodynamic potentials, compressibilities, the barotropic adiabatic index,
composition pairs, raw Aav/Zav/Yav plus calculated Nav, model-specific phase
codes, `Nadd` values, and optional `eos.micro` quantities. Missing optional data
remain NaN with explicit availability columns; they are not changed to physical
zeros. Quantities that require a temperature dimension are marked not
applicable for a one-point T=0 path.

The ordinary solver remains strict by default. Background-only diagnostics can
be requested explicitly when the only whole-domain findings are causality or a
sound-speed-sign finding and the positive `epsilon(P)` relation remains
invertible:

```python
diagnostic_eos = build_compose_eos(
    cold,
    ordering_policy="diagnostic_monotone_subsequence",
)
star = solve_star(
    diagnostic_eos,
    central_pressure_mev_fm3=100.0,
    validation_mode="background_diagnostic",
)
print(star.eos_validation_status, star.eos_validation_issues)
```

The returned result retains the failed EoS issue codes. This mode supports only
background sensitivity. It does not authorize tides, a stable branch, a maximum
mass, or a microscopic interpretation of an omitted seam.

Finite-temperature data can be parsed and preserved, but the present stellar
reduction requires an explicit `T=0` slice. Genuine plateau and jump handling
still needs a physical transition policy. Diagnostic omission is always visibly
labelled and never substitutes for that policy.
