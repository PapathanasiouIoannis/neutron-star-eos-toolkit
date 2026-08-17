# Plotting

The optional plotting layer presents already-loaded thermodynamics and
already-computed stellar results. It never runs a stellar solver, changes
source order, repairs data, smooths curves, extrapolates, shows a window, or
writes a file.

Install it with:

```text
python -m pip install -e ".[plot]"
```

Every plotting function accepts an optional Matplotlib `Axes` and returns the
axes it used. This makes notebook and publication layouts explicit:

```python
import matplotlib.pyplot as plt

from neutron_star_eos.plotting import (
    plot_pressure_energy,
    plot_sound_speed_squared,
)

figure, axes = plt.subplots(1, 2, constrained_layout=True)
plot_pressure_energy(model, ax=axes[0])
plot_sound_speed_squared(model, ax=axes[1])
```

## Thermodynamic plots

- `plot_pressure_energy` distinguishes source nodes, native CompOSE
  thermodynamics, and an optional continuous stellar barotrope.
- `plot_sound_speed_squared` marks the mechanical boundary `c_s^2 = 0` and
  causal boundary `c_s^2 = 1`. CompOSE sound-speed definitions remain separate.
- `plot_compose_closure_residuals` presents pressure/energy/Gibbs closure
  magnitudes on a logarithmic scale; exact zeros are disclosed and omitted only
  from the rendering.
- `plot_compose_free_energy_closure_residuals` gives the free-energy closures
  their own focused axes instead of overplotting every diagnostic together.
- `plot_compose_cold_residuals` presents the cold/beta-equilibrium residuals.
- `plot_composition` preserves missing coverage as `NaN`; it names the verified
  standard electron, muon, neutron, and proton codes while retaining each code,
  marks unknown codes as source-defined, and does not label every
  composition-like field as a fraction.
- `plot_phase_codes` displays source-defined codes without inventing physical
  labels.

Zero or negative native pressure remains visible using an explicit axis policy;
it is never silently discarded to make a logarithmic plot possible.

## Stellar plots

- `plot_mass_profile` presents the retained enclosed-mass profile and marks its
  positive-pressure source boundary.
- `plot_mass_radius` presents solved sequence attempts as source-boundary
  masses and radii. It does not infer stability or maximum mass.
- `plot_sequence_status` retains every requested central pressure and its
  solved/unavailable status.

Sequence curves are never connected across an unavailable attempt. A star must
have been solved with `retain_profile=True` before its mass profile is available.

## Export

Matplotlib owns export so the caller chooses the destination and format:

```python
figure.savefig("eos.svg", metadata={"Title": model.model_name})
figure.savefig("eos.png", dpi=300)
```

SVG or PDF is preferred for line art. PNG at 300 DPI is suitable for previews.
Every caption should state that stellar radii stop at the EoS lower-pressure
boundary and are not claimed vacuum-surface radii.
