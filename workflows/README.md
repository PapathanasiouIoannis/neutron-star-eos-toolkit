# Beginner workflows

These scripts expose one physical task at a time. Start at the top, edit the
clearly marked parameter block, then run the whole file from the repository
root:

```powershell
uv run python workflows\calculate_one_star.py
```

| Script | What it does |
|---|---|
| `analytical_eos.py` | Defines a complete analytical `P(epsilon)` and consistent `dP/d(epsilon)` |
| `csv_eos.py` | Loads and describes an ordinary pressure–energy-density table |
| `compose_eos.py` | Loads one locally acquired cold CompOSE archive |
| `calculate_one_star.py` | Integrates the TOV equations at one central pressure |
| `calculate_mass_radius.py` | Integrates an explicit central-pressure sequence and saves one PNG |
| `compare_equations_of_state.py` | Calculates and overlays two mass–radius sequences |

The plotting scripts save PNGs beside themselves. They never download data or
overwrite scientific result bundles. For explanations beside every result, use
[`notebooks/eos_experiments.ipynb`](../notebooks/eos_experiments.ipynb).
