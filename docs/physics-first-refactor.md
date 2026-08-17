# Physics-first refactor plan

This refactor reorganizes the toolkit without changing its equations, numerical
policies, defaults, result schemas, or public behaviour.  The intended reader
is a physicist who should be able to locate each calculation by its scientific
purpose.

## Baseline

- Branch point: merge commit `58080839cd97a4fca23e28384cd852e5714a6912`.
- Test suite: 103 tests passed.  One pre-existing notebook structure test
  failed because an interactive local kernel selection changed the notebook's
  portable `python3` metadata; the full headless notebook execution still
  passed.
- Representative one-star and three-pressure sequence values were recorded for
  an analytical polytrope, the bundled CSV, PCP(BSk26), and RG(SLy4).
- BSk26 uses its already-declared diagnostic monotone-subsequence policy and
  background-diagnostic stellar validation.  SLy4 uses the strict route.

## Dependency order

1. **Stellar physics:** separate physical constants and configuration, result
   records, TOV integration, one-star orchestration, and sequence orchestration.
2. **Plotting:** separate general EoS, CompOSE diagnostic, and stellar plots
   while keeping a compatibility import surface.
3. **EoS and CompOSE:** separate interfaces, analytical input, validation,
   source reading, cold-slice selection, native thermodynamics, diagnostics,
   and barotrope construction.
4. **Facade and output:** keep the public model as a thin coordinator and split
   tables, metadata, and bundle writing.
5. **Comparison campaign:** separate sampling/calculation, comparisons,
   acceptance, figures, reporting, and provenance; leave `run.py` as the
   readable entry point.
6. **Beginner workflows:** add short, editable scripts for common physical
   tasks.
7. **Notebook and documentation:** present the same workflow progressively,
   keeping advanced settings optional and preserving headless execution.

## Regression risks and controls

- **Floating-point drift:** move calculations mechanically and compare the
  recorded analytical, CSV, BSk26, and SLy4 results after each relevant phase.
- **Import breakage:** retain existing package-level and historical module
  imports through explicit re-exports and compatibility tests.
- **Circular imports:** keep low-level constants, configurations, protocols,
  and result records independent from orchestration layers.
- **Serialization drift:** leave result dataclasses and their `to_dict`
  contracts unchanged, then exercise existing bundle tests.
- **CompOSE policy drift:** do not alter source selection, interpolation,
  diagnostic ordering reductions, or validation gates.
- **Notebook portability:** commit a portable kernel name while documenting how
  a local project kernel is selected; execute every cell in CI-style headless
  mode.
- **Over-fragmentation:** split by scientific responsibility, not by arbitrary
  line count.  Small compatibility files are allowed only at public boundaries.

No GitHub push, merge, or publication is part of this refactor.

## Completed outcome

- The former `stellar.py` (781 lines), `plotting.py` (1,190 lines), `model.py`
  (835 lines), and campaign `run.py` (3,134 lines) are now task-focused
  packages/modules. The largest library module is 478 lines; the largest
  campaign module is 404 lines; `run.py` is 68 lines.
- Six beginner workflows are 39–63 lines each and execute successfully from
  top to bottom.
- The notebook now has ordered quick-start, domain, validation, one-star,
  sequence, advanced, and saving sections. Its committed kernel is portable
  and its default run completes headlessly with `EOS_NOTEBOOK_EXECUTION_OK`.
- Stale checkpoint copies were removed from the repository tree.

## Numerical regression evidence

The following one-star results use central pressure 100 MeV/fm³. Every value
after the structural refactor is bit-for-bit equal to its recorded baseline:

| Input | Mass [Msun] | Source-boundary radius [km] | Difference from baseline |
|---|---:|---:|---:|
| analytical `P(epsilon)` | 3.6681122484640776 | 22.23259039694679 | 0 |
| bundled CSV | 3.6681122484640802 | 22.232590396946794 | 0 |
| PCP(BSk26) | 1.5343451701245925 | 11.725051080626251 | 0 |
| RG(SLy4) | 1.4826569615509821 | 11.632725533518212 | 0 |

The requested-pressure sequences also retain their exact results and statuses.
For BSk26 and SLy4, pressures `(10, 50, 100)` MeV/fm³ remain three solved
stars. For the deliberately broad analytical/CSV demonstration, the 10
MeV/fm³ attempt still reaches the declared radius ceiling and the 50/100
MeV/fm³ attempts retain their baseline masses and radii. No tolerance was
needed: the observed numerical drift is zero.
