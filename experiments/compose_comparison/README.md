# Cold CompOSE comparison campaign

This directory defines a reproducible nine-model experiment without placing
downloaded tables or generated scientific artifacts under version control. The
eight core models span nucleonic, relativistic, non-relativistic, hyperonic,
unified, and crust-matched constructions. QHC19-C is an additional hybrid
crossover stress test.

The tracked files here do not contain scientific results. In particular,
`eos.mr` is an independent reference for results calculated from the tabulated
thermodynamics; it is never used as input to the toolkit's TOV equations.

## Model matrix

The exact archive URLs, byte counts, SHA-256 digests, catalogue benchmarks,
model-specific warnings, and primary references live in
[`config/models.json`](config/models.json).

| Role | Slug | CompOSE model | Catalogue ID | Mmax [Msun] | R(Mmax) [km] | R1.4 [km] |
|---|---|---|---:|---:|---:|---:|
| core | `bsk26` | PCP(BSk26) | 258 | 2.17 | 10.20 | 11.77 |
| core | `sly4` | RG(SLy4) | 134 | 2.06 | 10.02 | 11.70 |
| core | `dd2` | GPPVA(DD2) | 217 | 2.42 | 11.95 | 13.19 |
| core | `fsu2h` | GPPVA(FSU2H) | 213 | 2.37 | 12.43 | 13.29 |
| core | `tw` | GPPVA(TW) | 219 | 2.07 | 10.70 | 12.33 |
| core | `ddme_x` | XMLSLZ(DDME-X) | 281 | 2.56 | 12.36 | 13.37 |
| core | `gm1y6` | OPGR(GM1Y6) | 66 | 2.29 | 12.13 | 13.78 |
| core | `apr` | APR(APR) | 68 | 2.19 | 9.97 | 11.37 |
| stress | `qhc19_c` | BFH(QHC19-C) | 151 | 2.18 | 10.80 | 11.60 |

These are current CompOSE catalogue-page values, not outputs from this toolkit.
Crust, causality, surface, and model-version conventions differ, so the
registry's notes and citations must accompany comparisons.

## Acquire or verify the inputs

From the repository root, download every missing pinned archive and verify all
existing ones:

```powershell
.\.venv\Scripts\python.exe experiments\compose_comparison\acquire.py
```

Acquire a chosen ordered subset:

```powershell
.\.venv\Scripts\python.exe experiments\compose_comparison\acquire.py `
  --model bsk26 --model sly4 --model apr
```

After one successful acquisition, reproduce the verification with network
access forbidden:

```powershell
.\.venv\Scripts\python.exe experiments\compose_comparison\acquire.py --offline
```

Acquisition is fail-closed:

- each source is pinned to a stable Zenodo record URL, byte count, and SHA-256;
- an existing archive is reused only when its bytes, hash, ZIP CRC, required
  members, and declared optional members all match;
- an invalid existing archive is never silently overwritten;
- a missing archive fails immediately in offline mode;
- downloads are staged in the destination directory and published only after
  complete verification;
- `download.json` contains deterministic source and verification provenance,
  without a run-time timestamp.

The model registry and scientific-result contracts are version 2. The
acquisition sidecar remains `compose-comparison-acquisition-v1` because its
archive-provenance contract did not change; the runner explicitly validates
that version during preflight, so the pinned existing sidecars remain valid.

The live CompOSE download endpoints previously returned archives whose bytes
did not agree with their adjacent checksum text files. Stable Zenodo records
are therefore the acquisition authority for this campaign. The current
CompOSE pages remain the authority for model identity and catalogue benchmark
values. Embedded descriptive files are retained as upstream provenance and may
carry registry-declared non-gating metadata findings.

## Local directory contract

The acquisition command creates or verifies this layout:

```text
experiments/compose_comparison/
├── config/models.json             tracked model and benchmark registry
├── acquire.py                     tracked acquisition/verification command
├── README.md                      tracked campaign contract
├── data/
│   ├── raw/<slug>/
│   │   ├── archive.zip            ignored upstream input
│   │   └── download.json          ignored deterministic verification record
│   └── derived/<slug>/             ignored calculated tables and reports
├── figures/<slug>/                 ignored one-plot-per-PNG outputs
├── figures/comparison/             ignored cross-model PNG outputs
└── results/                        ignored campaign summaries and acceptance data
```

`data/raw`, `data/derived`, `figures`, and `results` are intentionally ignored.
The repository's MIT license applies to the toolkit code, not as a
relicensing of the upstream CompOSE/Zenodo datasets. Preserve each record's
source attribution and license when sharing data outside this local campaign.

## Run the calculation and validation campaign

Install the plotting extra, then run all nine models from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[plot]"
.\.venv\Scripts\python.exe experiments\compose_comparison\run.py
```

For a shorter development pass or a selected subset:

```powershell
.\.venv\Scripts\python.exe experiments\compose_comparison\run.py `
  --quick --model bsk26 --model sly4
```

The full runner:

1. Inspects the lossless native cold, beta-equilibrated path before attempting
   any continuous barotrope.
2. Saves native thermodynamics, selected source nodes, and continuous
   barotrope evaluations as separate data products.
3. Starts with the strict source ordering. If a model has a documented local
   seam conflict, retain the strict failure and run explicitly named
   keep-first and keep-later diagnostic sensitivity cases. The registry states
   which reduction is the primary conditional convention and why; neither is
   treated as a physical transition construction.
4. Calculates all primary mass-radius sequences independently with the TOV
   solver. Never replace a calculated curve with `eos.mr`.
5. Cross-checks calculated pre-peak, increasing-central-density samples against
   the corresponding pre-peak side selected from `eos.mr` when it exists. This
   is an explicit numerical heuristic, not a stability inference. It then
   compares peak coordinates and fixed-mass radii with the current CompOSE
   benchmark and convention-aware primary references in the registry. A generic
   triangle diagnostic classifies consistency among the calculated TOV result,
   catalogue benchmark, and optional `eos.mr`; the reference never becomes an
   acceptance gate.
6. Reports hydrostatic sampled peaks separately from the numerically verified
   `c_s^2=1` threshold whenever the high-density source becomes acausal.
7. Quantifies sensitivity between two declared positive source-node boundaries;
   this does not estimate the omitted layer from the lowest positive pressure to
   `P=0`, and it never calls the reported radii vacuum-surface radii.
8. Retains every requested central point, including structured failures.
9. Writes every plot to its own PNG. Combined multi-panel PNGs do not satisfy
   the campaign artifact contract.

Initial comparison tolerances are `max(0.01 Msun, 0.5%)` for a sampled peak
mass and 0.15 km for catalogue radii. RG(SLy4) may use a provisional 0.25 km
radius tolerance until its source-boundary truncation is explicitly measured.
Exceeding a tolerance triggers numerical, interpolation, domain, and surface
investigation; it never triggers physics tuning or silent data repair.

Optional `eos.mr` consistency uses a separate 0.15 km materiality threshold,
which is a diagnostic classification threshold rather than an uncertainty or
acceptance tolerance. `material_optional_reference_source_inconsistency` is
assigned only when the independent TOV calculation passes the current CompOSE
catalogue gate and both the absolute catalogue-minus-`eos.mr` and exact
TOV-minus-`eos.mr` R1.4 differences exceed that threshold. Any material
diagnostic without that complete corroborating triangle is classified as
`material_optional_reference_discrepancy_unattributed`; no individual source
vertex is blamed. A failed catalogue calculation is instead
`indeterminate_calculation_catalogue_disagreement`; missing references and
empty comparison domains have explicit states. Comparison coverage is an
acceptance gate; numerical agreement with optional `eos.mr` is not. The FSU2H
and TW source entries satisfy the full attribution rule and currently trigger
the material source-inconsistency state. Their cause is undocumented and is not
diagnosed as solver failure.

The APR registry also pins a non-gating provenance finding to the exact
embedded `eos.pdf` member. That archived document reports older stellar
benchmarks (2.17 Msun, 10.27 km, 11.33 km) than the current catalogue page
(2.19 Msun, 9.97 km, 11.37 km). The member hash is verified at run time; the
document is never solver input or acceptance authority.

For a registry-declared ordering seam, both diagnostic reductions must complete
without sequence failures, cover 1.4 Msun, bracket the sampled peak, avoid
significant pre-peak mass decreases, and independently satisfy those same
predeclared CompOSE catalogue tolerances. The 0.001 Msun peak-mass and 0.01 km
R1.4 differences between reductions are classification thresholds only, not
acceptance tolerances. A larger delta is saved as a material conditional
source-seam span and remains prominent in `summary.json`, `acceptance.json`, and
the report. Campaign acceptance therefore does not certify a unique crust-core
radius or a Maxwell construction.

The local `results/report.md` and `results/all_models_summary.csv` are the
short entry points after a run. Per-model `summary.json` files retain the full
acceptance evidence, while `manifest.json` hashes every input and generated
artifact. The runner exits nonzero if any catalogue, convergence, ordering,
sequence-coverage, peak-bracketing, required-plot, or literature-comparison
acceptance check fails.

`acceptance.json` keeps the numerical `passed` boolean separate from
`interpretive_status`. A numerically accepted campaign that contains a material
attributed optional-reference or archive-metadata finding reports
`PASS_WITH_MATERIAL_UPSTREAM_REFERENCE_FINDINGS` without converting the
upstream diagnostic into a solver failure. A future neutral material diagnostic
uses `PASS_WITH_MATERIAL_UNATTRIBUTED_DIAGNOSTIC_FINDINGS`; if both classes are
present, the status names both.

For GM1Y6, the bundled data sheet assigns the Douchin-Haensel crust below
`nB=1e-3 fm^-3`, so the later core-side row is the primary conditional
reduction. The keep-first result is still mandatory. Their radius difference is
reported as a source-construction systematic rather than hidden by a
model-specific tolerance.

## Scaffold tests

The acquisition tests use only temporary synthetic ZIP files and mocked
responses. They do not access the network or alter the saved real archives:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_compose_experiment -v
```
