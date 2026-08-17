# Reproducibility

The toolkit records the supplied source, declared policies, software versions,
validation results, solver configuration, and an EoS provenance identity. It
does not claim that a hash alone proves physical correctness.

## Result bundles

Commands print by default and write only when `--output NEW_DIRECTORY` is
requested. An existing directory is never overwritten. Inspection, star, and
sequence bundles contain the model report; stellar bundles also contain the
exact solver settings, physical conversion constants and their authority, and
the EoS provenance identity. Sequence bundles retain every attempted central
pressure and failure reason.

## Analytical definitions

Evaluated analytical behavior is fingerprinted on declared grids. That does
not identify the callable's source text, imported helper files, interpreter
state, or external resources.

The experiment notebook therefore treats `notebooks/analytical_eos.py` as a
self-contained authoritative definition. It records both the raw file SHA-256
and a UTF-8/LF-normalized SHA-256, incorporates the normalized identity into the
model source description, and can copy the exact definition into an explicitly
saved experiment bundle.

If a user definition imports additional local modules, those files must be
recorded separately; the single-file notebook identity does not cover them.

## Environments

CPython 3.12 with the versions in
[`constraints/verified-py312.txt`](../constraints/verified-py312.txt) is the
verified environment. Normal package metadata and the constraints file must be
reviewed together before changing numerical dependencies.

For a retained experiment, record:

- repository commit;
- dirty-worktree status;
- input source and hash;
- model report and diagnostic codes;
- Python, NumPy, SciPy, and toolkit versions;
- the physical conversion constants and named authority used by the stellar
  equations;
- all stellar and sampling parameters;
- every successful and unsuccessful requested result;
- the exact analytical definition file when applicable.

Tracked research campaigns should additionally hash the runner, acquisition
script, registry, and scientific package sources. A commit plus a dirty flag is
not enough to reconstruct calculations made from an uncommitted worktree.
