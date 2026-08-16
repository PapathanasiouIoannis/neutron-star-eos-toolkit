# Cold CompOSE input

The toolkit reads a CompOSE directory or ZIP directly. It does not extract,
rewrite, sort, repair, extrapolate, or splice the source.

## Supported subset

Version 1 requires:

- one temperature point at `T=0`;
- one charge-fraction point;
- a one-dimensional baryon-density path;
- beta-equilibrated stellar matter;
- leptons, confirmed both by the caller and by the CompOSE lepton flag;
- a continuous, strictly invertible pressure--energy-density relation.

The required files are `eos.t`, `eos.nb`, `eos.yq`, and `eos.thermo`.
`eos.compo` is optional. The parser follows the CompOSE Reference Manual
v3.01, sections 4.2.1--4.2.3 and 4.2.7.

```powershell
eos-tool validate compose path\to\eos.zip `
  --model-id "catalogue model" `
  --source-url "https://compose.obspm.fr/eos/..." `
  --includes-leptons
```

If a source extends beyond the domain you intend to assess, select the upper
baryon density explicitly:

```text
--baryon-density-max-fm3 1.088
```

This is a declared domain choice, not an automatically inferred causal root.
The original file hashes, source row count, and retained row count remain in
the returned provenance.

## Checks

The loader checks the CompOSE index mapping, payload widths, neutron/proton
mass header, lepton flag, `P=n_B Q1`, `mu_B=m_n(1+Q3)`,
`epsilon=n_B m_n(1+Q7)`, cold Euler closure, `Q5=0`, and `Q6=Q7` at zero
temperature. Phase codes are retained as diagnostics but are never converted
automatically into physical density jumps.

Finite-temperature, multidimensional, plateau, jump, or otherwise unsupported
tables fail closed with a specific error.
