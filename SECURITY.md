# Security policy

## Supported versions

The neutron-star EoS toolkit is currently in pre-release development. No
version has yet been published with a public security-support commitment.
Security fixes on the active development branch are handled on a best-effort
basis until a supported release policy is declared.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue, discussion,
notebook, or result bundle.

Use GitHub's private vulnerability-reporting or security-advisory interface for
this repository if it is available. Otherwise, contact the repository owner
privately through the GitHub account
[@PapathanasiouIoannis](https://github.com/PapathanasiouIoannis) and request a
private channel before sending sensitive details.

Include, where possible:

- the affected commit or development version;
- the operating system and Python version;
- a minimal reproduction that does not contain private scientific data;
- the expected and observed behavior;
- the potential impact; and
- any known workaround.

Receipt and remediation times are not guaranteed during the pre-release phase.
The owner will coordinate disclosure after the report has been assessed and an
appropriate fix is available.

## Security versus scientific correctness

Incorrect units, numerical results, validation behavior, or scientific claims
are important defects and should be reported through the normal issue workflow.
Treat a defect as a security report when it can also expose or alter data,
execute unintended code, escape an intended filesystem location, consume
resources in a way that affects service availability, or cross another trust
boundary.

Never attach proprietary CompOSE tables, credentials, unpublished research
data, or identifiable machine paths to a report unless the owner has provided
an approved private channel.
